# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ascend transfer adapter using PyTorch's supported tensor copy API.

This is a synchronous correctness-first adapter, not an asynchronous DMA kernel.
It preserves the host's block/group mapping and reports completed transfers only
once device work has finished. It never replaces CUDA or XPU handlers globally.
"""

from __future__ import annotations

import time

import torch
from vllm.v1.kv_offload.base import OffloadingWorker, TransferResult


def copy_plan(gpu_spec, cpu_spec, refs, factor):
    """Resolve canonical (tensor, device block, CPU block, offset, bytes) spans."""
    if factor < 1 or len(gpu_spec.group_sizes) != len(refs):
        raise ValueError("invalid block factor or KV group count")
    gpu_offset = cpu_offset = 0
    plan = []
    for size, logical, group in zip(
        gpu_spec.group_sizes, gpu_spec.block_indices, refs, strict=True
    ):
        if size < 0 or logical < 0:
            raise ValueError("negative group size or logical block index")
        if size == 0:
            continue
        skip = logical % factor
        count = (size + skip + factor - 1) // factor
        if gpu_offset + size > len(gpu_spec.block_ids) or cpu_offset + count > len(
            cpu_spec.block_ids
        ):
            raise ValueError("block list is shorter than its group metadata")
        for ref in group:
            for index in range(size):
                cpu_index, sub_block = divmod(skip + index, factor)
                plan.append(
                    (
                        ref.tensor_idx,
                        int(gpu_spec.block_ids[gpu_offset + index]),
                        int(cpu_spec.block_ids[cpu_offset + cpu_index]),
                        sub_block,
                        ref.page_size_bytes,
                    )
                )
        gpu_offset += size
        cpu_offset += count
    if gpu_offset != len(gpu_spec.block_ids) or cpu_offset != len(cpu_spec.block_ids):
        raise ValueError("unused blocks in transfer metadata")
    return plan


class AscendCopyWorker(OffloadingWorker):
    def __init__(self, kv_caches, block_size_factor, num_cpu_blocks, mmap_region):
        self.region = mmap_region
        self.factor = block_size_factor
        self.refs = kv_caches.group_data_refs
        self.device_tensors = []
        self.cpu_tensors = []
        self.completed = []
        self.pending = set()
        for cache in kv_caches.tensors:
            tensor = cache.tensor
            if tensor.device.type != "npu":
                raise ValueError("AscendCopyWorker requires NPU tensors")
            view = tensor.view(torch.int8).view(-1, cache.page_size_bytes)
            self.device_tensors.append(view)
            self.cpu_tensors.append(
                mmap_region.create_next_view(cache.page_size_bytes * block_size_factor)
            )
        self.device = self.device_tensors[0].device

    def _copy(self, job_id, gpu_spec, cpu_spec, store):
        if job_id in self.pending:
            raise ValueError("transfer job ID is already pending")
        plan = copy_plan(gpu_spec, cpu_spec, self.refs, self.factor)
        spans = self._spans(plan)
        start = time.monotonic()
        torch.npu.synchronize(self.device)
        for device, cpu in spans:
            destination, source = (cpu, device) if store else (device, cpu)
            destination.copy_(source, non_blocking=False)
        torch.npu.synchronize(self.device)
        self.completed.append(
            TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=sum(span[0].numel() for span in spans),
                transfer_time=time.monotonic() - start,
            )
        )
        self.pending.add(job_id)
        return True

    def _spans(self, plan):
        # Validate the entire transaction before touching any destination.
        spans = []
        for tensor_idx, device_block, cpu_block, sub_block, size in plan:
            device = self.device_tensors[tensor_idx]
            cpu = self.cpu_tensors[tensor_idx]
            page_size = device.shape[1]
            if not (
                0 <= device_block < device.shape[0] and 0 <= cpu_block < cpu.shape[0]
            ):
                raise ValueError("transfer block outside allocated cache")
            if not 0 < size <= page_size:
                raise ValueError("invalid canonical page size")
            offset = sub_block * page_size
            spans.append(
                (device[device_block, :size], cpu[cpu_block, offset : offset + size])
            )
        return spans

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._copy(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._copy(job_id, dst_spec, src_spec, False)

    def get_finished(self):
        completed, self.completed = self.completed, []
        self.pending.difference_update(item.job_id for item in completed)
        return completed

    def wait(self, job_ids):
        # submit_* is synchronous: all admitted copies are already complete.
        return None

    def shutdown(self):
        self.device_tensors.clear()
        self.cpu_tensors.clear()
        self.completed.clear()
        self.pending.clear()
        self.region.cleanup()


class AscendLayoutWorker(AscendCopyWorker):
    def __init__(self, layout, block_size_factor, num_cpu_blocks, mmap_region):
        self.region = mmap_region
        self.factor = block_size_factor
        self.refs = layout.group_data_refs
        self.layers = layout.layers
        self.pages = layout.page_sizes
        self.device_tensors = [row for _, rows in self.layers for row in rows]
        if not self.device_tensors or any(
            row.device.type != "npu" for row in self.device_tensors
        ):
            raise ValueError("AscendLayoutWorker requires NPU cache views")
        self.device = self.device_tensors[0].device
        if any(row.device != self.device for row in self.device_tensors):
            raise ValueError("cache views span multiple devices")
        self.cpu_tensors = [
            mmap_region.create_next_view(page * self.factor) for page in self.pages
        ]
        self.completed = []
        self.pending = set()

    def _spans(self, plan):
        spans = []
        for layer, device_block, cpu_block, sub_block, size in plan:
            allocation, rows = self.layers[layer]
            cpu = self.cpu_tensors[allocation]
            if not 0 <= cpu_block < cpu.shape[0]:
                raise ValueError("CPU block outside allocated cache")
            if size != sum(row.shape[1] for row in rows):
                raise ValueError("logical cache page size changed")
            offset = sub_block * self.pages[allocation]
            for row in rows:
                if not 0 <= device_block < row.shape[0]:
                    raise ValueError("device block outside allocated cache")
                end = offset + row.shape[1]
                spans.append((row[device_block], cpu[cpu_block, offset:end]))
                offset = end
        return spans

    def shutdown(self):
        self.layers.clear()
        super().shutdown()
