# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ascend transfer adapter using a dedicated NPU stream.

The previous revision used ``copy_(..., non_blocking=False)`` bracketed by
``torch.npu.synchronize()``, so ``submit_*`` blocked and ``wait()`` was a no-op.
That provided correctness but no device/transfer overlap.

This revision issues ``copy_(..., non_blocking=True)`` on a dedicated NPU
stream and records a per-job ``torch.npu.Event``. ``submit_*`` returns
immediately, ``get_finished()`` polls the events, and ``wait()`` blocks on
them. Errors surfaced at synchronize time are converted to ``success=False``
results instead of being raised out of the worker.

Buffer ownership: the worker holds long-lived *views* (not copies) into the
host mmap region and the runner's KV-cache tensors. The offloading manager
pins (ref-counts) the source/destination blocks for the duration of a job, so
those views remain valid until the job's event is observed complete. Once a
job is reported finished, the worker must not touch those views again — the
framework may reuse the blocks.
"""

from __future__ import annotations

import hashlib
import os
import time

import torch
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadingWorker, TransferResult

logger = init_logger(__name__)


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
        self.completed: list[TransferResult] = []
        self.pending: set[int] = set()
        # job_id -> (event, started_at, transfer_size, plan)
        self._inflight: dict[int, tuple[torch.npu.Event, float, int, list]] = {}
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
        # Dedicated stream: keep device<->CPU copies off the runner's stream so
        # transfers overlap with compute rather than serializing behind it.
        self._stream = torch.npu.Stream(device=self.device)

    # -- async submission -------------------------------------------------

    def _submit(self, job_id, gpu_spec, cpu_spec, store) -> bool:
        if job_id in self.pending:
            raise ValueError("transfer job ID is already pending")
        plan = copy_plan(gpu_spec, cpu_spec, self.refs, self.factor)
        spans = self._spans(plan)
        start = time.monotonic()
        if store:
            # The copy stream reads the live GPU KV cache, which the compute
            # stream keeps writing. Order this store after compute (mirroring
            # the CUDA reference handler's wait_stream barrier) so the copy
            # reads stable data instead of racing the compute kernels.
            self._stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self._stream):
            for device, cpu in spans:
                destination, source = (cpu, device) if store else (device, cpu)
                destination.copy_(source, non_blocking=True)
            event = torch.npu.Event()
            event.record(self._stream)
        size = sum(span[0].numel() for span in spans)
        self._inflight[job_id] = (event, start, size, plan)
        self.pending.add(job_id)
        return True

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, dst_spec, src_spec, False)

    # -- completion -------------------------------------------------------

    def get_finished(self):
        finished = []
        for job_id in list(self._inflight):
            event, start, size, plan = self._inflight[job_id]
            try:
                done = event.query()
                success = True
            except Exception as exc:  # noqa: BLE001 - surface as failed result
                logger.error("tiering transfer %s failed during query: %s", job_id, exc)
                done = True
                success = False
            if not done:
                continue
            # Remove before reporting so a job is never reported twice.
            del self._inflight[job_id]
            finished.append(self._finalize(job_id, success, start, size, plan))
        self.completed.extend(finished)
        completed, self.completed = self.completed, []
        return completed

    def wait(self, job_ids):
        for job_id in list(job_ids):
            entry = self._inflight.pop(job_id, None)
            if entry is None:
                continue
            event, start, size, plan = entry
            try:
                event.synchronize()
                self.completed.append(self._finalize(job_id, True, start, size, plan))
            except Exception as exc:  # noqa: BLE001 - surface as failed result
                logger.error("tiering transfer %s failed during wait: %s", job_id, exc)
                self.completed.append(self._finalize(job_id, False, start, size, plan))
            self.pending.discard(job_id)

    def _finalize(self, job_id, success, start, size, plan):
        self.pending.discard(job_id)
        if os.environ.get("HUST_TIERING_DIAGNOSTICS") == "1" and success:
            self._diagnose(job_id, plan)
        return TransferResult(
            job_id=job_id,
            success=success,
            transfer_size=size,
            transfer_time=time.monotonic() - start,
        )

    # -- helpers ----------------------------------------------------------

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

    def _diagnose(self, job_id, plan):
        # Diagnostic runs only, after the transfer has completed. Hash one
        # logical layer per group; no cache contents or request text are logged.
        first_layers = {group[0].tensor_idx for group in self.refs if group}
        seen = set()
        for layer, block, cpu_block, sub_block, size in plan:
            if layer not in first_layers or layer in seen:
                continue
            seen.add(layer)
            sampled = self._spans([(layer, block, cpu_block, sub_block, size)])
            digests = [
                hashlib.sha256(memoryview(cpu.numpy())).hexdigest()
                for _, cpu in sampled
            ]
            nonzero = [int(cpu.count_nonzero()) for _, cpu in sampled]
            logger.info(
                "TIERING_COPY job=%s layer=%s device_block=%s cpu_block=%s "
                "sha256=%s nonzero=%s",
                job_id,
                layer,
                block,
                cpu_block,
                digests,
                nonzero,
            )

    def shutdown(self):
        # Best-effort drain of in-flight work so the region is not torn down
        # while a stream copy may still be reading it.
        for job_id in list(self._inflight):
            event, _, _, _ = self._inflight[job_id]
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001, S110 - shutdown is best-effort
                pass
        self._inflight.clear()
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
        self._inflight = {}
        self._stream = torch.npu.Stream(device=self.device)

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
