# SPDX-License-Identifier: Apache-2.0
"""Ascend transfers with conservative ownership and fail-fast fences.

Copies request ``non_blocking=True`` on a dedicated NPU stream; this does not
promise actual asynchronous overlap for mmap-backed CPU memory. No runtime
host-pinning assumptions are made here.

Every job owns its exact source/destination views *before* any enqueue. Only a
successfully queried/synchronized event permits normal completion. A runtime
error is terminal: submit, poll and wait raise, never emit a failed result or
return a successful reuse fence (the frozen host only accepts success results).
A full-stream drain is attempted on failure, including copies whose event was
never recorded. Even a successful drain cannot validate a partially failed
transfer, so it permits teardown, not continued use of this worker.

``shutdown`` closes admission immediately and discards undelivered results.
It releases views/unmaps only after a successful stream drain. A failed drain
raises and retains ownership; callers may retry shutdown. Successful shutdown
is idempotent. Calls are serialized by the host worker; this is not a
thread-safe public API.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

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


@dataclass
class _TransferJob:
    started_at: float
    size: int
    plan: list
    # Keep even temporary slice views alive through failed/partial enqueues.
    spans: list
    event: object | None = None


class AscendCopyWorker(OffloadingWorker):
    def __init__(self, kv_caches, block_size_factor, num_cpu_blocks, mmap_region):
        self.region = mmap_region
        self.factor = block_size_factor
        self.refs = kv_caches.group_data_refs
        self.device_tensors = []
        self.cpu_tensors = []
        self._init_transfer_state()
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
        self._stream = torch.npu.Stream(device=self.device)

    def _init_transfer_state(self):
        self.completed: list[TransferResult] = []
        # IDs stay reserved until get_finished consumes the result, not wait.
        self.pending: set[int] = set()
        self._inflight: dict[int, _TransferJob] = {}
        self._failure: str | None = None
        self._closing = False
        self._closed = False
        self._stream_drained = True

    def _check_open(self):
        if self._closing or self._closed:
            raise RuntimeError("Ascend transfer worker is closing or closed")
        if self._failure is not None:
            raise RuntimeError(f"Ascend transfer worker is fatal: {self._failure}")

    def _drain(self):
        if not self._stream_drained:
            # An event is insufficient after copy/record/context-exit failure.
            self._stream.synchronize()
            self._stream_drained = True

    def _fail(self, operation, exc):
        if self._failure is None:
            # Do not retain exception tracebacks (and their temporary views).
            self._failure = f"{operation}: {exc}"
        try:
            self._drain()
        except Exception:
            logger.exception("tiering stream drain failed; retaining ownership")

    # -- async submission -------------------------------------------------

    def _submit(self, job_id, gpu_spec, cpu_spec, store) -> bool:
        self._check_open()
        if job_id in self.pending:
            raise ValueError("transfer job ID is already pending or unconsumed")
        # Pure metadata validation happens before reserving the ID or enqueueing.
        plan = copy_plan(gpu_spec, cpu_spec, self.refs, self.factor)
        spans = self._spans(plan)
        entry = _TransferJob(
            time.monotonic(), sum(device.numel() for device, _ in spans), plan, spans
        )
        self._inflight[job_id] = entry
        self.pending.add(job_id)
        self._stream_drained = False
        try:
            # Construct and retain the event before the first possible enqueue.
            entry.event = torch.npu.Event()
            if store:
                # Order live KV reads after the runner's current compute stream.
                self._stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self._stream):
                for device, cpu in spans:
                    destination, source = (cpu, device) if store else (device, cpu)
                    destination.copy_(source, non_blocking=True)
                entry.event.record(self._stream)
        except BaseException as exc:
            self._fail(f"submitting job {job_id}", exc)
            raise RuntimeError(
                f"Ascend transfer job {job_id} submission failed"
            ) from exc
        return True

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, dst_spec, src_spec, False)

    # -- completion -------------------------------------------------------

    def get_finished(self):
        self._check_open()
        for job_id, entry in list(self._inflight.items()):
            try:
                done = entry.event.query()
            except BaseException as exc:
                self._fail(f"querying job {job_id}", exc)
                raise RuntimeError(
                    f"Ascend transfer job {job_id} query failed"
                ) from exc
            if done:
                self._finish(job_id, entry)
        completed, self.completed = self.completed, []
        self.pending.difference_update(result.job_id for result in completed)
        return completed

    def wait(self, job_ids):
        self._check_open()
        for job_id in list(job_ids):
            entry = self._inflight.get(job_id)
            if entry is None:
                # Already fenced, consumed, or not owned by this worker.
                continue
            try:
                entry.event.synchronize()
            except BaseException as exc:
                self._fail(f"waiting for job {job_id}", exc)
                raise RuntimeError(f"Ascend transfer job {job_id} wait failed") from exc
            self._finish(job_id, entry)

    def _finish(self, job_id, entry):
        # Called only after event completion proves these exact spans are idle.
        if os.environ.get("HUST_TIERING_DIAGNOSTICS") == "1":
            try:
                self._diagnose(job_id, entry.plan)
            except Exception as exc:
                logger.warning(
                    "tiering diagnostics failed for job %s: %s",
                    job_id,
                    exc,
                    exc_info=True,
                )
        result = TransferResult(
            job_id=job_id,
            success=True,
            transfer_size=entry.size,
            transfer_time=time.monotonic() - entry.started_at,
        )
        self.completed.append(result)
        del self._inflight[job_id]

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

    def _release_views(self):
        self.device_tensors.clear()
        self.cpu_tensors.clear()

    def shutdown(self):
        if self._closed:
            return
        self._closing = True
        try:
            self._drain()
        except BaseException as exc:
            if self._failure is None:
                self._failure = f"shutdown drain: {exc}"
            # Do not clear ANY views/jobs or unmap on an unproven drain.
            raise RuntimeError(
                "Ascend stream drain failed; ownership retained"
            ) from exc
        self._inflight.clear()
        self._release_views()
        self.completed.clear()
        self.pending.clear()
        # If cleanup itself raises, admission remains closed and shutdown may
        # retry it. No further DMA can be submitted after the successful drain.
        self.region.cleanup()
        self._closed = True


class AscendLayoutWorker(AscendCopyWorker):
    def __init__(self, layout, block_size_factor, num_cpu_blocks, mmap_region):
        self.region = mmap_region
        self.factor = block_size_factor
        self.refs = layout.group_data_refs
        # Snapshot containers without copying tensor storage or mutating the
        # runner's layout during shutdown.
        self.layers = tuple(
            (allocation, tuple(rows)) for allocation, rows in layout.layers
        )
        self.pages = tuple(layout.page_sizes)
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
        self._init_transfer_state()
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

    def _release_views(self):
        self.layers = ()
        super()._release_views()
