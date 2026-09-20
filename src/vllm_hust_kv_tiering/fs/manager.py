# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    The default segment layout appends a job's blocks to large segment files
    and commits their locations to a persistent SQLite index. The legacy file
    layout writes one atomic file per block.

Load path:
    Segment blocks with adjacent offsets are loaded with batched preadv calls.
    The legacy layout reads one block file per task.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
from vllm_hust_kv_tiering.async_lookup import AsyncLookupManager
from vllm_hust_kv_tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierDeleteResult,
)
from vllm_hust_kv_tiering.fs.io import load_block, store_block
from vllm_hust_kv_tiering.fs.segment_store import (
    SegmentLocation,
    SegmentStore,
)
from vllm_hust_kv_tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        if self._tier.segment_store is not None:
            segment_hits = self._tier.segment_store.contains_many(keys)
            missing_indices = [
                index for index, is_hit in enumerate(segment_hits) if not is_hit
            ]
            if not missing_indices:
                return segment_hits
            missing_paths = [
                self._tier.file_mapper.get_file_name(keys[index])
                for index in missing_indices
            ]
            legacy_hits = _batch_file_lookup(missing_paths)
            for index, is_hit in zip(missing_indices, legacy_hits):
                segment_hits[index] = is_hit
            return segment_hits
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        return _batch_file_lookup(paths)


def _batch_file_lookup(paths: list[str]) -> list[bool]:
    if _HAS_BATCH_LOOKUP_C:
        # C extension: GIL released for the entire faccessat() batch.
        return batch_lookup_C(paths)
    return [os.path.exists(path) for path in paths]


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int | None = None,
        n_write_threads: int | None = None,
        storage_layout: str = "segment",
        segment_size_bytes: int = 1024**3,
        segment_io_batch_blocks: int = 0,
        segment_shards: int = 0,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads. If omitted,
                the available CPU count is split between reads and writes.
            n_write_threads: Number of write-priority I/O threads. If omitted,
                the available CPU count is split between reads and writes.
            storage_layout: ``segment`` for indexed append-only objects or
                ``file`` for one file per KV block.
            segment_size_bytes: Maximum segment object size.
            segment_io_batch_blocks: Blocks per batched segment I/O task. Zero
                selects about 1 MiB per task.
            segment_shards: Parallel append streams. Zero selects up to 16,
                based on the total number of I/O threads.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # The FS tier is CPU-bound (preadv, memcpy and index bookkeeping).  Do
        # not silently fall back to the historical 16+16 thread setting when
        # the caller does not specify a pool size: use all online CPUs, split
        # between read- and write-priority workers.  Explicit values remain
        # supported for controlled experiments or machines with constrained
        # I/O bandwidth.
        cpu_threads = max(2, os.cpu_count() or 2)
        if n_read_threads is None and n_write_threads is None:
            n_read_threads = (cpu_threads + 1) // 2
            n_write_threads = cpu_threads // 2
        elif n_read_threads is None:
            n_read_threads = max(1, cpu_threads - int(n_write_threads))
        elif n_write_threads is None:
            n_write_threads = max(1, cpu_threads - int(n_read_threads))
        n_read_threads = max(1, int(n_read_threads))
        n_write_threads = max(1, int(n_write_threads))

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]
        if storage_layout not in {"file", "segment"}:
            raise ValueError("storage_layout must be 'file' or 'segment'")
        if segment_io_batch_blocks < 0:
            raise ValueError("segment_io_batch_blocks must be non-negative")
        if segment_shards < 0:
            raise ValueError("segment_shards must be non-negative")
        self.storage_layout = storage_layout
        self.segment_io_batch_blocks = segment_io_batch_blocks or max(
            1,
            1024**2 // self._block_size,
        )
        self.segment_shards = segment_shards or min(
            16,
            max(1, n_read_threads + n_write_threads),
        )

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            # vLLM-HUST v1 names the normalized chunk geometry
            # ``blocks_per_chunk``.  It is the old block_size_factor value.
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self.segment_store = (
            SegmentStore(
                root_dir=os.path.join(
                    self.file_mapper.get_rank_path(),
                    "segments",
                ),
                block_size=self._block_size,
                segment_size_bytes=segment_size_bytes,
                segment_shards=self.segment_shards,
            )
            if storage_layout == "segment"
            else None
        )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.segment_store is not None:
            plan = self.segment_store.prepare_store(
                list(job_metadata.keys),
                [int(block_id) for block_id in job_metadata.block_ids],
            )
            tasks = [
                functools.partial(
                    self.segment_store.write_entries,
                    entries,
                    self._primary_kv_view,
                    self.segment_io_batch_blocks,
                )
                for entries in self.segment_store.group_writes(plan.entries)
            ]
            if not tasks:
                tasks = [lambda: None]
            self._pool.enqueue_store(
                job_metadata.job_id,
                len(tasks),
                tasks,
                on_complete=functools.partial(
                    self.segment_store.complete_store,
                    plan,
                ),
            )
            return
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        if self.segment_store is not None:
            keys = list(job_metadata.keys)
            block_ids = [int(block_id) for block_id in job_metadata.block_ids]
            locations = self.segment_store.get_locations(keys)
            segment_entries: dict[
                int,
                list[tuple[OffloadKey, int, SegmentLocation]],
            ] = {}
            tasks = []
            for key, block_id, location in zip(keys, block_ids, locations):
                if location is not None:
                    segment_entries.setdefault(location.segment_id, []).append(
                        (key, block_id, location)
                    )
                else:
                    tasks.append(
                        functools.partial(
                            load_block,
                            self.file_mapper.get_file_name(key),
                            self._primary_kv_view,
                            block_id * self._block_size,
                            self._block_size,
                        )
                    )
            for chunk in segment_entries.values():
                tasks.append(
                    functools.partial(
                        self.segment_store.load_many,
                        [entry[0] for entry in chunk],
                        self._primary_kv_view,
                        [entry[1] for entry in chunk],
                        [entry[2] for entry in chunk],
                        self.segment_io_batch_blocks,
                    )
                )
            self._pool.enqueue_load(job_metadata.job_id, len(tasks), tasks)
            return
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        return (
            JobResult(job_id=job_id, success=success)
            for job_id, success in self._pool.get_finished()
        )

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def delete(self, keys: Collection[OffloadKey]) -> TierDeleteResult:
        key_list = list(keys)
        deleted_keys = (
            self.segment_store.delete(key_list)
            if self.segment_store is not None
            else set()
        )
        removed_keys: set[OffloadKey] = set()
        for key in key_list:
            try:
                os.unlink(self.file_mapper.get_file_name(key))
            except FileNotFoundError:
                removed_keys.add(key)
            except OSError:
                logger.warning(
                    "Failed to delete KV block from filesystem tier",
                    exc_info=True,
                )
            else:
                deleted_keys.add(key)
                removed_keys.add(key)
        return TierDeleteResult(
            removed_keys=removed_keys,
            deleted_count=len(deleted_keys),
        )

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
        if self.segment_store is not None:
            self.segment_store.close()


# A descriptive public name for configuration and documentation.  Keep the
# historical class name as an alias for scripts developed before pluginization.
SegmentFileSystemTier = FileSystemTierManager
