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
import threading
from collections import deque
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

from vllm_hust_kv_tiering.base import TierDeleteResult
from vllm_hust_kv_tiering.fs.segment_store import SegmentStore

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

_FS_METRIC_PREFIX = "vllm:kv_tiering_fs_"
_FS_COUNTERS = {
    "load_bytes": "Filesystem bytes read: all segment I/O and successful legacy blocks.",
    "store_bytes": "Bytes actually written to segment files; legacy writes are not observable.",
    "legacy_store_blocks": "Successful legacy store callbacks, including already-present blocks (not physical writes).",
    "load_hits": "Blocks successfully read from filesystem storage.",
    "lookup_hits": "Positive segment-index observations, including internal queries.",
    "lookup_misses": "Negative segment-index observations, including internal queries.",
    "load_jobs": "Filesystem load jobs submitted, including admission rejections.",
    "store_jobs": "Filesystem store jobs submitted, including admission rejections.",
    "load_errors": "Filesystem load job failures observed on completion.",
    "store_errors": "Filesystem store job failures observed on completion.",
    "capacity_rejections": "Segment allocation attempts rejected by the byte quota.",
    "checksum_errors": "Segment blocks rejected by CRC32 integrity checks.",
    "queue_rejections": "Jobs rejected without enqueue because the pending-job limit was reached.",
    "gc_bytes": "Physical segment-file bytes removed by garbage collection.",
    "gc_segments": "Dead segment reservations reclaimed by garbage collection.",
}


def _positive_integer(name: str, value: Any, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        suffix = f" in [1, {maximum}]" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return value


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
            if not missing_indices or self._tier.integrity == "crc32":
                # An integrity-protected namespace must not advertise unchecked
                # legacy block files as verified segment cache hits.
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
        max_storage_bytes: int | None = None,
        durability: str = "none",
        max_pending_jobs: int = 64,
        integrity: str | None = None,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Read-priority I/O threads, default 2; integer 1..64.
            n_write_threads: Write-priority I/O threads, default 2; integer 1..64.
            storage_layout: ``segment`` for indexed append-only objects or
                ``file`` for one file per KV block.
            segment_size_bytes: Maximum segment object size.
            segment_io_batch_blocks: Blocks per batched segment I/O task. Zero
                selects about 1 MiB per task.
            segment_shards: Parallel append streams. Zero selects up to 16,
                based on the total number of I/O threads.
            max_storage_bytes: Optional positive segment-reservation quota.
                Includes failed/in-flight writes, not SQLite/WAL or legacy files.
                Capacity exhaustion fails the async job without deleting live
                data. Partially live segments are not compacted. None is unbounded.
                Quota/durability are persisted; mismatched reopen values fail.
            durability: Segment-only 'none' (default, disposable cache, no
                stable-media guarantee) or 'fsync' (flush files and directory
                before SQLite FULL publication). File layout supports only none.
            max_pending_jobs: Positive integer, default 64. Bounds admitted jobs
                queued, running or completed but not yet polled. Saturation
                enqueues no I/O and reports a failed JobResult on the next poll;
                the host then releases its source/destination reservation.
            integrity: None selects 'crc32' for segment layout and 'none' for
                legacy file layout. Explicit 'none' disables verification;
                'crc32' checks each block before a successful load completion
                and never falls back to unverified legacy files. Persisted and
                immutable; existing unchecksummed data requires a fresh namespace.
                CRC32 detects accidental corruption, not malicious modification;
                checksum overhead has not been measured on target hardware.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        try:
            root_dir = os.fspath(root_dir)
        except TypeError as error:
            raise ValueError("root_dir must be a nonempty absolute path") from error
        if not isinstance(root_dir, str) or not root_dir or not os.path.isabs(root_dir):
            raise ValueError("root_dir must be a nonempty absolute path")

        # Large inference hosts can expose hundreds of logical CPUs. Storage
        # must not silently create a correspondingly large competing pool.
        n_read_threads = _positive_integer(
            "n_read_threads", 2 if n_read_threads is None else n_read_threads, 64
        )
        n_write_threads = _positive_integer(
            "n_write_threads", 2 if n_write_threads is None else n_write_threads, 64
        )
        self.max_pending_jobs = _positive_integer("max_pending_jobs", max_pending_jobs)

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]
        if storage_layout not in {"file", "segment"}:
            raise ValueError("storage_layout must be 'file' or 'segment'")
        for name, value in (
            ("segment_io_batch_blocks", segment_io_batch_blocks),
            ("segment_shards", segment_shards),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(durability, str) or durability not in {"none", "fsync"}:
            raise ValueError("durability must be 'none' or 'fsync'")
        if storage_layout == "file" and (
            max_storage_bytes is not None or durability != "none"
        ):
            raise ValueError(
                "quota and fsync durability require storage_layout='segment'"
            )
        if integrity is None:
            integrity = "crc32" if storage_layout == "segment" else "none"
        if not isinstance(integrity, str) or integrity not in {"none", "crc32"}:
            raise ValueError("integrity must be None, 'none' or 'crc32'")
        if storage_layout == "file" and integrity != "none":
            raise ValueError("crc32 integrity requires storage_layout='segment'")
        self.integrity = integrity
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
        ascend_profile = getattr(offloading_spec, "extra_config", {}).get(
            "ascend_copy_backend"
        ) in {"torch_async", "torch_sync"}
        geometry = (
            {"gpu_blocks_per_file": offloading_spec.block_size_factor}
            if ascend_profile
            else {"blocks_per_file": offloading_spec.blocks_per_chunk}
        )
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            # vLLM-HUST v1 names the normalized chunk geometry
            # ``blocks_per_chunk``.  It is the old block_size_factor value.
            **geometry,
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
                    f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
                    if ascend_profile
                    else self.file_mapper.get_rank_path(),
                    "segments",
                ),
                block_size=self._block_size,
                segment_size_bytes=segment_size_bytes,
                segment_shards=self.segment_shards,
                max_storage_bytes=max_storage_bytes,
                durability=durability,
                integrity=self.integrity,
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
        # Scheduler-owned only: workers must not mutate host async lookup state.
        self._active_requests: set[str] = set()
        self._failed_load_keys: dict[str, set[OffloadKey]] = {}
        self._pending_loads: dict[int, tuple[str, list[OffloadKey]]] = {}
        self._pending_stores: dict[int, list[OffloadKey]] = {}
        self._admitted_jobs: set[int] = set()
        self._rejected_jobs: deque[tuple[int, bool]] = deque()
        self._job_counters = dict.fromkeys(
            (
                "load_jobs",
                "store_jobs",
                "load_errors",
                "store_errors",
                "queue_rejections",
            ),
            0,
        )
        self._legacy_counter_lock = threading.Lock()
        self._legacy_counters = dict.fromkeys(
            ("load_bytes", "load_hits", "legacy_store_blocks"), 0
        )
        self._metrics_lock = threading.Lock()
        self._metric_snapshot: dict[str, int] = {}
        self._counter_baseline: dict[str, int] = {}

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._active_requests.add(req_context.req_id)
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        self._active_requests.add(req_context.req_id)
        # The host caches positive lookup results for a request. A file can
        # disappear (or become unreadable) between lookup and promotion. Treat
        # a failed promotion as a request-local miss rather than retry forever.
        if key in self._failed_load_keys.get(req_context.req_id, ()):
            return LookupResult.MISS
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        self._check_job_id(job_metadata.job_id)
        keys = list(job_metadata.keys)
        self._pending_stores[job_metadata.job_id] = keys
        self._job_counters["store_jobs"] += 1
        if not self._admit_job(job_metadata.job_id):
            return
        if self.segment_store is not None:
            # The host pins primary slots until the resulting JobResult. Keep
            # allocation, quota/GC, I/O and index publication in the worker, so
            # SQLite/flock contention never blocks the scheduler. A successful
            # task result is emitted only after complete_store() has committed.
            task = functools.partial(
                self.segment_store.store_many,
                keys,
                self._primary_kv_view,
                [int(block_id) for block_id in job_metadata.block_ids],
                self.segment_io_batch_blocks,
            )
            self._pool.enqueue_store(job_metadata.job_id, 1, [task])
            return
        task = functools.partial(
            self._file_job,
            keys,
            [int(block_id) for block_id in job_metadata.block_ids],
            True,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        self._check_job_id(job_metadata.job_id)
        keys = list(job_metadata.keys)
        req_id = job_metadata.req_context.req_id
        self._active_requests.add(req_id)
        self._pending_loads[job_metadata.job_id] = (req_id, keys)
        self._job_counters["load_jobs"] += 1
        if not self._admit_job(job_metadata.job_id):
            return
        if self.segment_store is not None:
            task = functools.partial(
                self._load_segment_job,
                keys,
                [int(block_id) for block_id in job_metadata.block_ids],
            )
            self._pool.enqueue_load(job_metadata.job_id, 1, [task])
            return
        task = functools.partial(
            self._file_job,
            keys,
            [int(block_id) for block_id in job_metadata.block_ids],
            False,
        )
        self._pool.enqueue_load(job_metadata.job_id, 1, [task])

    def _check_job_id(self, job_id: int) -> None:
        if job_id in self._pending_loads or job_id in self._pending_stores:
            raise ValueError(f"duplicate filesystem job_id {job_id}")

    def _admit_job(self, job_id: int) -> bool:
        # Admission and completion polling are scheduler-thread owned. Rejecting
        # never waits on a worker, touches SQLite or mutates the async lookup
        # cache. Rejected results still reach the host, releasing its pins.
        if len(self._admitted_jobs) >= self.max_pending_jobs:
            self._job_counters["queue_rejections"] += 1
            self._rejected_jobs.append((job_id, False))
            return False
        self._admitted_jobs.add(job_id)
        return True

    def _file_job(
        self, keys: list[OffloadKey], block_ids: list[int], store: bool
    ) -> None:
        if len(keys) != len(block_ids):
            raise ValueError("keys and block_ids must have the same length")
        for key, block_id in zip(keys, block_ids):
            transfer = store_block if store else load_block
            transfer(
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                block_id * self._block_size,
                self._block_size,
            )
            # Legacy store_block silently skips existing files, so successful
            # callbacks must NOT be claimed as physical writes. Failed/partial
            # legacy I/O is also unavailable through the host helper's ABI.
            with self._legacy_counter_lock:
                if store:
                    self._legacy_counters["legacy_store_blocks"] += 1
                else:
                    self._legacy_counters["load_bytes"] += self._block_size
                    self._legacy_counters["load_hits"] += 1

    def _load_segment_job(self, keys: list[OffloadKey], block_ids: list[int]) -> None:
        """Resolve metadata off-scheduler; retain compatibility with legacy files."""
        assert self.segment_store is not None
        if self.integrity == "crc32":
            self.segment_store.load_many(
                keys,
                self._primary_kv_view,
                block_ids,
                io_batch_blocks=self.segment_io_batch_blocks,
            )
            return
        locations = self.segment_store.get_locations(keys)
        segment_keys = []
        segment_ids = []
        for key, block_id, location in zip(keys, block_ids, locations):
            if location is None:
                self._file_job([key], [block_id], False)
            else:
                segment_keys.append(key)
                segment_ids.append(block_id)
        if segment_keys:
            self.segment_store.load_many(
                segment_keys,
                self._primary_kv_view,
                segment_ids,
                io_batch_blocks=self.segment_io_batch_blocks,
            )

    def _storage_counter_snapshot(self) -> dict[str, int]:
        counters = dict.fromkeys(_FS_COUNTERS, 0)
        if self.segment_store is not None:
            counters.update(self.segment_store.get_storage_counters())
        with self._legacy_counter_lock:
            for name, value in self._legacy_counters.items():
                counters[name] += value
        counters.update(self._job_counters)
        return counters

    def get_storage_counters(self, reset: bool = False) -> dict[str, int]:
        """Cheap process-local FS observations, independent of Prometheus drains.

        Segment bytes include duplicate/partial/failed-job I/O. Legacy loads
        expose successful full-block bytes only; legacy store callbacks are
        counted separately, never claimed as physical writes. Job errors are counted when
        observed on the scheduler. reset starts a new snapshot interval without
        resetting the underlying cumulative counters used by get_stats().
        No SQLite/filesystem lock is acquired here.
        """
        with self._metrics_lock:
            current = self._storage_counter_snapshot()
            result = {
                name: value - self._counter_baseline.get(name, 0)
                for name, value in current.items()
            }
            if reset:
                self._counter_baseline = current
            return result

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        metrics: dict[str, OffloadingMetricMetadata] = {
            _FS_METRIC_PREFIX + name: OffloadingCounterMetadata(
                documentation=description,
                labelnames=("tier",),
            )
            for name, description in _FS_COUNTERS.items()
        }
        metrics[_FS_METRIC_PREFIX + "pending_jobs"] = OffloadingGaugeMetadata(
            documentation="Admitted filesystem jobs awaiting completion polling.",
            labelnames=("tier",),
        )
        return metrics

    @override
    def get_stats(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        stats = OffloadingConnectorStats()
        with self._metrics_lock:
            current = self._storage_counter_snapshot()
            for name, value in current.items():
                previous = self._metric_snapshot.get(name, 0)
                # A caller resetting the underlying leaf counters directly is
                # tolerated, though manager snapshots never reset them.
                delta = value - previous if value >= previous else value
                # Publish zero on first observation as well: a lazily missing
                # label otherwise loses the first measured restore delta.
                stats.increase_counter(
                    _FS_METRIC_PREFIX + name, delta, (self.tier_type,)
                )
            self._metric_snapshot = current
            stats.set_gauge(
                _FS_METRIC_PREFIX + "pending_jobs",
                len(self._admitted_jobs),
                (self.tier_type,),
            )
        return stats

    def collect_garbage(self) -> int:
        """Admin/worker-only reclamation; returns released segment quota bytes.

        Not a scheduler hook. No partially live compaction; returns zero if any
        sharing process currently holds a read/store activity lease.
        """
        return (
            self.segment_store.collect_garbage()
            if self.segment_store is not None
            else 0
        )

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = list(self._pool.get_finished())
        results.extend(self._rejected_jobs)
        self._rejected_jobs.clear()
        for job_id, success in results:
            self._admitted_jobs.discard(job_id)
            load = self._pending_loads.pop(job_id, None)
            if load is not None and not success:
                self._job_counters["load_errors"] += 1
                req_id, keys = load
                self._lookup_manager.cleanup(req_id)
                if req_id in self._active_requests:
                    self._failed_load_keys.setdefault(req_id, set()).update(keys)
            stored_keys = self._pending_stores.pop(job_id, None)
            if stored_keys is not None and not success:
                self._job_counters["store_errors"] += 1
            if stored_keys is not None and success:
                for failures in self._failed_load_keys.values():
                    failures.difference_update(stored_keys)
            yield JobResult(job_id=job_id, success=success)

    @override
    def has_pending_work(self) -> bool:
        return bool(self._pending_loads or self._pending_stores)

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._active_requests.discard(req_context.req_id)
        self._failed_load_keys.pop(req_context.req_id, None)
        self._lookup_manager.cleanup(req_context.req_id)

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
