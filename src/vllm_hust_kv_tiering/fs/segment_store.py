# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
import uuid
import zlib
from collections import defaultdict
from collections.abc import Collection, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from vllm.v1.kv_offload.base import OffloadKey

O_DIRECT = getattr(os, "O_DIRECT", 0)
_DEFAULT_SEGMENT_SIZE = 1024**3
_INDEX_BATCH_SIZE = 500


class StorageCapacityError(OSError):
    """The reserved segment-byte budget cannot accommodate a whole store job."""


@dataclass(frozen=True, slots=True)
class SegmentLocation:
    segment_id: int
    offset: int
    length: int
    checksum: int | None = None


@dataclass(frozen=True, slots=True)
class SegmentWrite:
    key: bytes
    segment_id: int
    offset: int
    block_id: int


@dataclass(frozen=True, slots=True)
class SegmentStorePlan:
    entries: tuple[SegmentWrite, ...]
    lease_id: str | None = None
    delete_generation: int = 0


class SegmentStore:
    """Append-only indexed cache, with optional bounded segment reservations.

    ``max_storage_bytes`` bounds reserved segment extents, including failed and
    in-flight writes, NOT SQLite/WAL/lock-file overhead or legacy block files.
    None preserves the historical unbounded configuration. Exhaustion rejects
    the whole job without evicting live data. Fully dead segments can be removed
    by collect_garbage(); partially live segments are never compacted or reused.
    Quota, durability and integrity are immutable shared-store settings: reopening
    the same root with different values raises ValueError (including None vs finite).
    integrity='crc32' detects accidental block corruption using stdlib CRC32;
    'none' preserves the unverified low-level default. CRC32 is not cryptographic
    authentication. A CRC namespace cannot trust old rows without checksums:
    use a fresh namespace, not an in-place upgrade of unverified cached data.

    ``durability='none'`` (default) is a disposable cache: process completion is
    not a stable-media guarantee. ``'fsync'`` flushes data and directory entries
    before publishing the index, with SQLite synchronous=FULL. This relies on
    the filesystem/device honoring fsync; it is not a transactional checkpoint
    for any other tier. All sharing processes must use this locking protocol.

    A prepared plan holds a shared activity lease until complete_store(), even
    on failure. GC requires an exclusive lease and returns immediately if busy.
    Crashes release leases; abandoned reservations remain charged until GC.
    Callers using the low-level plan API MUST complete every plan exactly once.
    """

    def __init__(
        self,
        root_dir: str,
        block_size: int,
        segment_size_bytes: int = _DEFAULT_SEGMENT_SIZE,
        segment_shards: int = 1,
        max_storage_bytes: int | None = None,
        durability: str = "none",
        integrity: str = "none",
    ) -> None:
        for name, value in (
            ("block_size", block_size),
            ("segment_size_bytes", segment_size_bytes),
            ("segment_shards", segment_shards),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if segment_size_bytes < block_size:
            raise ValueError("segment_size_bytes must be at least one block")
        if max_storage_bytes is not None and (
            isinstance(max_storage_bytes, bool)
            or not isinstance(max_storage_bytes, int)
            or max_storage_bytes < block_size
        ):
            raise ValueError("max_storage_bytes must be None or at least one block")
        if not isinstance(durability, str) or durability not in {"none", "fsync"}:
            raise ValueError("durability must be 'none' or 'fsync'")
        if not isinstance(integrity, str) or integrity not in {"none", "crc32"}:
            raise ValueError("integrity must be 'none' or 'crc32'")

        self.block_size = block_size
        self.segment_size_bytes = (segment_size_bytes // block_size) * block_size
        self.segment_shards = segment_shards
        self.max_storage_bytes = max_storage_bytes
        self.durability = durability
        self.integrity = integrity
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root_dir / "segments.lock"
        self._activity_path = self.root_dir / "segments.activity.lock"
        self._lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        self._thread_lock = threading.RLock()
        self._segment_locks: dict[int, threading.Lock] = {}
        self._leases: dict[str, int] = {}
        # Only active CRC plans own entries here; never retain source views.
        self._plan_checksums: dict[str, dict[tuple[int, int], int]] = {}
        self._extent_plans: dict[tuple[int, int], str] = {}
        self._closed = False
        self._counter_lock = threading.Lock()
        self._counters = dict.fromkeys(
            (
                "lookup_hits",
                "lookup_misses",
                "load_bytes",
                "store_bytes",
                "load_hits",
                "load_jobs",
                "store_jobs",
                "load_errors",
                "store_errors",
                "capacity_rejections",
                "checksum_errors",
                "gc_bytes",
                "gc_segments",
            ),
            0,
        )
        self._db = sqlite3.connect(
            self.root_dir / "index.sqlite3",
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "PRAGMA synchronous=" + ("FULL" if durability == "fsync" else "NORMAL")
            )
            self._db.execute("PRAGMA temp_store=MEMORY")
            with self._write_lock():
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS blocks ("
                    "key BLOB PRIMARY KEY, segment_id INTEGER NOT NULL, "
                    "offset INTEGER NOT NULL, length INTEGER NOT NULL, checksum INTEGER"
                    ") WITHOUT ROWID"
                )
                columns = {
                    row[1] for row in self._db.execute("PRAGMA table_info(blocks)")
                }
                if "checksum" not in columns:
                    self._db.execute("ALTER TABLE blocks ADD COLUMN checksum INTEGER")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS metadata ("
                    "name TEXT PRIMARY KEY, value INTEGER NOT NULL"
                    ") WITHOUT ROWID"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS segment_extents ("
                    "segment_id INTEGER PRIMARY KEY, length INTEGER NOT NULL)"
                )
                self._initialize_extents()
        except BaseException:
            self._db.close()
            os.close(self._lock_fd)
            raise

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._counter_lock:
            self._counters[name] += amount

    def get_storage_counters(self, reset: bool = False) -> dict[str, int]:
        """Cheap process-local observations; byte counters measure actual I/O.

        Bytes include duplicate/failed-job I/O, not logical GPU transfer sizes.
        Counters do not include the manager's legacy-file fallback path.
        """
        with self._counter_lock:
            result = dict(self._counters)
            if reset:
                self._counters = dict.fromkeys(self._counters, 0)
            return result

    @contextmanager
    def _write_lock(self):
        with self._thread_lock:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _activity_fd(self, exclusive: bool = False) -> int | None:
        fd = os.open(self._activity_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            flags = fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _segment_path(self, segment_id: int) -> Path:
        return self.root_dir / f"segment-{segment_id:08d}.bin"

    def _segment_lock(self, segment_id: int) -> threading.Lock:
        with self._thread_lock:
            return self._segment_locks.setdefault(segment_id, threading.Lock())

    def _initialize_extents(self) -> None:
        metadata = dict(self._db.execute("SELECT name, value FROM metadata"))
        if int(metadata.get("block_size", self.block_size)) != self.block_size:
            raise ValueError("block_size does not match the existing segment store")
        expected_integrity = int(self.integrity == "crc32")
        if (
            "integrity_crc32" in metadata
            and int(metadata["integrity_crc32"]) != expected_integrity
        ):
            raise ValueError(
                "integrity does not match the existing segment store; "
                "use a fresh namespace to change integrity mode"
            )
        if (
            self.integrity == "crc32"
            and self._db.execute(
                "SELECT 1 FROM blocks WHERE checksum IS NULL LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise ValueError(
                "existing index contains blocks without checksums; "
                "use a fresh namespace for integrity='crc32'"
            )
        shared_config = [
            ("integrity_crc32", expected_integrity),
            (
                "max_storage_bytes",
                -1 if self.max_storage_bytes is None else self.max_storage_bytes,
            ),
            ("durability_fsync", int(self.durability == "fsync")),
        ]
        for name, value in shared_config:
            if name in metadata and int(metadata[name]) != value:
                raise ValueError(f"{name} does not match the existing segment store")
        extents = dict(
            self._db.execute("SELECT segment_id, length FROM segment_extents")
        )
        for segment_id, length in self._db.execute(
            "SELECT segment_id, MAX(offset + length) FROM blocks GROUP BY segment_id"
        ):
            extents[segment_id] = max(extents.get(segment_id, 0), length)
        # Account for orphaned/failed writes as well as indexed bytes. Never
        # infer a reusable offset from the live index after keys were deleted.
        for path in self.root_dir.glob("segment-*.bin"):
            try:
                segment_id = int(path.stem.removeprefix("segment-"))
            except ValueError:
                continue
            extents[segment_id] = max(extents.get(segment_id, 0), path.stat().st_size)
        for name, segment_id in metadata.items():
            if name.endswith("segment_id") and name != "next_segment_id":
                offset_name = name.removesuffix("segment_id") + "segment_offset"
                extents[segment_id] = max(
                    extents.get(segment_id, 0), metadata.get(offset_name, 0)
                )
        next_id = max(
            int(metadata.get("next_segment_id", 0)), max(extents, default=-1) + 1
        )
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.executemany(
                "INSERT INTO segment_extents VALUES (?, ?) "
                "ON CONFLICT(segment_id) DO UPDATE SET length=excluded.length",
                list(extents.items()),
            )
            self._set_metadata(
                [("block_size", self.block_size), ("next_segment_id", next_id)]
                + shared_config
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _set_metadata(self, items: list[tuple[str, int]]) -> None:
        self._db.executemany(
            "INSERT INTO metadata(name, value) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
            items,
        )

    def _allocation_state(self) -> tuple[list[tuple[int, int]], int, int]:
        metadata = dict(self._db.execute("SELECT name, value FROM metadata"))
        extents = dict(
            self._db.execute("SELECT segment_id, length FROM segment_extents")
        )
        next_id = max(
            int(metadata.get("next_segment_id", 0)), max(extents, default=-1) + 1
        )
        state = []
        for shard in range(self.segment_shards):
            name = f"shard_{shard}_segment_id"
            if name in metadata:
                segment_id = int(metadata[name])
                offset = max(
                    int(metadata.get(f"shard_{shard}_segment_offset", 0)),
                    extents.get(segment_id, 0),
                )
            elif shard == 0 and "segment_id" in metadata:
                segment_id = int(metadata["segment_id"])
                offset = max(
                    int(metadata.get("segment_offset", 0)), extents.get(segment_id, 0)
                )
            else:
                segment_id, offset = next_id, 0
                next_id += 1
            state.append((segment_id, offset))
        return (
            state,
            next_id,
            int(metadata.get("shard_cursor", 0)) % self.segment_shards,
        )

    @staticmethod
    def _chunks(values: Sequence[bytes]) -> list[Sequence[bytes]]:
        return [
            values[start : start + _INDEX_BATCH_SIZE]
            for start in range(0, len(values), _INDEX_BATCH_SIZE)
        ]

    def get_locations(self, keys: Sequence[OffloadKey]) -> list[SegmentLocation | None]:
        if not keys:
            return []
        key_bytes = [bytes(key) for key in keys]
        locations: dict[bytes, SegmentLocation] = {}
        # No positive location cache: every lookup observes committed SQLite
        # rows, including deletes/reinserts by other connections/processes.
        with self._thread_lock:
            for chunk in self._chunks(list(dict.fromkeys(key_bytes))):
                placeholders = ",".join("?" for _ in chunk)
                for key, segment_id, offset, length, checksum in self._db.execute(
                    "SELECT key, segment_id, offset, length, checksum FROM blocks "
                    f"WHERE key IN ({placeholders})",
                    chunk,
                ):
                    locations[bytes(key)] = SegmentLocation(
                        segment_id, offset, length, checksum
                    )
        result = [locations.get(key) for key in key_bytes]
        hits = sum(location is not None for location in result)
        self._increment("lookup_hits", hits)
        self._increment("lookup_misses", len(result) - hits)
        return result

    def contains_many(self, keys: Sequence[OffloadKey]) -> list[bool]:
        return [location is not None for location in self.get_locations(keys)]

    @staticmethod
    def _iov_max() -> int:
        try:
            return max(1, int(os.sysconf("SC_IOV_MAX")))
        except (ValueError, OSError):
            return 1024

    def _vector_io(
        self,
        fd: int,
        buffers: Sequence[memoryview],
        offset: int,
        max_buffers: int | None,
        write: bool,
    ) -> None:
        if max_buffers is not None and max_buffers <= 0:
            raise ValueError("io_batch_blocks must be positive")
        iov_max = min(self._iov_max(), max_buffers or self._iov_max())
        for start in range(0, len(buffers), iov_max):
            chunk = buffers[start : start + iov_max]
            expected = sum(len(view) for view in chunk)
            if write and hasattr(os, "pwritev"):
                count = os.pwritev(fd, chunk, offset)
            elif not write and hasattr(os, "preadv"):
                count = os.preadv(fd, chunk, offset)
            else:
                # POSIX platforms such as macOS lack Python's vector syscalls.
                count = 0
                for view in chunk:
                    if write:
                        size = os.pwrite(fd, view, offset + count)
                    else:
                        data = os.pread(fd, len(view), offset + count)
                        size = len(data)
                        view[:size] = data
                    count += size
                    if size != len(view):
                        break
            self._increment("store_bytes" if write else "load_bytes", count)
            if count != expected:
                raise OSError(
                    f"Short {'write' if write else 'read'}: "
                    f"expected {expected}, got {count}"
                )
            offset += count

    def _pwritev_all(
        self,
        fd: int,
        buffers: Sequence[memoryview],
        offset: int,
        max_buffers: int | None = None,
    ) -> None:
        self._vector_io(fd, buffers, offset, max_buffers, True)

    def _preadv_all(
        self,
        fd: int,
        buffers: Sequence[memoryview],
        offset: int,
        max_buffers: int | None = None,
    ) -> None:
        self._vector_io(fd, buffers, offset, max_buffers, False)

    def _validate_buffer(
        self, buffer: memoryview, block_ids: Sequence[int], writable: bool = False
    ) -> memoryview:
        view = buffer.cast("B")
        if writable and view.readonly:
            raise ValueError("load destination must be writable")
        for block_id in block_ids:
            if int(block_id) < 0 or (int(block_id) + 1) * self.block_size > len(view):
                raise ValueError("block_id is outside the supplied buffer")
        return view

    def store_many(
        self,
        keys: Sequence[OffloadKey],
        buffer: memoryview,
        block_ids: Sequence[int],
        io_batch_blocks: int | None = None,
    ) -> None:
        self._increment("store_jobs")
        plan = None
        try:
            self._validate_buffer(buffer, block_ids)
            try:
                plan = self.prepare_store(keys, block_ids)
            except StorageCapacityError:
                self.collect_garbage()
                plan = self.prepare_store(keys, block_ids)
            self.write_entries(plan.entries, buffer, io_batch_blocks)
            self.complete_store(plan, True)
            plan = None
        except BaseException:
            self._increment("store_errors")
            if plan is not None:
                self.complete_store(plan, False)
            raise

    def prepare_store(
        self, keys: Sequence[OffloadKey], block_ids: Sequence[int]
    ) -> SegmentStorePlan:
        if len(keys) != len(block_ids):
            raise ValueError("keys and block_ids must have the same length")
        if any(int(block_id) < 0 for block_id in block_ids):
            raise ValueError("block_ids must be non-negative")
        if not keys:
            return SegmentStorePlan(())
        fd = self._activity_fd()
        assert fd is not None
        try:
            with self._write_lock():
                existing = self.contains_many(keys)
                unique: dict[bytes, int] = {}
                for key, block_id, is_stored in zip(keys, block_ids, existing):
                    if not is_stored:
                        unique.setdefault(bytes(key), int(block_id))
                if not unique:
                    os.close(fd)
                    return SegmentStorePlan(())
                reserved = self._db.execute(
                    "SELECT COALESCE(SUM(length), 0) FROM segment_extents"
                ).fetchone()[0]
                if (
                    self.max_storage_bytes is not None
                    and reserved + len(unique) * self.block_size
                    > self.max_storage_bytes
                ):
                    self._increment("capacity_rejections")
                    raise StorageCapacityError(
                        "FS segment quota exhausted; "
                        "delete keys and collect fully dead segments"
                    )
                state, next_id, cursor = self._allocation_state()
                groups: list[list[SegmentWrite]] = [[] for _ in state]
                extents: dict[int, int] = {}
                for index, (key, block_id) in enumerate(unique.items()):
                    shard = (cursor + index) % self.segment_shards
                    segment_id, offset = state[shard]
                    if offset + self.block_size > self.segment_size_bytes:
                        segment_id, offset = next_id, 0
                        next_id += 1
                    groups[shard].append(
                        SegmentWrite(key, segment_id, offset, block_id)
                    )
                    state[shard] = (segment_id, offset + self.block_size)
                    extents[segment_id] = offset + self.block_size
                generation = self._delete_generation()
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    self._set_metadata(
                        [
                            ("next_segment_id", next_id),
                            (
                                "shard_cursor",
                                (cursor + len(unique)) % self.segment_shards,
                            ),
                        ]
                        + [
                            (f"shard_{shard}_segment_id", segment_id)
                            for shard, (segment_id, _) in enumerate(state)
                        ]
                        + [
                            (f"shard_{shard}_segment_offset", offset)
                            for shard, (_, offset) in enumerate(state)
                        ]
                    )
                    self._db.executemany(
                        "INSERT INTO segment_extents VALUES (?, ?) "
                        "ON CONFLICT(segment_id) DO UPDATE SET length=excluded.length",
                        list(extents.items()),
                    )
                    self._db.execute("COMMIT")
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise
                lease_id = uuid.uuid4().hex
                entries = tuple(entry for group in groups for entry in group)
                self._leases[lease_id] = fd
                if self.integrity == "crc32":
                    self._plan_checksums[lease_id] = {}
                    for entry in entries:
                        self._extent_plans[(entry.segment_id, entry.offset)] = lease_id
                return SegmentStorePlan(entries, lease_id, generation)
        except BaseException:
            os.close(fd)
            raise

    def write_entries(
        self,
        entries: Sequence[SegmentWrite],
        buffer: memoryview,
        io_batch_blocks: int | None = None,
    ) -> None:
        if not entries:
            return
        byte_view = self._validate_buffer(buffer, [entry.block_id for entry in entries])
        checksums: dict[tuple[int, int], int] = {}
        if self.integrity == "crc32":
            with self._thread_lock:
                if any(
                    (entry.segment_id, entry.offset) not in self._extent_plans
                    for entry in entries
                ):
                    raise ValueError(
                        "checksum write entries must belong to an active store plan"
                    )
                for entry in entries:
                    extent = (entry.segment_id, entry.offset)
                    self._plan_checksums[self._extent_plans[extent]].pop(extent, None)
            # zlib consumes the pinned memoryview directly: no block-sized copy.
            checksums = {
                (entry.segment_id, entry.offset): zlib.crc32(
                    byte_view[
                        entry.block_id * self.block_size : (entry.block_id + 1)
                        * self.block_size
                    ]
                )
                for entry in entries
            }
        current_segment, current_offset = entries[0].segment_id, entries[0].offset
        views: list[memoryview] = []

        def flush() -> None:
            nonlocal views
            if not views:
                return
            with self._segment_lock(current_segment):
                fd = os.open(
                    self._segment_path(current_segment),
                    os.O_CREAT | os.O_WRONLY | O_DIRECT,
                    0o644,
                )
                try:
                    self._pwritev_all(fd, views, current_offset, io_batch_blocks)
                    if self.durability == "fsync":
                        os.fsync(fd)
                finally:
                    os.close(fd)
            views = []

        for entry in entries:
            expected_offset = current_offset + len(views) * self.block_size
            if views and (
                entry.segment_id != current_segment or entry.offset != expected_offset
            ):
                flush()
            if not views:
                current_segment, current_offset = entry.segment_id, entry.offset
            offset = entry.block_id * self.block_size
            views.append(byte_view[offset : offset + self.block_size])
        flush()
        # Retain checksums only after this entire write call succeeded. Failed
        # or partial calls cannot authorize publication of their extents.
        with self._thread_lock:
            for extent, checksum in checksums.items():
                lease_id = self._extent_plans.get(extent)
                if lease_id is None:
                    raise ValueError(
                        "store plan completed before checksum write finished"
                    )
                self._plan_checksums[lease_id][extent] = checksum

    def _sync_directory(self) -> None:
        fd = os.open(self.root_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _delete_generation(self) -> int:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE name='delete_generation'"
        ).fetchone()
        return 0 if row is None else int(row[0])

    def complete_store(self, plan: SegmentStorePlan, success: bool) -> None:
        if not plan.entries:
            return
        with self._write_lock():
            fd = self._leases.get(plan.lease_id)
            if fd is None:
                if success:
                    raise ValueError("store plan is no longer active")
                return
            try:
                if not success:
                    return
                # Conservatively reject any pre-delete plan, rather than
                # resurrecting deleted keys from an already queued write.
                if plan.delete_generation != self._delete_generation():
                    raise OSError("store plan invalidated by concurrent delete")
                checksums = self._plan_checksums.get(plan.lease_id, {})
                if self.integrity == "crc32" and any(
                    (entry.segment_id, entry.offset) not in checksums
                    for entry in plan.entries
                ):
                    raise OSError(
                        "store plan has incomplete checksummed writes; index not published"
                    )
                if self.durability == "fsync":
                    self._sync_directory()
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    self._db.executemany(
                        "INSERT OR IGNORE INTO blocks(key, segment_id, offset, length, checksum) "
                        "VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                entry.key,
                                entry.segment_id,
                                entry.offset,
                                self.block_size,
                                checksums.get((entry.segment_id, entry.offset)),
                            )
                            for entry in plan.entries
                        ],
                    )
                    self._db.execute("COMMIT")
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise
            finally:
                self._plan_checksums.pop(plan.lease_id, None)
                for entry in plan.entries:
                    self._extent_plans.pop((entry.segment_id, entry.offset), None)
                del self._leases[plan.lease_id]
                os.close(fd)

    def load_many(
        self,
        keys: Sequence[OffloadKey],
        buffer: memoryview,
        block_ids: Sequence[int],
        locations: Sequence[SegmentLocation | None] | None = None,
        io_batch_blocks: int | None = None,
    ) -> None:
        self._increment("load_jobs")
        fd = None
        read_segment: int | None = None
        try:
            if len(keys) != len(block_ids):
                raise ValueError("keys and block_ids must have the same length")
            if locations is not None and len(keys) != len(locations):
                raise ValueError("keys and locations must have the same length")
            view = self._validate_buffer(buffer, block_ids, writable=True)
            fd = self._activity_fd()
            # Hints may predate another connection's delete/GC/reinsert. Resolve
            # the authoritative rows while a lease prevents physical reclamation.
            locations = self.get_locations(keys)
            if any(location is None for location in locations):
                raise FileNotFoundError("KV block(s) not indexed")
            entries = sorted(
                [
                    (
                        location,
                        view[
                            int(block_id) * self.block_size : (int(block_id) + 1)
                            * self.block_size
                        ],
                    )
                    for location, block_id in zip(locations, block_ids)
                    if location is not None
                ],
                key=lambda entry: (entry[0].segment_id, entry[0].offset),
            )
            index = 0
            while index < len(entries):
                location, destination = entries[index]
                read_segment = location.segment_id
                if location.length != self.block_size:
                    raise OSError("indexed block size does not match store geometry")
                views = [destination]
                end = index + 1
                while end < len(entries):
                    following, destination = entries[end]
                    if (
                        following.segment_id != location.segment_id
                        or following.offset
                        != location.offset + len(views) * self.block_size
                    ):
                        break
                    if following.length != self.block_size:
                        raise OSError(
                            "indexed block size does not match store geometry"
                        )
                    views.append(destination)
                    end += 1
                read_fd = os.open(
                    self._segment_path(location.segment_id), os.O_RDONLY | O_DIRECT
                )
                try:
                    self._preadv_all(read_fd, views, location.offset, io_batch_blocks)
                finally:
                    os.close(read_fd)
                if self.integrity == "crc32":
                    bad_locations = {
                        checked
                        for checked, destination in entries[index:end]
                        if checked.checksum is None
                        or zlib.crc32(destination) != checked.checksum
                    }
                    if bad_locations:
                        self._increment("checksum_errors", len(bad_locations))
                        # CRC failure identifies exact blocks, unlike a short
                        # segment read. Preserve healthy neighbours and any
                        # concurrently re-published replacement rows.
                        read_segment = None
                        self._discard_locations(
                            [
                                (bytes(key), checked)
                                for key, checked in zip(keys, locations)
                                if checked in bad_locations
                            ]
                        )
                        raise OSError(
                            f"CRC32 checksum mismatch for {len(bad_locations)} KV block(s)"
                        )
                index = end
                read_segment = None
            self._increment("load_hits", len(keys))
        except BaseException as error:
            self._increment("load_errors")
            if isinstance(error, OSError) and read_segment is not None:
                assert locations is not None
                self._discard_locations(
                    [
                        (bytes(key), location)
                        for key, location in zip(keys, locations)
                        if location is not None and location.segment_id == read_segment
                    ]
                )
            raise
        finally:
            if fd is not None:
                os.close(fd)

    def _discard_locations(self, records: list[tuple[bytes, SegmentLocation]]) -> None:
        """Invalidate failed reads, but never a concurrently replaced index row.

        Caller buffer/argument errors and plain lookup misses never reach here.
        Healthy keys in a failed segment may be conservatively re-computed.
        """
        with self._write_lock():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.executemany(
                    "DELETE FROM blocks WHERE key=? AND segment_id=? AND offset=? "
                    "AND length=? AND checksum IS ?",
                    [
                        (
                            key,
                            location.segment_id,
                            location.offset,
                            location.length,
                            location.checksum,
                        )
                        for key, location in records
                    ],
                )
                if cursor.rowcount:
                    self._set_metadata(
                        [("delete_generation", self._delete_generation() + 1)]
                    )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    @staticmethod
    def group_writes(entries: Sequence[SegmentWrite]) -> list[tuple[SegmentWrite, ...]]:
        grouped: dict[int, list[SegmentWrite]] = defaultdict(list)
        for entry in entries:
            grouped[entry.segment_id].append(entry)
        return [tuple(group) for group in grouped.values()]

    def delete(self, keys: Collection[OffloadKey]) -> set[OffloadKey]:
        """Remove index rows; return only keys actually indexed before deletion.

        No I/O storage is reused here. A later GC may reclaim fully dead files.
        Any in-flight pre-delete store plan fails conservatively at publication.
        """
        key_bytes = list(dict.fromkeys(bytes(key) for key in keys))
        if not key_bytes:
            return set()
        existing: set[bytes] = set()
        with self._write_lock():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for chunk in self._chunks(key_bytes):
                    placeholders = ",".join("?" for _ in chunk)
                    existing.update(
                        bytes(row[0])
                        for row in self._db.execute(
                            f"SELECT key FROM blocks WHERE key IN ({placeholders})",
                            chunk,
                        )
                    )
                    self._db.execute(
                        f"DELETE FROM blocks WHERE key IN ({placeholders})", chunk
                    )
                self._set_metadata(
                    [("delete_generation", self._delete_generation() + 1)]
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return {OffloadKey(key) for key in existing}

    def collect_garbage(self) -> int:
        """Reclaim fully dead segments, returning reserved bytes released.

        Worker/admin operation, not a scheduler hook. Returns zero when ANY
        process has an in-flight read or prepared store. Current dead shards
        are retired onto fresh monotonic IDs, never reset onto dirty offsets.
        Partial-live segments and SQLite/WAL space are intentionally retained.
        """
        fd = self._activity_fd(exclusive=True)
        if fd is None:
            return 0
        released = 0
        try:
            with self._write_lock():
                metadata = dict(self._db.execute("SELECT name, value FROM metadata"))
                live = {
                    row[0]
                    for row in self._db.execute(
                        "SELECT DISTINCT segment_id FROM blocks"
                    )
                }
                dead = [
                    (segment_id, length)
                    for segment_id, length in self._db.execute(
                        "SELECT segment_id, length FROM segment_extents"
                    )
                    if segment_id not in live
                ]
                next_id = int(metadata.get("next_segment_id", 0))
                for segment_id, length in dead:
                    path = self._segment_path(segment_id)
                    physical = 0
                    try:
                        physical = path.stat().st_size
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    if self.durability == "fsync":
                        self._sync_directory()
                    self._db.execute("BEGIN IMMEDIATE")
                    try:
                        updates = []
                        for name, value in metadata.items():
                            if (
                                name.endswith("segment_id")
                                and name != "next_segment_id"
                                and value == segment_id
                            ):
                                updates.extend(
                                    [
                                        (name, next_id),
                                        (
                                            name.removesuffix("segment_id")
                                            + "segment_offset",
                                            0,
                                        ),
                                    ]
                                )
                                next_id += 1
                        updates.append(("next_segment_id", next_id))
                        self._set_metadata(updates)
                        metadata.update(updates)
                        self._db.execute(
                            "DELETE FROM segment_extents WHERE segment_id=?",
                            (segment_id,),
                        )
                        self._db.execute("COMMIT")
                    except BaseException:
                        self._db.execute("ROLLBACK")
                        raise
                    released += length
                    self._increment("gc_bytes", physical)
                    self._increment("gc_segments")
        finally:
            os.close(fd)
        return released

    def close(self) -> None:
        """Close after draining all users; abandoned local plans are cancelled."""
        with self._thread_lock:
            if self._closed:
                return
            self._closed = True
            for fd in self._leases.values():
                os.close(fd)
            self._leases.clear()
            self._plan_checksums.clear()
            self._extent_plans.clear()
            self._db.close()
            os.close(self._lock_fd)
