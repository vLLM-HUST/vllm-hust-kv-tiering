# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Collection, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from vllm.v1.kv_offload.base import OffloadKey

O_DIRECT = getattr(os, "O_DIRECT", 0)
_DEFAULT_SEGMENT_SIZE = 1024**3
_INDEX_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class SegmentLocation:
    segment_id: int
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class SegmentWrite:
    key: bytes
    segment_id: int
    offset: int
    block_id: int


@dataclass(frozen=True, slots=True)
class SegmentStorePlan:
    entries: tuple[SegmentWrite, ...]


class SegmentStore:
    """Append-only segment storage with a persistent block index."""

    def __init__(
        self,
        root_dir: str,
        block_size: int,
        segment_size_bytes: int = _DEFAULT_SEGMENT_SIZE,
        segment_shards: int = 1,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if segment_size_bytes < block_size:
            raise ValueError("segment_size_bytes must be at least one block")
        if segment_shards <= 0:
            raise ValueError("segment_shards must be greater than zero")

        self.block_size = block_size
        self.segment_size_bytes = (
            segment_size_bytes // block_size
        ) * block_size
        self.segment_shards = segment_shards
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root_dir / "segments.lock"
        self._lock_fd = os.open(
            self._lock_path,
            os.O_CREAT | os.O_RDWR,
            0o644,
        )
        self._thread_lock = threading.RLock()
        self._segment_locks: dict[int, threading.Lock] = {}
        self._closed = False
        self._db = sqlite3.connect(
            self.root_dir / "index.sqlite3",
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA temp_store=MEMORY")
        # An async existence lookup is immediately followed by promotion on
        # a secondary-tier hit.  Cache only positive results so that
        # submit_load() can reuse the location without another SQLite query
        # on the scheduler critical path.  Misses stay uncached because a
        # different process sharing the tier may append the key at any time.
        self._location_cache: dict[bytes, SegmentLocation] = {}
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS blocks (
                key BLOB PRIMARY KEY,
                segment_id INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                length INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )

    @contextmanager
    def _write_lock(self):
        with self._thread_lock:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _segment_path(self, segment_id: int) -> Path:
        return self.root_dir / f"segment-{segment_id:08d}.bin"

    def _segment_lock(self, segment_id: int) -> threading.Lock:
        with self._thread_lock:
            return self._segment_locks.setdefault(segment_id, threading.Lock())

    def _latest_segment(self) -> tuple[int, int]:
        rows = dict(
            self._db.execute(
                "SELECT name, value FROM metadata "
                "WHERE name IN ('segment_id', 'segment_offset')"
            )
        )
        if len(rows) == 2:
            return int(rows["segment_id"]), int(rows["segment_offset"])

        row = self._db.execute(
            "SELECT segment_id, MAX(offset + length) FROM blocks "
            "GROUP BY segment_id ORDER BY segment_id DESC LIMIT 1"
        ).fetchone()
        return (0, 0) if row is None else (int(row[0]), int(row[1]))

    def _allocation_state(
        self,
    ) -> tuple[list[tuple[int, int]], int, int]:
        metadata = dict(self._db.execute("SELECT name, value FROM metadata"))
        has_shard_state = "shard_0_segment_id" in metadata
        if has_shard_state:
            existing_segment_ids = [
                int(value)
                for name, value in metadata.items()
                if name.startswith("shard_") and name.endswith("_segment_id")
            ]
            legacy_segment = (0, 0)
        else:
            legacy_segment = self._latest_segment()
            existing_segment_ids = [legacy_segment[0]]
        next_segment_id = max(
            max(existing_segment_ids, default=-1) + 1,
            int(metadata.get("next_segment_id", 0)),
        )
        shard_state: list[tuple[int, int]] = []
        for shard_id in range(self.segment_shards):
            segment_key = f"shard_{shard_id}_segment_id"
            offset_key = f"shard_{shard_id}_segment_offset"
            if segment_key in metadata and offset_key in metadata:
                shard_state.append(
                    (int(metadata[segment_key]), int(metadata[offset_key]))
                )
            elif shard_id == 0 and not has_shard_state:
                shard_state.append(legacy_segment)
                next_segment_id = max(next_segment_id, legacy_segment[0] + 1)
            else:
                shard_state.append((next_segment_id, 0))
                next_segment_id += 1
        cursor = int(metadata.get("shard_cursor", 0)) % self.segment_shards
        return shard_state, next_segment_id, cursor

    @staticmethod
    def _chunks(values: Sequence[bytes]) -> list[Sequence[bytes]]:
        return [
            values[start : start + _INDEX_BATCH_SIZE]
            for start in range(0, len(values), _INDEX_BATCH_SIZE)
        ]

    def get_locations(
        self, keys: Sequence[OffloadKey]
    ) -> list[SegmentLocation | None]:
        if not keys:
            return []
        key_bytes = [bytes(key) for key in keys]
        locations: dict[bytes, SegmentLocation] = {}
        with self._thread_lock:
            missing_keys: list[bytes] = []
            for key in key_bytes:
                location = self._location_cache.get(key)
                if location is None:
                    missing_keys.append(key)
                else:
                    locations[key] = location
            for chunk in self._chunks(missing_keys):
                placeholders = ",".join("?" for _ in chunk)
                rows = self._db.execute(
                    "SELECT key, segment_id, offset, length "
                    f"FROM blocks WHERE key IN ({placeholders})",
                    chunk,
                )
                for key, segment_id, offset, length in rows:
                    location = SegmentLocation(
                        segment_id=int(segment_id),
                        offset=int(offset),
                        length=int(length),
                    )
                    key_bytes_value = bytes(key)
                    locations[key_bytes_value] = location
                    self._location_cache[key_bytes_value] = location
        return [locations.get(key) for key in key_bytes]

    def contains_many(self, keys: Sequence[OffloadKey]) -> list[bool]:
        return [location is not None for location in self.get_locations(keys)]

    @staticmethod
    def _iov_max() -> int:
        try:
            return int(os.sysconf("SC_IOV_MAX"))
        except (ValueError, OSError):
            return 1024

    @classmethod
    def _pwritev_all(
        cls,
        fd: int,
        buffers: Sequence[memoryview],
        offset: int,
        max_buffers: int | None = None,
    ) -> None:
        iov_max = cls._iov_max()
        if max_buffers is not None:
            iov_max = min(iov_max, max_buffers)
        for start in range(0, len(buffers), iov_max):
            chunk = buffers[start : start + iov_max]
            expected = sum(len(view) for view in chunk)
            written = os.pwritev(fd, chunk, offset)
            if written != expected:
                raise OSError(f"Short write: expected {expected}, wrote {written}")
            offset += written

    @classmethod
    def _preadv_all(
        cls,
        fd: int,
        buffers: Sequence[memoryview],
        offset: int,
        max_buffers: int | None = None,
    ) -> None:
        iov_max = cls._iov_max()
        if max_buffers is not None:
            iov_max = min(iov_max, max_buffers)
        for start in range(0, len(buffers), iov_max):
            chunk = buffers[start : start + iov_max]
            expected = sum(len(view) for view in chunk)
            bytes_read = os.preadv(fd, chunk, offset)
            if bytes_read != expected:
                raise OSError(
                    f"Short read: expected {expected}, read {bytes_read}"
                )
            offset += bytes_read

    def store_many(
        self,
        keys: Sequence[OffloadKey],
        buffer: memoryview,
        block_ids: Sequence[int],
    ) -> None:
        plan = self.prepare_store(keys, block_ids)
        try:
            self.write_entries(plan.entries, buffer)
        except Exception:
            self.complete_store(plan, False)
            raise
        self.complete_store(plan, True)

    def prepare_store(
        self,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
    ) -> SegmentStorePlan:
        if len(keys) != len(block_ids):
            raise ValueError("keys and block_ids must have the same length")
        if not keys:
            return SegmentStorePlan(())

        with self._write_lock():
            existing = self.contains_many(keys)
            unique: dict[bytes, tuple[OffloadKey, int]] = {}
            for key, block_id, is_stored in zip(keys, block_ids, existing):
                key_bytes = bytes(key)
                if not is_stored and key_bytes not in unique:
                    unique[key_bytes] = (key, int(block_id))
            if not unique:
                return SegmentStorePlan(())

            shard_state, next_segment_id, cursor = self._allocation_state()
            entries_by_shard: list[list[SegmentWrite]] = [
                [] for _ in range(self.segment_shards)
            ]
            for index, (key_bytes, (_, block_id)) in enumerate(unique.items()):
                shard_id = (cursor + index) % self.segment_shards
                segment_id, segment_offset = shard_state[shard_id]
                if segment_offset + self.block_size > self.segment_size_bytes:
                    segment_id = next_segment_id
                    next_segment_id += 1
                    segment_offset = 0
                entries_by_shard[shard_id].append(
                    SegmentWrite(
                        key=key_bytes,
                        segment_id=segment_id,
                        offset=segment_offset,
                        block_id=block_id,
                    )
                )
                segment_offset += self.block_size
                shard_state[shard_id] = (segment_id, segment_offset)

            entries = [
                entry for shard_entries in entries_by_shard for entry in shard_entries
            ]
            cursor = (cursor + len(unique)) % self.segment_shards

            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.executemany(
                    "INSERT INTO metadata(name, value) VALUES (?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                    [
                        ("next_segment_id", next_segment_id),
                        ("shard_cursor", cursor),
                    ]
                    + [
                        (f"shard_{shard_id}_segment_id", segment_id)
                        for shard_id, (segment_id, _) in enumerate(shard_state)
                    ]
                    + [
                        (f"shard_{shard_id}_segment_offset", segment_offset)
                        for shard_id, (_, segment_offset) in enumerate(shard_state)
                    ],
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            return SegmentStorePlan(tuple(entries))

    def write_entries(
        self,
        entries: Sequence[SegmentWrite],
        buffer: memoryview,
        io_batch_blocks: int | None = None,
    ) -> None:
        if not entries:
            return
        byte_view = buffer.cast("B")
        current_segment = entries[0].segment_id
        current_offset = entries[0].offset
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
                    self._pwritev_all(
                        fd,
                        views,
                        current_offset,
                        io_batch_blocks,
                    )
                finally:
                    os.close(fd)
            views = []

        for entry in entries:
            expected_offset = current_offset + len(views) * self.block_size
            if views and (
                entry.segment_id != current_segment
                or entry.offset != expected_offset
            ):
                flush()
            if not views:
                current_segment = entry.segment_id
                current_offset = entry.offset
            source_offset = entry.block_id * self.block_size
            views.append(byte_view[source_offset : source_offset + self.block_size])
        flush()

    def complete_store(self, plan: SegmentStorePlan, success: bool) -> None:
        if not success or not plan.entries:
            return
        records = [
            (entry.key, entry.segment_id, entry.offset, self.block_size)
            for entry in plan.entries
        ]
        with self._write_lock():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.executemany(
                    "INSERT OR IGNORE INTO blocks"
                    "(key, segment_id, offset, length) VALUES (?, ?, ?, ?)",
                    records,
                )
                self._db.execute("COMMIT")
                # INSERT OR IGNORE permits another process to win a racing
                # append. Read back the durable rows rather than assuming our
                # planned offsets won, otherwise a cached location could point
                # to a never-indexed write.
                self.get_locations([OffloadKey(entry.key) for entry in plan.entries])
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def load_many(
        self,
        keys: Sequence[OffloadKey],
        buffer: memoryview,
        block_ids: Sequence[int],
        locations: Sequence[SegmentLocation | None] | None = None,
        io_batch_blocks: int | None = None,
    ) -> None:
        if len(keys) != len(block_ids):
            raise ValueError("keys and block_ids must have the same length")
        if locations is None:
            locations = self.get_locations(keys)
        elif len(keys) != len(locations):
            raise ValueError("keys and locations must have the same length")
        missing = [key for key, location in zip(keys, locations) if location is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} KV block(s) not indexed")

        byte_view = buffer.cast("B")
        entries = [
            (
                location,
                byte_view[
                    int(block_id)
                    * self.block_size : (int(block_id) + 1)
                    * self.block_size
                ],
            )
            for location, block_id in zip(locations, block_ids)
            if location is not None
        ]
        entries.sort(key=lambda entry: (entry[0].segment_id, entry[0].offset))

        current_segment = -1
        current_offset = -1
        current_length = 0
        current_views: list[memoryview] = []

        def flush() -> None:
            nonlocal current_views
            if not current_views:
                return
            fd = os.open(
                self._segment_path(current_segment),
                os.O_RDONLY | O_DIRECT,
            )
            try:
                self._preadv_all(
                    fd,
                    current_views,
                    current_offset,
                    io_batch_blocks,
                )
            finally:
                os.close(fd)
            current_views = []

        try:
            for location, destination in entries:
                if location.length != self.block_size:
                    raise OSError(
                        f"Indexed block size {location.length} does not match "
                        f"expected size {self.block_size}"
                    )
                expected_offset = current_offset + current_length
                if (
                    current_views
                    and (
                        location.segment_id != current_segment
                        or location.offset != expected_offset
                    )
                ):
                    flush()
                if not current_views:
                    current_segment = location.segment_id
                    current_offset = location.offset
                    current_length = 0
                current_views.append(destination)
                current_length += len(destination)
            flush()
        except Exception:
            self.delete(keys)
            raise

    @staticmethod
    def group_writes(
        entries: Sequence[SegmentWrite],
    ) -> list[tuple[SegmentWrite, ...]]:
        grouped: dict[int, list[SegmentWrite]] = defaultdict(list)
        for entry in entries:
            grouped[entry.segment_id].append(entry)
        return [tuple(group) for group in grouped.values()]

    def delete(self, keys: Collection[OffloadKey]) -> set[OffloadKey]:
        key_bytes = list(dict.fromkeys(bytes(key) for key in keys))
        if not key_bytes:
            return set()
        existing: set[bytes] = set()
        with self._write_lock():
            for chunk in self._chunks(key_bytes):
                placeholders = ",".join("?" for _ in chunk)
                existing.update(
                    bytes(row[0])
                    for row in self._db.execute(
                        f"SELECT key FROM blocks WHERE key IN ({placeholders})",
                        chunk,
                    )
                )
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for chunk in self._chunks(key_bytes):
                    placeholders = ",".join("?" for _ in chunk)
                    self._db.execute(
                        f"DELETE FROM blocks WHERE key IN ({placeholders})",
                        chunk,
                    )
                self._db.execute("COMMIT")
                for key in existing:
                    self._location_cache.pop(key, None)
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return {OffloadKey(key) for key in existing}

    def close(self) -> None:
        with self._thread_lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()
            os.close(self._lock_fd)
