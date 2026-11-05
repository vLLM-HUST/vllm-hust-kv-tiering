# SPDX-License-Identifier: Apache-2.0
"""CPU-only storage tests: no vLLM, torch, accelerator or package import.

Only the leaf module is loaded, with a temporary OffloadKey stub. All temporary
sys.modules entries are restored before yielding, so canonical host tests in
the same pytest process cannot accidentally inherit a fake vLLM module.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
import threading
import types
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def storage_module(monkeypatch):
    name = "_segment_store_safety_" + uuid.uuid4().hex
    source = (
        Path(__file__).resolve().parents[1]
        / "src/vllm_hust_kv_tiering/fs/segment_store.py"
    )
    stub = types.ModuleType("vllm.v1.kv_offload.base")
    stub.OffloadKey = bytes
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, stub.__name__, stub)
        imports.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "O_DIRECT", 0)
    return module


@pytest.fixture
def stores(storage_module, tmp_path):
    opened = []

    def create(**kwargs):
        options = {"root_dir": str(tmp_path), "block_size": 4, "segment_size_bytes": 8}
        options.update(kwargs)
        store = storage_module.SegmentStore(**options)
        opened.append(store)
        return store

    yield create
    for store in opened:
        store.close()


def put(store, keys, data=None):
    if data is None:
        data = b"abcd" * len(keys)
    store.store_many(keys, memoryview(data), list(range(len(keys))))


def get(store, keys, **kwargs):
    destination = bytearray(4 * len(keys))
    store.load_many(keys, memoryview(destination), list(range(len(keys))), **kwargs)
    return bytes(destination)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0},
        {"block_size": True},
        {"segment_size_bytes": 3},
        {"segment_size_bytes": 4.5},
        {"segment_shards": 0},
        {"segment_shards": True},
        {"max_storage_bytes": 0},
        {"max_storage_bytes": 3},
        {"max_storage_bytes": True},
        {"max_storage_bytes": 8.0},
        {"durability": "sync"},
        {"durability": []},
        {"durability": None},
    ],
)
def test_invalid_configuration(stores, kwargs):
    with pytest.raises(ValueError):
        stores(**kwargs)


def test_leaf_import_does_not_pollute_vllm(storage_module):
    assert all(value is not storage_module for value in sys.modules.values())
    assert (
        getattr(sys.modules.get("vllm.v1.kv_offload.base"), "OffloadKey", None)
        is not bytes
    )


def test_cross_instance_delete_and_reinsert_are_coherent(stores):
    first, second = stores(), stores()
    put(first, [b"key"], b"old!")
    stale = first.get_locations([b"key"])
    assert second.delete([b"key"]) == {b"key"}
    assert first.contains_many([b"key"]) == [False]
    with pytest.raises(FileNotFoundError):
        get(first, [b"key"], locations=stale)
    second.collect_garbage()
    put(second, [b"key"], b"new!")
    assert get(first, [b"key"], locations=stale) == b"new!"
    assert first.get_locations([b"key"]) != stale


def test_quota_rejects_whole_job_without_index_or_allocation_loss(
    stores, storage_module
):
    store = stores(max_storage_bytes=8)
    put(store, [b"live"])
    with pytest.raises(storage_module.StorageCapacityError):
        put(store, [b"new1", b"new2"])
    assert store.contains_many([b"live", b"new1", b"new2"]) == [True, False, False]
    assert get(store, [b"live"]) == b"abcd"
    put(store, [b"fits"])
    assert (
        store._db.execute("SELECT SUM(length) FROM segment_extents").fetchone()[0] == 8
    )


def test_cross_instance_quota_counts_inflight_reservations(stores, storage_module):
    first, second = stores(max_storage_bytes=4), stores(max_storage_bytes=4)
    plan = first.prepare_store([b"pending"], [0])
    with pytest.raises(storage_module.StorageCapacityError):
        second.prepare_store([b"other"], [0])
    assert second.collect_garbage() == 0
    first.complete_store(plan, False)
    assert second.collect_garbage() == 4
    put(second, [b"other"])
    assert get(second, [b"other"]) == b"abcd"


def test_gc_reclaims_only_fully_dead_segments(stores):
    store = stores(max_storage_bytes=12)
    put(store, [b"a", b"b", b"c"], b"aaaabbbbcccc")
    first_segment = store.get_locations([b"a"])[0].segment_id
    store.delete([b"a"])
    assert store.collect_garbage() == 0
    assert store._segment_path(first_segment).exists()
    store.delete([b"b"])
    assert store.collect_garbage() == 8
    assert not store._segment_path(first_segment).exists()
    assert get(store, [b"c"]) == b"cccc"
    put(store, [b"d", b"e"], b"ddddeeee")
    assert get(store, [b"c", b"d", b"e"]) == b"ccccddddeeee"


def test_gc_retires_dead_current_shard_without_reusing_id(stores):
    store = stores(max_storage_bytes=4)
    put(store, [b"a"])
    previous = store.get_locations([b"a"])[0]
    store.delete([b"a"])
    # The next store automatically attempts dead-segment GC on quota pressure.
    put(store, [b"b"], b"bbbb")
    current = store.get_locations([b"b"])[0]
    assert current.segment_id > previous.segment_id
    assert not store._segment_path(previous.segment_id).exists()
    assert get(store, [b"b"]) == b"bbbb"


def test_gc_skips_pending_store_across_instances(stores):
    first, second = stores(), stores()
    plan = first.prepare_store([b"pending"], [0])
    assert second.collect_garbage() == 0
    first.write_entries(plan.entries, memoryview(b"data"))
    assert second.collect_garbage() == 0
    first.complete_store(plan, True)
    assert second.collect_garbage() == 0
    assert get(second, [b"pending"]) == b"data"


def test_gc_skips_inflight_read_even_after_delete(stores, monkeypatch):
    reader, collector = stores(), stores()
    put(reader, [b"key"])
    entered, release = threading.Event(), threading.Event()
    original = reader._preadv_all
    errors = []

    def paused_read(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(reader, "_preadv_all", paused_read)

    def read():
        try:
            assert get(reader, [b"key"]) == b"abcd"
        except (AssertionError, OSError) as error:
            errors.append(error)

    thread = threading.Thread(target=read)
    thread.start()
    try:
        assert entered.wait(5)
        collector.delete([b"key"])
        assert collector.collect_garbage() == 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert collector.collect_garbage() == 4


def test_failed_write_is_unpublished_charged_and_not_reused(stores, monkeypatch):
    store = stores()
    original = store._pwritev_all

    def partial(fd, views, offset, max_buffers):
        import os

        os.pwrite(fd, b"xx", offset)
        raise OSError("injected partial write")

    monkeypatch.setattr(store, "_pwritev_all", partial)
    with pytest.raises(OSError, match="partial"):
        put(store, [b"failed"])
    assert store.contains_many([b"failed"]) == [False]
    failed_extent = store._db.execute(
        "SELECT segment_id, length FROM segment_extents"
    ).fetchone()
    assert failed_extent[1] == 4
    monkeypatch.setattr(store, "_pwritev_all", original)
    put(store, [b"good"], b"good")
    location = store.get_locations([b"good"])[0]
    assert (location.segment_id, location.offset) != (failed_extent[0], 0)
    assert get(store, [b"good"]) == b"good"


def test_abandoned_plan_can_be_reclaimed_after_close(stores):
    first = stores(max_storage_bytes=4)
    plan = first.prepare_store([b"abandoned"], [0])
    first.write_entries(plan.entries, memoryview(b"data"))
    first.close()
    second = stores(max_storage_bytes=4)
    assert second.contains_many([b"abandoned"]) == [False]
    assert second.collect_garbage() == 4
    put(second, [b"new!"])
    assert get(second, [b"new!"]) == b"abcd"


def test_delete_invalidates_prepared_store_without_resurrection(stores):
    first, second = stores(), stores()
    plan = first.prepare_store([b"pending"], [0])
    first.write_entries(plan.entries, memoryview(b"data"))
    assert second.delete([b"pending"]) == set()
    with pytest.raises(OSError, match="invalidated"):
        first.complete_store(plan, True)
    assert first.contains_many([b"pending"]) == [False]
    assert second.collect_garbage() == 4


def test_reopen_never_overwrites_orphaned_dirty_file(stores, tmp_path):
    orphan = tmp_path / "segment-00000000.bin"
    orphan.write_bytes(b"dirty!!!")
    store = stores()
    put(store, [b"live"], b"live")
    assert orphan.read_bytes() == b"dirty!!!"
    assert store.get_locations([b"live"])[0].segment_id != 0
    assert get(store, [b"live"]) == b"live"


def test_buffer_errors_preserve_rows_but_corrupt_reads_invalidate(stores):
    store = stores()
    put(store, [b"key"])
    with pytest.raises(ValueError, match="outside"):
        store.load_many([b"key"], memoryview(bytearray(3)), [0])
    assert store.contains_many([b"key"]) == [True]
    location = store.get_locations([b"key"])[0]
    store._segment_path(location.segment_id).write_bytes(b"x")
    with pytest.raises(OSError, match="Short read"):
        get(store, [b"key"])
    assert store.contains_many([b"key"]) == [False]
    put(store, [b"key"], b"good")
    assert get(store, [b"key"]) == b"good"


def test_read_failure_never_deletes_concurrent_healthy_replacement(stores, monkeypatch):
    first, second = stores(), stores()
    put(first, [b"key"], b"old!")
    original = first._preadv_all

    def replace_then_fail(*args):
        second.delete([b"key"])
        put(second, [b"key"], b"new!")
        raise OSError("old read failed")

    monkeypatch.setattr(first, "_preadv_all", replace_then_fail)
    with pytest.raises(OSError, match="old read failed"):
        get(first, [b"key"])
    monkeypatch.setattr(first, "_preadv_all", original)
    assert get(first, [b"key"]) == b"new!"


def test_duplicate_keys_do_not_double_charge(stores):
    store = stores(max_storage_bytes=4)
    put(store, [b"same", b"same"], b"aaaabbbb")
    assert get(store, [b"same"]) == b"aaaa"
    put(store, [b"same"], b"cccc")
    assert get(store, [b"same"]) == b"aaaa"
    assert store.get_storage_counters()["store_bytes"] == 4


def test_counters_report_actual_bytes_hits_and_failures(stores, storage_module):
    store = stores(max_storage_bytes=4)
    put(store, [b"key"])
    assert get(store, [b"key"]) == b"abcd"
    with pytest.raises(FileNotFoundError):
        get(store, [b"missing"])
    with pytest.raises(storage_module.StorageCapacityError):
        put(store, [b"full"])
    counters = store.get_storage_counters()
    assert counters["store_bytes"] == 4
    assert counters["load_bytes"] == 4
    assert counters["load_hits"] == 1
    assert counters["load_jobs"] == 2
    assert counters["store_jobs"] == 2
    assert counters["load_errors"] == 1
    assert counters["store_errors"] == 1
    assert counters["capacity_rejections"] >= 1
    assert store.get_storage_counters(reset=True) == counters
    assert not any(store.get_storage_counters().values())


def test_fsync_precedes_index_publication(stores, monkeypatch, storage_module):
    store = stores(durability="fsync")
    events = []
    monkeypatch.setattr(storage_module.os, "fsync", lambda fd: events.append("fsync"))
    store._db.set_trace_callback(lambda statement: events.append(statement))
    put(store, [b"key"])
    publish = next(
        index
        for index, event in enumerate(events)
        if event.startswith("INSERT OR IGNORE INTO blocks")
    )
    assert events[:publish].count("fsync") == 2  # data, then directory
    assert store._db.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


def test_fsync_failure_does_not_publish(stores, monkeypatch, storage_module):
    store = stores(durability="fsync")

    def fail(fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(storage_module.os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failed"):
        put(store, [b"key"])
    assert store.contains_many([b"key"]) == [False]
    assert not store._leases


def test_none_durability_does_not_call_fsync(stores, monkeypatch, storage_module):
    store = stores()

    def unexpected(fd):
        pytest.fail("default disposable-cache mode must not claim fsync durability")

    monkeypatch.setattr(storage_module.os, "fsync", unexpected)
    put(store, [b"key"])
    assert get(store, [b"key"]) == b"abcd"


def test_process_lease_blocks_gc_and_delete_prevents_publish(
    stores, storage_module, tmp_path
):
    store = stores(max_storage_bytes=4)
    child_code = r"""
import importlib.util, sys, types
stub = types.ModuleType("vllm.v1.kv_offload.base")
stub.OffloadKey = bytes
sys.modules[stub.__name__] = stub
spec = importlib.util.spec_from_file_location("isolated_store", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.O_DIRECT = 0
store = module.SegmentStore(sys.argv[2], 4, 8, max_storage_bytes=4)
plan = store.prepare_store([b"pending"], [0])
store.write_entries(plan.entries, memoryview(b"data"))
print("ready", flush=True)
sys.stdin.readline()
try:
    store.complete_store(plan, True)
except OSError:
    print("invalidated", flush=True)
finally:
    store.close()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, storage_module.__file__, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert store.collect_garbage() == 0
        assert store.delete([b"pending"]) == set()
        stdout, stderr = child.communicate("complete\n", timeout=10)
        assert child.returncode == 0, stderr
        assert stdout.strip() == "invalidated"
        assert store.collect_garbage() == 4
        assert store.contains_many([b"pending"]) == [False]
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_counter_snapshot_does_not_wait_for_metadata_lock(stores):
    store = stores()
    locked, release = threading.Event(), threading.Event()

    def lock_metadata():
        with store._thread_lock:
            locked.set()
            assert release.wait(5)

    thread = threading.Thread(target=lock_metadata)
    thread.start()
    try:
        assert locked.wait(5)
        assert store.get_storage_counters()["load_bytes"] == 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()


def test_gc_unlink_failure_retains_quota_and_live_data(stores, monkeypatch):
    store = stores()
    put(store, [b"dead", b"live", b"last"])
    store.delete([b"last"])
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path.name.startswith("segment-"):
            raise PermissionError("injected unlink failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(PermissionError):
        store.collect_garbage()
    assert (
        store._db.execute("SELECT SUM(length) FROM segment_extents").fetchone()[0] == 12
    )
    assert get(store, [b"live"]) == b"abcd"


def test_existing_block_size_mismatch_rejected(stores):
    stores()
    with pytest.raises(ValueError, match="block_size"):
        stores(block_size=8)


@pytest.mark.parametrize(
    "first, second",
    [
        ({"max_storage_bytes": 8}, {"max_storage_bytes": None}),
        ({"max_storage_bytes": None}, {"max_storage_bytes": 8}),
        ({"max_storage_bytes": 8}, {"max_storage_bytes": 12}),
        ({"durability": "none"}, {"durability": "fsync"}),
        ({"durability": "fsync"}, {"durability": "none"}),
    ],
)
def test_shared_quota_and_durability_configuration_must_agree(stores, first, second):
    stores(**first)
    with pytest.raises(ValueError, match="does not match"):
        stores(**second)


def test_multiple_shards_and_reopen_roundtrip(stores):
    first = stores(segment_shards=3, max_storage_bytes=40)
    keys = [bytes([index]) for index in range(9)]
    data = b"".join(bytes([index]) * 4 for index in range(9))
    put(first, keys, data)
    first.close()
    second = stores(segment_shards=1, max_storage_bytes=40)
    assert get(second, keys) == data
    put(second, [b"last"], b"last")
    assert get(second, keys + [b"last"]) == data + b"last"


def test_legacy_index_is_accounted_and_never_overwritten(stores, tmp_path):
    db = sqlite3.connect(tmp_path / "index.sqlite3")
    db.execute(
        "CREATE TABLE blocks (key BLOB PRIMARY KEY, segment_id INTEGER, offset INTEGER, length INTEGER)"
    )
    db.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value INTEGER)")
    db.execute("INSERT INTO blocks VALUES (?, 0, 0, 4)", (b"old",))
    db.commit()
    db.close()
    (tmp_path / "segment-00000000.bin").write_bytes(b"old!")
    store = stores(max_storage_bytes=8)
    put(store, [b"new"], b"new!")
    assert get(store, [b"old", b"new"]) == b"old!new!"


@pytest.mark.parametrize("integrity", [None, True, [], "sha256"])
def test_invalid_integrity_mode_rejected(stores, integrity):
    with pytest.raises(ValueError, match="integrity"):
        stores(integrity=integrity)


def test_crc32_roundtrip_persists_checksum_and_releases_plan_metadata(stores):
    import zlib

    store = stores(integrity="crc32")
    put(store, [b"key"], b"data")
    location = store.get_locations([b"key"])[0]
    assert location.checksum == zlib.crc32(b"data")
    assert get(store, [b"key"]) == b"data"
    assert not store._plan_checksums and not store._extent_plans and not store._leases
    store.close()
    reopened = stores(integrity="crc32")
    assert get(reopened, [b"key"]) == b"data"


def test_crc32_same_size_bitflip_invalidates_only_bad_block(stores):
    store = stores(integrity="crc32")
    put(store, [b"bad", b"healthy"], b"aaaabbbb")
    location = store.get_locations([b"bad"])[0]
    path = store._segment_path(location.segment_id)
    data = bytearray(path.read_bytes())
    data[location.offset] ^= 1
    path.write_bytes(data)
    with pytest.raises(OSError, match="CRC32 checksum mismatch"):
        get(store, [b"bad", b"healthy"])
    assert path.stat().st_size == 8
    assert store.contains_many([b"bad", b"healthy"]) == [False, True]
    assert get(store, [b"healthy"]) == b"bbbb"
    assert store.get_storage_counters()["checksum_errors"] == 1
    put(store, [b"bad"], b"good")
    assert get(store, [b"bad"]) == b"good"


def test_crc32_missing_checksum_is_never_trusted(stores):
    store = stores(integrity="crc32")
    put(store, [b"bad", b"good"], b"aaaabbbb")
    store._db.execute("UPDATE blocks SET checksum=NULL WHERE key=?", (b"bad",))
    with pytest.raises(OSError, match="checksum mismatch"):
        get(store, [b"bad", b"good"])
    assert store.contains_many([b"bad", b"good"]) == [False, True]


def test_crc32_incomplete_plan_is_not_published_and_cleans_all_temporary_maps(stores):
    store = stores(integrity="crc32")
    plan = store.prepare_store([b"first", b"unwritten"], [0, 1])
    store.write_entries(plan.entries[:1], memoryview(b"aaaabbbb"))
    assert len(store._plan_checksums[plan.lease_id]) == 1
    with pytest.raises(OSError, match="incomplete checksummed writes"):
        store.complete_store(plan, True)
    assert store.contains_many([b"first", b"unwritten"]) == [False, False]
    assert not store._plan_checksums and not store._extent_plans and not store._leases


def test_crc32_failed_store_is_unpublished_and_cleans_maps(stores, monkeypatch):
    store = stores(integrity="crc32", segment_shards=2)
    original = store._pwritev_all
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original(*args)

    monkeypatch.setattr(store, "_pwritev_all", fail_second)
    with pytest.raises(OSError, match="injected"):
        put(store, [b"a", b"b"], b"aaaabbbb")
    assert store.contains_many([b"a", b"b"]) == [False, False]
    assert not store._plan_checksums and not store._extent_plans and not store._leases


def test_crc32_grouped_plan_writes_keep_checksums_until_publication(stores):
    store = stores(integrity="crc32", segment_shards=2)
    plan = store.prepare_store([b"a", b"b"], [0, 1])
    for group in store.group_writes(plan.entries):
        store.write_entries(group, memoryview(b"aaaabbbb"))
    assert len(store._plan_checksums[plan.lease_id]) == 2
    assert store.contains_many([b"a", b"b"]) == [False, False]
    store.complete_store(plan, True)
    assert get(store, [b"a", b"b"]) == b"aaaabbbb"
    assert not store._plan_checksums and not store._extent_plans


@pytest.mark.parametrize("first, second", [("none", "crc32"), ("crc32", "none")])
def test_integrity_mode_is_immutable_shared_store_configuration(stores, first, second):
    stores(integrity=first)
    with pytest.raises(ValueError, match="fresh namespace"):
        stores(integrity=second)


def test_crc32_refuses_unverified_legacy_index_without_destroying_rows(
    stores, tmp_path
):
    db = sqlite3.connect(tmp_path / "index.sqlite3")
    db.execute(
        "CREATE TABLE blocks (key BLOB PRIMARY KEY, segment_id INTEGER, offset INTEGER, length INTEGER)"
    )
    db.execute("INSERT INTO blocks VALUES (?, 0, 0, 4)", (b"legacy",))
    db.commit()
    db.close()
    (tmp_path / "segment-00000000.bin").write_bytes(b"old!")
    with pytest.raises(ValueError, match="without checksums.*fresh namespace"):
        stores(integrity="crc32")
    db = sqlite3.connect(tmp_path / "index.sqlite3")
    assert db.execute("SELECT key, checksum FROM blocks").fetchone() == (
        b"legacy",
        None,
    )
    db.close()
    unverified = stores(integrity="none")
    assert get(unverified, [b"legacy"]) == b"old!"


def test_crc32_can_migrate_empty_old_schema(stores, tmp_path):
    db = sqlite3.connect(tmp_path / "index.sqlite3")
    db.execute(
        "CREATE TABLE blocks (key BLOB PRIMARY KEY, segment_id INTEGER, offset INTEGER, length INTEGER)"
    )
    db.close()
    store = stores(integrity="crc32")
    put(store, [b"new"], b"data")
    assert get(store, [b"new"]) == b"data"


def test_crc32_failed_old_read_preserves_concurrent_verified_replacement(
    stores, monkeypatch
):
    first, second = stores(integrity="crc32"), stores(integrity="crc32")
    put(first, [b"key"], b"old!")
    original = first._preadv_all

    def replace_and_corrupt_old_read(fd, views, offset, max_buffers):
        original(fd, views, offset, max_buffers)
        second.delete([b"key"])
        put(second, [b"key"], b"new!")
        views[0][0] ^= 1

    monkeypatch.setattr(first, "_preadv_all", replace_and_corrupt_old_read)
    with pytest.raises(OSError, match="checksum mismatch"):
        get(first, [b"key"])
    monkeypatch.setattr(first, "_preadv_all", original)
    assert get(first, [b"key"]) == b"new!"


def test_crc32_plan_failure_and_close_release_checksum_metadata(stores):
    store = stores(integrity="crc32")
    plan = store.prepare_store([b"key"], [0])
    store.write_entries(plan.entries, memoryview(b"data"))
    assert store._plan_checksums
    store.complete_store(plan, False)
    assert not store._plan_checksums and not store._extent_plans and not store._leases
    plan = store.prepare_store([b"other"], [0])
    store.write_entries(plan.entries, memoryview(b"data"))
    store.close()
    assert not store._plan_checksums and not store._extent_plans and not store._leases
