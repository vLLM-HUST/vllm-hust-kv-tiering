"""Matched-host filesystem factory, I/O, integrity and resource contracts."""

from __future__ import annotations

import mmap
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.v1.kv_offload.tiering.base")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

from vllm_hust_kv_tiering.fs import manager as fs_manager
from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec

pytestmark = pytest.mark.host


class HeldPool:
    """Deterministic pool retaining the frozen host's exact public signature."""

    def __init__(self, reads, writes, thread_name_prefix):
        self.reads, self.writes = reads, writes
        self.tasks = deque()
        self.finished = deque()

    def enqueue_store(self, job_id, n_tasks, tasks):
        tasks = list(tasks)
        assert n_tasks == len(tasks) == 1
        self.tasks.append((job_id, tasks[0]))

    enqueue_load = enqueue_store

    def run_one(self):
        job_id, task = self.tasks.popleft()
        try:
            task()
        except (OSError, ValueError):
            self.finished.append((job_id, False))
        else:
            self.finished.append((job_id, True))

    def get_finished(self):
        result = list(self.finished)
        self.finished.clear()
        return result

    def wait_idle(self):
        while self.tasks:
            self.run_one()

    def shutdown(self, wait=True):
        self.wait_idle()


@pytest.fixture
def make_tier(tmp_path, monkeypatch):
    opened = []

    def create(*, backend="torch_async", held=True, **options):
        if held:
            monkeypatch.setattr(fs_manager, "DualQueueThreadPool", HeldPool)
        parallel = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            rank=0,
        )
        group = KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=8, num_kv_heads=1, head_size=1, dtype=torch.int8
            ),
        )
        spec = SimpleNamespace(
            extra_config={"ascend_copy_backend": backend},
            block_size_factor=1,
            vllm_config=SimpleNamespace(
                parallel_config=parallel,
                use_v2_model_runner=False,
                cache_config=SimpleNamespace(cache_dtype="auto", block_size=8),
                model_config=SimpleNamespace(model="fixture"),
            ),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
        )
        raw = mmap.mmap(-1, 4 * 4096)
        raw[:] = bytes(range(256)) * 64
        view = memoryview(raw).cast("B", shape=(4, 4096))
        config = {"type": "hust_fs", "root_dir": str(tmp_path), **options}
        try:
            HustTieringOffloadingSpec._register_storage()
            tier = SecondaryTierFactory.create_secondary_tier(config, view, spec)
        except BaseException:
            view.release()
            raw.close()
            raise
        opened.append((tier, view, raw))
        return tier, raw

    yield create
    for tier, view, raw in opened:
        tier.shutdown()
        view.release()
        raw.close()


def key(index=0):
    return make_offload_key(bytes([index]) * 32, 0)


def job(job_id, keys, *, promotion=False, req_id="fixture"):
    return JobMetadata(
        job_id, keys, np.array(list(range(len(keys)))), promotion, ReqContext(req_id)
    )


def finish(tier):
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


@pytest.mark.parametrize("backend", ["torch_async", "torch_sync"])
def test_actual_factory_accepts_ascend_aliases_and_roundtrips(make_tier, backend):
    tier, raw = make_tier(backend=backend)
    expected = bytes(raw[:4096])
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success
    raw[:4096] = bytes(4096)
    tier.submit_load(job(2, [key()], promotion=True))
    assert finish(tier)[0].success
    assert bytes(raw[:4096]) == expected


def test_default_pool_is_conservative_independent_of_cpu_count(make_tier, monkeypatch):
    monkeypatch.setattr(fs_manager.os, "cpu_count", lambda: 512)
    tier, _ = make_tier()
    assert (tier._pool.reads, tier._pool.writes) == (2, 2)
    assert tier.max_pending_jobs == 64


def test_real_host_pool_uses_four_default_workers(make_tier):
    tier, _ = make_tier(held=False)
    assert len(tier._pool._threads) == 4
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success


@pytest.mark.parametrize("name", ["n_read_threads", "n_write_threads"])
@pytest.mark.parametrize("value", [False, True, 0, -1, 1.5, "2", 65])
def test_invalid_thread_counts_are_rejected(make_tier, name, value):
    with pytest.raises(ValueError, match=name):
        make_tier(**{name: value})


@pytest.mark.parametrize("value", [False, True, 0, -1, 1.5, "2", None])
def test_invalid_pending_job_limits_are_rejected(make_tier, value):
    with pytest.raises(ValueError, match="max_pending_jobs"):
        make_tier(max_pending_jobs=value)


def test_one_explicit_thread_count_does_not_expand_other_group(make_tier):
    tier, _ = make_tier(n_read_threads=1)
    assert (tier._pool.reads, tier._pool.writes) == (1, 2)


def test_saturation_never_enqueues_and_fails_load_as_request_local_miss(make_tier):
    tier, _ = make_tier(max_pending_jobs=1)
    tier.submit_store(job(1, [key()]))
    request = ReqContext("load")
    # Simulate a cached positive existence result without racing worker timing.
    from vllm.v1.kv_offload.tiering.async_lookup import LookupState

    tier._lookup_manager._lookup_state[key(1)] = LookupState(True, {request.req_id})
    tier._lookup_manager._req_keys[request.req_id] = {key(1)}
    assert tier.lookup(key(1), request) is LookupResult.HIT
    tier.submit_load(job(2, [key(1)], promotion=True, req_id=request.req_id))
    assert len(tier._pool.tasks) == 1
    results = list(tier.get_finished_jobs())
    assert [(result.job_id, result.success) for result in results] == [(2, False)]
    assert tier.lookup(key(1), request) is LookupResult.MISS
    assert tier.get_storage_counters()["queue_rejections"] == 1
    assert tier._admitted_jobs == {1}
    assert finish(tier)[0].success
    tier.submit_store(job(3, [key(1)]))
    assert finish(tier)[0].success
    assert key(1) not in tier._failed_load_keys.get(request.req_id, ())


def test_unobserved_completed_jobs_still_consume_pending_budget(make_tier):
    tier, _ = make_tier(max_pending_jobs=1)
    tier.submit_store(job(1, [key()]))
    tier._pool.run_one()
    tier.submit_store(job(2, [key(1)]))
    assert not tier._pool.tasks
    results = list(tier.get_finished_jobs())
    assert {(result.job_id, result.success) for result in results} == {
        (1, True),
        (2, False),
    }
    assert not tier._admitted_jobs
    tier.submit_store(job(3, [key(1)]))
    assert finish(tier)[0].success


def test_worker_load_failure_invalidates_stale_positive_lookup(make_tier):
    tier, _ = make_tier()
    request = ReqContext("failed-load")
    from vllm.v1.kv_offload.tiering.async_lookup import LookupState

    tier._lookup_manager._lookup_state[key()] = LookupState(True, {request.req_id})
    tier._lookup_manager._req_keys[request.req_id] = {key()}
    assert tier.lookup(key(), request) is LookupResult.HIT
    tier.submit_load(job(1, [key()], promotion=True, req_id=request.req_id))
    assert not finish(tier)[0].success
    assert tier.lookup(key(), request) is LookupResult.MISS
    tier.on_request_finished(request)
    assert request.req_id not in tier._failed_load_keys


def test_late_failed_load_does_not_restore_finished_request_state(make_tier):
    tier, _ = make_tier()
    tier.submit_load(job(1, [key()], promotion=True))
    tier.on_request_finished(ReqContext("fixture"))
    assert not finish(tier)[0].success
    assert "fixture" not in tier._failed_load_keys


def test_quota_propagates_and_fails_async_without_losing_live_key(make_tier):
    tier, _ = make_tier(max_storage_bytes=4096, durability="none")
    assert tier.segment_store.max_storage_bytes == 4096
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success
    tier.submit_store(job(2, [key(1)]))
    assert not finish(tier)[0].success
    assert tier.segment_store.contains_many([key(), key(1)]) == [True, False]
    assert tier.get_storage_counters()["capacity_rejections"] >= 1
    assert tier.get_storage_counters()["store_errors"] == 1


def test_metadata_work_runs_only_inside_worker_task(make_tier, monkeypatch):
    tier, _ = make_tier()

    def fail_if_called(*args, **kwargs):
        pytest.fail("scheduler performed segment metadata work")

    monkeypatch.setattr(tier.segment_store, "prepare_store", fail_if_called)
    monkeypatch.setattr(tier.segment_store, "get_locations", fail_if_called)
    tier.submit_store(job(1, [key()]))
    tier.submit_load(job(2, [key()], promotion=True))
    assert len(tier._pool.tasks) == 2
    # Drop deliberately unexecuted tasks before normal fixture shutdown.
    tier._pool.tasks.clear()


def test_legacy_dedup_callbacks_are_not_claimed_as_physical_write_bytes(make_tier):
    tier, _ = make_tier(storage_layout="file")
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success
    tier.submit_store(job(2, [key()]))
    assert finish(tier)[0].success
    tier.submit_load(job(3, [key()], promotion=True))
    assert finish(tier)[0].success
    counters = tier.get_storage_counters()
    assert counters["store_bytes"] == 0
    assert counters["legacy_store_blocks"] == 2
    assert counters["load_bytes"] == 4096
    assert counters["load_hits"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"integrity": True},
        {"integrity": []},
        {"integrity": "sha256"},
        {"storage_layout": "file", "integrity": "crc32"},
    ],
)
def test_invalid_integrity_configuration_is_rejected(make_tier, options):
    with pytest.raises(ValueError, match="integrity"):
        make_tier(**options)


def test_default_segment_integrity_detects_same_size_corruption_and_reports_miss(
    make_tier,
):
    tier, raw = make_tier()
    assert tier.integrity == tier.segment_store.integrity == "crc32"
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success
    location = tier.segment_store.get_locations([key()])[0]
    path = tier.segment_store._segment_path(location.segment_id)
    corrupted = bytearray(path.read_bytes())
    corrupted[location.offset] ^= 1
    path.write_bytes(corrupted)
    request = ReqContext("corruption")
    from vllm.v1.kv_offload.tiering.async_lookup import LookupState

    tier._lookup_manager._lookup_state[key()] = LookupState(True, {request.req_id})
    tier._lookup_manager._req_keys[request.req_id] = {key()}
    assert tier.lookup(key(), request) is LookupResult.HIT
    raw[:4096] = bytes(4096)
    tier.submit_load(job(2, [key()], promotion=True, req_id=request.req_id))
    assert not finish(tier)[0].success
    assert tier.lookup(key(), request) is LookupResult.MISS
    assert tier.segment_store.contains_many([key()]) == [False]
    assert tier.get_storage_counters()["checksum_errors"] == 1
    stats = tier.get_stats().reduce()
    assert stats["vllm:kv_tiering_fs_checksum_errors:('hust_fs',)"] == 1
    assert (
        not tier.segment_store._plan_checksums and not tier.segment_store._extent_plans
    )


@pytest.mark.parametrize("integrity, expected_hit", [(None, False), ("none", True)])
def test_crc32_namespace_never_falls_back_to_unverified_legacy_files(
    make_tier, integrity, expected_hit
):
    tier, _ = make_tier(integrity=integrity)
    path = Path(tier.file_mapper.get_file_name(key()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(4096))
    request = ReqContext("legacy")
    assert list(tier._lookup_manager.batch_lookup([key()], request)) == [expected_hit]
    tier.submit_load(job(1, [key()], promotion=True, req_id=request.req_id))
    assert finish(tier)[0].success is expected_hit


def test_high_level_crc32_refuses_existing_unverified_namespace(make_tier):
    tier, _ = make_tier(integrity="none")
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success
    with pytest.raises(ValueError, match="fresh namespace"):
        make_tier()
    assert tier.segment_store.contains_many([key()]) == [True]


@pytest.mark.parametrize("name", ["segment_io_batch_blocks", "segment_shards"])
@pytest.mark.parametrize("value", [True, False, -1, 1.5, "2"])
def test_segment_tuning_requires_nonnegative_integers(make_tier, name, value):
    with pytest.raises(ValueError, match=name):
        make_tier(**{name: value})


@pytest.mark.parametrize("root", [None, "", "relative/path", 42, b"/tmp/not-a-string"])
def test_root_directory_requires_nonempty_absolute_path(make_tier, root):
    with pytest.raises(ValueError, match="root_dir"):
        make_tier(root_dir=root)


def test_absolute_pathlike_root_is_accepted(make_tier, tmp_path):
    tier, _ = make_tier(root_dir=tmp_path / "pathlike")
    assert tier.segment_store is not None


def test_fs_metrics_report_actual_io_once_without_resetting_snapshots(make_tier):
    tier, _ = make_tier()
    initial = tier.get_stats().reduce()
    assert initial["vllm:kv_tiering_fs_load_bytes:('hust_fs',)"] == 0
    assert initial["vllm:kv_tiering_fs_checksum_errors:('hust_fs',)"] == 0
    tier.submit_store(job(1, [key()]))
    assert finish(tier)[0].success
    tier.submit_load(job(2, [key()], promotion=True))
    assert finish(tier)[0].success
    definitions = tier.build_metric_definitions({})
    prefix = "vllm:kv_tiering_fs_"
    assert prefix + "load_bytes" in definitions
    assert definitions[prefix + "load_bytes"].labelnames == ("tier",)
    assert all(name.startswith(prefix) for name in definitions)
    first = tier.get_stats().reduce()
    assert first[prefix + "load_bytes:('hust_fs',)"] == 4096
    assert first[prefix + "store_bytes:('hust_fs',)"] == 4096
    second = tier.get_stats().reduce()
    assert second[prefix + "load_bytes:('hust_fs',)"] == 0
    assert tier.get_storage_counters()["load_bytes"] == 4096
    tier.get_storage_counters(reset=True)
    tier.submit_load(job(3, [key()], promotion=True))
    assert finish(tier)[0].success
    assert tier.get_storage_counters()["load_bytes"] == 4096
    assert tier.get_stats().reduce()[prefix + "load_bytes:('hust_fs',)"] == 4096


def test_actual_host_factory_segment_store_restore(make_tier):
    tier, raw = make_tier(
        backend="torch_sync",
        held=False,
        n_read_threads=1,
        n_write_threads=1,
        storage_layout="segment",
    )
    key = make_offload_key(bytes(range(32)), 0)
    request = SimpleNamespace(req_id="fixture")
    tier.submit_store(JobMetadata(1, [key], np.array([1]), False, request))
    result = finish(tier)
    assert len(result) == 1 and result[0].success
    expected = bytes(raw[4096:8192])
    raw[12288:16384] = bytes(4096)
    tier.submit_load(JobMetadata(2, [key], np.array([3]), True, request))
    result = finish(tier)
    assert len(result) == 1 and result[0].success
    assert bytes(raw[12288:16384]) == expected
