"""Matched-host plugin registration, scheduler and HBM pressure contracts."""

import uuid
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.tiering.spec")
pytest.importorskip("vllm.v1.core.block_pool")
pytest.importorskip("vllm.v1.kv_offload.tiering.base")

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_offload import base as host_base
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key

from vllm_hust_kv_tiering import register
from vllm_hust_kv_tiering.ascend_connector import HustTieringConnector
from vllm_hust_kv_tiering.fs.manager import SegmentFileSystemTier
from vllm_hust_kv_tiering.host_adapter import HBMPressureSource
from vllm_hust_kv_tiering.policy import PressureGate
from vllm_hust_kv_tiering.scheduler import TieringConnectorScheduler
from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec

pytestmark = pytest.mark.host


class Pool:
    num_gpu_blocks = 11

    def __init__(self, allocated=0.1, cached=0):
        self.allocated = allocated
        self.cached_block_hash_to_block = dict.fromkeys(range(1, cached + 1))
        self.cached_block_hashes_by_block = {}  # supplemental aliases only

    def get_usage(self):
        return self.allocated


def fixture_configs(tmp_path, extra=None):
    options = {
        "spec_name": "HustTieringOffloadingSpec",
        "spec_module_path": "vllm_hust_kv_tiering.spec",
        "cpu_bytes_to_use": 32768,
        "ascend_copy_backend": "torch_async",
        "tiering_secondary_pressure_aware": False,
        "secondary_tiers": [
            {
                "type": "hust_fs",
                "root_dir": str(tmp_path),
                "n_read_threads": 1,
                "n_write_threads": 1,
                "max_storage_bytes": 1024 * 1024,
            }
        ],
    }
    options.update(extra or {})
    parallel = SimpleNamespace(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        rank=0,
    )
    vconfig = SimpleNamespace(
        instance_id=uuid.uuid4().hex,
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=options),
        kv_events_config=None,
        parallel_config=parallel,
        speculative_config=None,
        cache_config=SimpleNamespace(
            cache_dtype="auto", block_size=8, enable_prefix_caching=True
        ),
        model_config=SimpleNamespace(model="fixture"),
        use_v2_model_runner=False,
    )
    group = KVCacheGroupSpec(
        ["attention"],
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.int8),
    )
    caches = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[group],
        kv_cache_tensors=[
            SimpleNamespace(size=16384, block_stride=0, shared_by=["attention"])
        ],
    )
    return vconfig, caches


@pytest.fixture
def connector(tmp_path, monkeypatch):
    # Freeze the test geometry's hash alignment, not any transfer/scheduler code.
    monkeypatch.setattr(host_base, "resolve_kv_cache_block_sizes", lambda *args: (8, 8))
    vconfig, caches = fixture_configs(tmp_path)
    result = HustTieringConnector(vconfig, KVConnectorRole.SCHEDULER, caches)
    result.bind_gpu_block_pool(Pool(0.9))
    yield result
    result.shutdown()


def tick(connector):
    return connector.build_connector_meta(SchedulerOutput.make_empty())


def test_actual_factory_constructs_policy_scheduler_and_reuses_manager(connector):
    scheduler = connector.connector_scheduler
    assert isinstance(scheduler, TieringConnectorScheduler)
    assert scheduler.manager._cpu.capacity == 4
    assert scheduler.manager._delegate is not scheduler.manager
    assert connector.connector_worker is None
    assert not tick(connector).store_jobs
    assert scheduler.manager._step_initialized


def test_pressure_includes_cached_keys_without_adding_allocation_usage():
    pool = Pool(allocated=0.1, cached=9)
    source = HBMPressureSource(pool)
    assert source() == 0.9
    pool.allocated = 0.8
    assert source() == 0.9  # Max, not an additive score of 1.7.
    pool.cached_block_hash_to_block.clear()
    assert source() == 0.8


def test_prefix_cache_pressure_opens_gate_at_low_active_concurrency(connector):
    connector.bind_gpu_block_pool(Pool(allocated=0.1, cached=9))
    tick(connector)
    assert connector.connector_scheduler.manager.should_build_store_jobs()
    connector.bind_gpu_block_pool(Pool(allocated=0.1, cached=1))
    tick(connector)
    assert not connector.connector_scheduler.manager.should_build_store_jobs()


def test_low_pressure_hook_defers_without_touching_host_progress(
    monkeypatch, connector
):
    scheduler = connector.connector_scheduler
    connector.bind_gpu_block_pool(Pool(0.1))
    scheduler.manager.begin_step()
    sentinel = SimpleNamespace(next_stored_block_idx=7)
    scheduler._req_status["sentinel"] = sentinel
    try:
        assert scheduler._build_store_jobs(SchedulerOutput.make_empty()) == {}
        assert sentinel.next_stored_block_idx == 7
    finally:
        del scheduler._req_status["sentinel"]


def test_lifetime_rejection_avoids_host_candidate_rebuild(monkeypatch, connector):
    scheduler = connector.connector_scheduler
    tick(connector)
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"rejected": 4096}
    scheduler.manager._request_store_skips["rejected"] = "working_set_capacity"

    def unexpected(*args):
        pytest.fail("permanently rejected request rebuilt host store candidates")

    monkeypatch.setattr(OffloadingConnectorScheduler, "_build_store_jobs", unexpected)
    assert scheduler._build_store_jobs(output) == {}
    assert output.num_scheduled_tokens == {"rejected": 4096}


def test_mixed_store_view_preserves_original_scheduler_output(monkeypatch, connector):
    scheduler = connector.connector_scheduler
    tick(connector)
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"rejected": 4096, "eligible": 256}
    output.preempted_req_ids = {"rejected"}
    scheduler.manager._request_store_skips["rejected"] = "working_set_capacity"
    sentinel = SimpleNamespace(next_stored_block_idx=7, transfer_jobs={123})
    scheduler._req_status["rejected"] = sentinel

    def probe(instance, store_output):
        assert instance is scheduler
        assert store_output is not output
        assert store_output.num_scheduled_tokens == {"eligible": 256}
        assert store_output.preempted_req_ids is output.preempted_req_ids
        return {456: "host-owned-job"}

    monkeypatch.setattr(OffloadingConnectorScheduler, "_build_store_jobs", probe)
    try:
        assert scheduler._build_store_jobs(output) == {456: "host-owned-job"}
        assert output.num_scheduled_tokens == {"rejected": 4096, "eligible": 256}
        assert sentinel.next_stored_block_idx == 7
        assert sentinel.transfer_jobs == {123}
    finally:
        del scheduler._req_status["rejected"]


def test_transient_rejections_do_not_filter_future_store_attempts(
    monkeypatch, connector
):
    scheduler = connector.connector_scheduler
    tick(connector)
    scheduler.manager._decision("step_budget")
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"later": 4096}

    def probe(instance, store_output):
        assert instance is scheduler
        assert store_output is output
        return {789: "host-retry"}

    monkeypatch.setattr(OffloadingConnectorScheduler, "_build_store_jobs", probe)
    assert scheduler._build_store_jobs(output) == {789: "host-retry"}


def test_complete_spec_manager_fs_promotion_roundtrip(connector):
    manager = connector.connector_scheduler.manager
    tick(connector)
    context = ReqContext("roundtrip", {"session_id": "conversation"})
    manager.on_new_request(context)
    key = make_offload_key(bytes(range(32)), 0)
    output = manager.prepare_store([key], context)
    assert output is not None and output.keys_to_store == [key]
    bid = int(output.store_spec.block_ids[0])
    view = manager._cpu.primary.get_kv_memoryview()
    flat = view.cast("B")
    stride = view.strides[0]
    expected = bytes(range(256)) * (stride // 256)
    try:
        flat[bid * stride : (bid + 1) * stride] = expected
        manager.complete_store([key], context)
        tier = manager.secondary_tiers[0]
        tier.drain_jobs()
        tick(connector)
        assert manager.residency.has_secondary_copy(key)
        assert manager._cpu.remove([key]) == {key}
        manager._mark_cpu([key], False)
        assert manager.lookup(key, context) == LookupResult.RETRY
        tick(connector)  # posts the actual asynchronous existence lookup
        # Synchronize the test at the worker-to-scheduler queue boundary,
        # without sleeps/busy polling or substituting a fake existence result.
        results = tier._lookup_manager._pending_results.get(timeout=5)
        tier._lookup_manager._pending_results.put(results)
        assert manager.lookup(key, context) == LookupResult.RETRY  # reserves CPU
        tick(connector)  # submits the deferred FS-to-CPU promotion
        tier.drain_jobs()
        tick(connector)  # observes successful completion
        assert manager.lookup(key, context) == LookupResult.HIT
        loaded = manager.prepare_load([key], context)
        loaded_bid = int(loaded.block_ids[0])
        assert bytes(flat[loaded_bid * stride : (loaded_bid + 1) * stride]) == expected
        manager.complete_load([key], context)
        assert tier.get_storage_counters()["load_bytes"] == stride
        manager.on_request_finished(context)
    finally:
        flat.release()


@pytest.mark.parametrize(
    "options",
    [
        {"cpu_bytes_to_use": True},
        {"cpu_bytes_to_use": -1},
        {"cpu_bytes_to_use": 1},
        {"cpu_bytes_to_use": 2.5},
        {"block_size": 8.5},
        {"offload_prompt_only": "true"},
        {"eviction_policy": "typo"},
        {"tiering_hbm_pressure_aware": "false"},
        {"tiering_typo": 1},
        {"lifecycle_idle_ttl_sec": -1},
        {"lifecycle_typo": 1},
        {"pressure_aware": True},
        {"store_threshold": 2},
        {"store_threshold": "1"},
        {"store_threshold": True},
        {"max_tracker_size": 128},
    ],
)
def test_invalid_profile_fails_before_allocating_mmap(tmp_path, monkeypatch, options):
    monkeypatch.setattr(host_base, "resolve_kv_cache_block_sizes", lambda *args: (8, 8))
    config, caches = fixture_configs(tmp_path, options)
    with pytest.raises(ValueError):
        HustTieringOffloadingSpec(config, caches)


class NoIterationIndex:
    def __init__(self, count):
        self.count = count
        self.len_calls = 0

    def __len__(self):
        self.len_calls += 1
        return self.count

    def __iter__(self):
        raise AssertionError("pressure collection must not iterate the prefix index")

    def items(self):
        raise AssertionError("pressure collection must not scan index items")

    def values(self):
        raise AssertionError("pressure collection must not scan physical blocks")


def fake_pool(count=9, usage=0.0):
    return SimpleNamespace(
        num_gpu_blocks=11,
        get_usage=lambda: usage,
        cached_block_hash_to_block=NoIterationIndex(count),
        cached_block_hashes_by_block={},
    )


def key(value):
    return make_block_hash_with_group_id(BlockHash(value.to_bytes(32, "big")), 0)


def test_real_pool_primary_hashes_open_pressure_gate_with_no_aliases():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    source = HBMPressureSource(pool)
    blocks = pool.get_new_blocks(9)
    for index, block in enumerate(blocks):
        pool._insert_block_hash(key(index), block, num_tokens=1)
    pool.free_blocks(blocks)
    assert pool.get_usage() == 0.0
    assert pool.cached_block_hashes_by_block == {}  # supplemental aliases ONLY
    assert len(pool.cached_block_hash_to_block) == 9
    assert source() == pytest.approx(0.9)
    assert PressureGate(0.8, 0.6).update(source())


def test_real_pool_reset_replaces_index_and_pressure_follows_new_object():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    source = HBMPressureSource(pool)
    blocks = pool.get_new_blocks(9)
    for index, block in enumerate(blocks):
        pool._insert_block_hash(key(index), block, num_tokens=1)
    pool.free_blocks(blocks)
    assert source() == pytest.approx(0.9)
    previous_index = pool.cached_block_hash_to_block
    assert pool.reset_prefix_cache()
    assert pool.cached_block_hash_to_block is not previous_index
    assert source() == 0.0


def test_primary_count_is_o1_without_iteration_and_refetched_after_replacement():
    pool = fake_pool()
    source = HBMPressureSource(pool)
    assert source() == 0.9
    previous = pool.cached_block_hash_to_block
    assert previous.len_calls == 1
    pool.cached_block_hash_to_block = NoIterationIndex(1)
    assert source() == 0.1
    assert previous.len_calls == 1
    assert pool.cached_block_hash_to_block.len_calls == 1


@pytest.mark.parametrize("bad_index", [None, object(), 7])
def test_primary_index_abi_required_even_when_alias_dictionary_exists(bad_index):
    pool = fake_pool()
    pool.cached_block_hash_to_block = bad_index
    with pytest.raises(TypeError, match="primary prefix-cache index"):
        HBMPressureSource(pool)


@pytest.mark.parametrize(
    "count,usage,expected",
    [(0, -0.2, 0.0), (0, 1.2, 1.0), (35, 0.0, 1.0), (9, 0.8, 0.9)],
)
def test_pressure_clamps_and_takes_maximum_not_sum(count, usage, expected):
    assert HBMPressureSource(fake_pool(count, usage))() == expected


def test_cached_key_count_can_undercount_duplicate_physical_blocks():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        pool._insert_block_hash(key(1), block, num_tokens=1)
    pool.free_blocks(blocks)
    assert len(pool.cached_block_hash_to_block) == 1
    assert HBMPressureSource(pool)() == 0.1  # not an exact physical count of2


def test_cached_key_count_can_exceed_physical_count_with_aliases():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    block = pool.get_new_blocks(1)[0]
    pool._insert_block_hash(key(1), block, num_tokens=1)
    pool._insert_block_hash(key(2), block, num_tokens=1)
    pool.free_blocks([block])
    assert len(pool.cached_block_hash_to_block) == 2
    assert HBMPressureSource(pool)() == 0.2  # one physical block, two cache keys


def test_v1_plugin_contract() -> None:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

    register()
    assert issubclass(HustTieringOffloadingSpec, OffloadingSpec)
    assert issubclass(SegmentFileSystemTier, SecondaryTierManager)
    assert "hust_fs" in SecondaryTierFactory._registry
