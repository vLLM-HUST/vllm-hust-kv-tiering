"""Real frozen-host CPU manager tests; no model/device initialization."""

from collections import deque

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.tiering.manager")

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.base import JobResult
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

from vllm_hust_kv_tiering.base import TierDeleteResult
from vllm_hust_kv_tiering.lifecycle import LifecycleConfig, SessionLifecycleManager
from vllm_hust_kv_tiering.lifecycle_manager import LifecycleAwareManager
from vllm_hust_kv_tiering.policy import TieringPolicyConfig

pytestmark = pytest.mark.host


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class MemoryRegion:
    def __init__(self, capacity):
        self.data = bytearray(capacity * 4096)

    def create_kv_memoryview(self):
        return memoryview(self.data).cast("B", shape=(len(self.data) // 4096, 4096))

    def cleanup(self):
        pass


class DeferredTier:
    tier_type = "fixture"

    def __init__(self):
        self.pending = {}
        self.results = deque()
        self.keys = set()
        self.deleted = set()
        self.closed = False

    def on_new_request(self, context):
        return RequestOffloadingContext()

    def on_request_finished(self, context):
        pass

    def touch(self, keys, context):
        pass

    def lookup(self, key, context):
        return LookupResult.HIT if key in self.keys else LookupResult.MISS

    def submit_store(self, metadata):
        self.pending[metadata.job_id] = metadata

    def submit_load(self, metadata):
        self.pending[metadata.job_id] = metadata

    def finish(self, success=True):
        for job_id, metadata in list(self.pending.items()):
            if success and not metadata.is_promotion:
                self.keys.update(metadata.keys)
            self.results.append(JobResult(job_id, success))
            del self.pending[job_id]

    def get_finished_jobs(self):
        result = list(self.results)
        self.results.clear()
        return result

    def on_schedule_end(self, context):
        pass

    def has_pending_work(self):
        return bool(self.pending) or bool(self.results)

    def drain_jobs(self):
        self.finish()

    def delete(self, keys):
        self.keys.difference_update(keys)
        self.deleted.update(keys)
        return TierDeleteResult(set(keys), len(keys))

    def get_stats(self):
        return None

    def shutdown(self):
        self.closed = True


@pytest.fixture
def make_manager():
    managers = []

    def make(
        *, capacity=4, policy=None, lifecycle=None, cache_policy="lru", secondary=True
    ):
        clock, tier = Clock(), DeferredTier()
        config = lifecycle or LifecycleConfig(residency_tracking_enabled=True)
        primary = CPUPrimaryTierOffloadingManager(
            capacity, MemoryRegion(capacity), cache_policy=cache_policy
        )
        delegate = TieringOffloadingManager(primary, [tier] if secondary else [])
        owner = LifecycleAwareManager(
            delegate,
            SessionLifecycleManager(config, clock=clock),
            policy or TieringPolicyConfig(),
            clock=clock,
        )
        owner.bind_pressure_provider(lambda: 0.9)
        owner.begin_step()
        managers.append(owner)
        return owner, tier, clock

    yield make
    for manager in managers:
        manager.shutdown()


def request(manager, name="r1", session="s1"):
    context = ReqContext(name, {"session_id": session})
    manager.on_new_request(context)
    return context


def store(manager, context, keys):
    output = manager.prepare_store(keys, context)
    assert output is not None
    if output.keys_to_store:
        manager.complete_store(output.keys_to_store, context)
    return output


def step(manager, clock):
    clock.now += 0.1
    manager.on_schedule_end(ScheduleEndContext((), ()))
    manager.begin_step()


def test_missing_pressure_binding_fails_instead_of_silent_noop():
    primary = CPUPrimaryTierOffloadingManager(2, MemoryRegion(2))
    owner = LifecycleAwareManager(
        TieringOffloadingManager(primary, []),
        SessionLifecycleManager(LifecycleConfig(residency_tracking_enabled=True)),
    )
    try:
        with pytest.raises(RuntimeError, match="bind_gpu_block_pool"):
            owner.begin_step()
    finally:
        owner.shutdown()


def test_low_hbm_pressure_defers_store_admission(make_manager):
    manager, _, _ = make_manager()
    manager.bind_pressure_provider(lambda: 0.1)
    manager.begin_step()
    assert not manager.should_build_store_jobs()
    context = request(manager)
    assert manager.prepare_store([b"a"], context) is None
    assert manager._cpu.free_slots == 4


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_cpu_spill_is_pinned_until_completed_and_then_reclaimed(
    make_manager, cache_policy
):
    config = LifecycleConfig(
        residency_tracking_enabled=True, cpu_high_watermark=0.75, cpu_low_watermark=0.25
    )
    manager, tier, clock = make_manager(lifecycle=config, cache_policy=cache_policy)
    context = request(manager)
    store(manager, context, [b"a", b"b", b"c"])
    assert len(tier.pending) == 1
    assert not manager.residency.has_secondary_copy(b"a")
    step(manager, clock)
    assert manager._cpu.free_slots == 1  # no premature demotion
    assert manager.prepare_store([b"d", b"e"], context) is None
    tier.finish()
    step(manager, clock)
    assert manager._cpu.free_slots >= 3
    assert {b"a", b"b", b"c"}.issubset(tier.keys)
    assert manager.prepare_store([b"d", b"e"], context) is not None


def test_low_cpu_pressure_skips_fs_without_inventing_secondary_copy(make_manager):
    manager, tier, clock = make_manager()
    context = request(manager)
    store(manager, context, [b"a"])
    assert not tier.pending
    step(manager, clock)
    assert not manager.residency.has_secondary_copy(b"a")
    assert manager._cpu.ready(b"a")
    assert manager._cpu.status(b"a").ref_cnt == 0


def test_idle_demotion_spills_dirty_copy_then_frees(make_manager):
    config = LifecycleConfig(residency_tracking_enabled=True, cpu_demote_after_sec=1)
    manager, tier, clock = make_manager(lifecycle=config)
    context = request(manager)
    store(manager, context, [b"a"])
    manager.on_request_finished(context)
    step(manager, clock)  # consumes skipped cascade and finalizes request
    clock.now += 2
    step(manager, clock)
    assert tier.pending
    assert manager._cpu.ready(b"a")
    tier.finish()
    step(manager, clock)
    assert manager._cpu.status(b"a") is None
    assert b"a" in tier.keys


def test_failed_spill_retains_cpu_copy_and_backs_off(make_manager):
    config = LifecycleConfig(residency_tracking_enabled=True, cpu_demote_after_sec=1)
    manager, tier, clock = make_manager(lifecycle=config)
    context = request(manager)
    store(manager, context, [b"a"])
    manager.on_request_finished(context)
    step(manager, clock)
    clock.now += 2
    step(manager, clock)
    tier.finish(success=False)
    step(manager, clock)
    assert manager._cpu.ready(b"a")
    assert not manager.residency.has_secondary_copy(b"a")
    assert not tier.pending
    assert manager.get_policy_snapshot()["decisions"]["secondary_failure"] == 1


def test_request_finish_waits_for_device_and_secondary_jobs(make_manager):
    policy = TieringPolicyConfig(secondary_pressure_aware=False)
    manager, tier, clock = make_manager(policy=policy)
    context = request(manager)
    output = manager.prepare_store([b"a"], context)
    manager.on_request_finished(context)
    assert manager._lifecycle.pending_request_ids() == {"r1"}
    manager.complete_store(output.keys_to_store, context)
    assert manager._lifecycle.pending_request_ids() == {"r1"}
    tier.finish()
    step(manager, clock)
    assert manager._lifecycle.pending_request_ids() == set()


def test_ttl_does_not_expire_unfinished_transfers(make_manager):
    config = LifecycleConfig(
        residency_tracking_enabled=True, idle_ttl_sec=1, delete_expired_secondary=True
    )
    manager, tier, clock = make_manager(
        lifecycle=config, policy=TieringPolicyConfig(secondary_pressure_aware=False)
    )
    context = request(manager)
    store(manager, context, [b"a"])
    manager.on_request_finished(context)
    clock.now += 5
    step(manager, clock)
    assert not tier.deleted
    assert manager._lifecycle.snapshot()["sessions"] == 1
    tier.finish()
    step(manager, clock)
    if manager._delete_future:
        manager._delete_future.result(timeout=5)
    step(manager, clock)
    assert tier.deleted == {b"a"}


def test_ttl_delete_false_preserves_fs_but_releases_cpu(make_manager):
    config = LifecycleConfig(
        residency_tracking_enabled=True, idle_ttl_sec=1, delete_expired_secondary=False
    )
    manager, tier, clock = make_manager(
        lifecycle=config, policy=TieringPolicyConfig(secondary_pressure_aware=False)
    )
    context = request(manager)
    store(manager, context, [b"a"])
    tier.finish()
    step(manager, clock)
    manager.on_request_finished(context)
    clock.now += 2
    step(manager, clock)
    assert not tier.deleted
    assert b"a" in tier.keys
    assert manager._cpu.status(b"a") is None


def test_shared_prefix_survives_one_sessions_expiry(make_manager):
    config = LifecycleConfig(
        residency_tracking_enabled=True, idle_ttl_sec=1, delete_expired_secondary=True
    )
    manager, tier, clock = make_manager(lifecycle=config)
    a, b = request(manager, "r1", "s1"), request(manager, "r2", "s2")
    store(manager, a, [b"shared"])
    step(manager, clock)
    assert manager.lookup(b"shared", b) == LookupResult.HIT
    manager.on_request_finished(a)
    clock.now += 2
    step(manager, clock)
    assert not tier.deleted
    assert manager._cpu.ready(b"shared")


def test_inflight_limit_and_soft_whole_request_budget(make_manager):
    policy = TieringPolicyConfig(max_inflight_store_jobs=1, max_store_blocks_per_step=1)
    manager, _, clock = make_manager(policy=policy)
    context = request(manager)
    output = manager.prepare_store([b"a", b"b"], context)
    assert output is not None and len(output.keys_to_store) == 2
    assert manager.prepare_store([b"c"], context) is None
    manager.complete_store(output.keys_to_store, context)
    assert manager.prepare_store([b"c"], context) is None
    step(manager, clock)
    assert manager.prepare_store([b"c"], context) is not None


def test_secondary_failure_does_not_release_unrelated_load_pin(make_manager):
    manager, tier, clock = make_manager(
        policy=TieringPolicyConfig(secondary_pressure_aware=False)
    )
    context = request(manager)
    store(manager, context, [b"a"])
    manager.prepare_load([b"a"], context)
    tier.finish(False)
    step(manager, clock)
    assert manager._cpu.status(b"a").ref_cnt == 1
    assert manager._cpu.remove([b"a"]) == set()
    manager.complete_load([b"a"], context)
    assert manager._cpu.status(b"a").ref_cnt == 0


def test_lifecycle_prometheus_gauges_and_expiry_deltas(make_manager):
    from vllm_hust_kv_tiering.residency import TieringMetrics

    config = LifecycleConfig(residency_tracking_enabled=True, idle_ttl_sec=1)
    manager, _, clock = make_manager(lifecycle=config)
    context = request(manager)
    first = manager.get_stats().reduce()
    assert first["vllm:kv_offload_load_bytes"] == 0
    assert first["vllm:kv_offload_store_bytes"] == 0
    assert first[TieringMetrics.EXPIRED_SESSIONS] == 0
    assert first[TieringMetrics.SESSION_STATES + ":('active',)"] == 1
    assert first[TieringMetrics.SESSION_STATES + ":('idle',)"] == 0
    store(manager, context, [b"a"])
    manager.on_request_finished(context)
    step(manager, clock)
    idle = manager.get_stats().reduce()
    assert idle[TieringMetrics.SESSION_STATES + ":('idle',)"] == 1
    clock.now += 2
    step(manager, clock)
    expired = manager.get_stats().reduce()
    assert expired[TieringMetrics.SESSION_STATES + ":('idle',)"] == 0
    assert expired[TieringMetrics.EXPIRED_SESSIONS] == 1
    assert manager.get_stats().reduce()[TieringMetrics.EXPIRED_SESSIONS] == 0


def test_cpu_reclaim_metric_reports_completion_once(make_manager):
    from vllm_hust_kv_tiering.residency import TieringMetrics

    config = LifecycleConfig(residency_tracking_enabled=True, cpu_demote_after_sec=1)
    manager, tier, clock = make_manager(lifecycle=config)
    context = request(manager)
    store(manager, context, [b"a"])
    manager.on_request_finished(context)
    step(manager, clock)
    clock.now += 2
    step(manager, clock)
    tier.finish()
    step(manager, clock)
    name = TieringMetrics.PRIMARY_RECLAIMED_BLOCKS + ":('policy',)"
    assert manager.get_stats().reduce()[name] == 1
    assert manager.get_stats().reduce()[name] == 0
