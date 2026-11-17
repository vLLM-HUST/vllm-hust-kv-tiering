"""Matched-host CPU manager: pressure, lifecycle, ownership and failure recovery.

Uses the real CPUPrimaryTierOffloadingManager and TieringOffloadingManager.
Only secondary I/O and delete completion are deterministic test doubles."""

import random
from collections import deque
from concurrent.futures import Future

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


class ManualDeleteExecutor:
    """Expose asynchronous Future completion without thread timing or sleeps."""

    def __init__(self):
        self.pending = deque()
        self.closed = False

    def submit(self, function, *args, **kwargs):
        assert not self.closed
        future = Future()
        self.pending.append((future, function, args, kwargs))
        return future

    def run_next(self):
        future, function, args, kwargs = self.pending.popleft()
        assert future.set_running_or_notify_cancel()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - match Executor Future error delivery
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True):
        assert wait
        while self.pending:
            self.run_next()
        self.closed = True


def ttl_manager(make_manager, *, max_entries=64_000):
    config = LifecycleConfig(
        residency_tracking_enabled=True,
        idle_ttl_sec=1,
        delete_expired_secondary=True,
        residency_max_entries=max_entries,
    )
    manager, tier, clock = make_manager(
        lifecycle=config,
        policy=TieringPolicyConfig(secondary_pressure_aware=False),
    )
    executor = ManualDeleteExecutor()
    manager._delete_executor = executor
    return manager, tier, clock, executor


def expire_stored(manager, tier, clock, keys, *, context=None):
    context = context or request(manager)
    store(manager, context, keys)
    tier.finish()
    step(manager, clock)
    manager.on_request_finished(context)
    clock.now += 2
    step(manager, clock)
    assert manager._delete_future is not None
    assert not manager._delete_future.done()
    return context


def test_deferred_promotion_abort_retains_ownership_until_actual_completion(
    make_manager, monkeypatch
):
    manager, tier, clock = make_manager()
    tier.keys.add(b"promoted")
    finalized = []
    monkeypatch.setattr(
        tier, "on_request_finished", lambda context: finalized.append(context.req_id)
    )
    context = request(manager, "aborted", None)

    assert manager.lookup(b"promoted", context) is LookupResult.RETRY
    assert not tier.pending  # Frozen host defers submit_load until schedule end.
    assert manager._cpu.status(b"promoted").ref_cnt == -1
    manager.on_request_finished(context)
    assert manager._lifecycle.pending_request_ids() == {"aborted"}
    assert manager.residency.get_session_keys("aborted") == {b"promoted"}
    assert finalized == []

    step(manager, clock)
    assert len(tier.pending) == 1
    assert next(iter(tier.pending.values())).is_promotion
    assert manager._lifecycle.pending_request_ids() == {"aborted"}
    assert finalized == []
    assert b"promoted" in manager.residency.get_inflight_keys()

    tier.finish()
    step(manager, clock)
    assert manager._cpu.ready(b"promoted")
    assert manager._lifecycle.pending_request_ids() == set()
    assert manager.residency.get_session_keys("aborted") == set()
    assert manager.residency.get_inflight_keys() == set()
    assert finalized == ["aborted"]
    step(manager, clock)
    assert finalized == ["aborted"]


def test_reset_cancels_deferred_promotion_and_clears_pending_counts(make_manager):
    manager, tier, clock = make_manager()
    tier.keys.add(b"promoted")
    context = request(manager, "aborted", None)
    assert manager.lookup(b"promoted", context) is LookupResult.RETRY
    manager.on_request_finished(context)
    assert manager._lifecycle.pending_request_ids() == {"aborted"}
    assert not tier.pending

    manager.reset_cache()
    assert not tier.pending
    assert manager._cpu.status(b"promoted") is None
    assert manager.residency.get_inflight_keys() == set()
    assert manager._lifecycle.pending_request_ids() == set()
    assert not any(manager._pending_requests.values())
    assert manager.get_policy_snapshot()["pending_device_jobs"] == 0
    step(manager, clock)
    assert not tier.pending
    assert manager.residency.get_inflight_keys() == set()


def test_completed_async_ttl_delete_failure_does_not_block_reset(
    make_manager, monkeypatch
):
    manager, tier, clock, executor = ttl_manager(make_manager)

    def fail_delete(keys):
        raise OSError("injected delete failure")

    monkeypatch.setattr(tier, "delete", fail_delete)
    expire_stored(manager, tier, clock, [b"expired"])
    future = executor.run_next()
    with pytest.raises(OSError, match="injected delete failure"):
        future.result(timeout=1)

    # A completed maintenance failure must be handled as retryable cache work,
    # not escape result() and prevent the host's primary-cache reset.
    manager.reset_cache()
    assert manager._cpu.free_slots == manager._cpu.capacity
    assert manager._cpu.status(b"expired") is None
    assert manager.residency.get_inflight_keys() == set()
    assert manager.get_policy_snapshot()["decisions"]["delete_failure"] >= 1
    assert b"expired" in tier.keys


def test_partial_ttl_delete_retries_only_unacknowledged_subset(
    make_manager, monkeypatch
):
    manager, tier, clock, executor = ttl_manager(make_manager)
    calls = []

    def partial_delete(keys):
        batch = set(keys)
        calls.append(batch)
        removed = {min(batch)} if len(calls) == 1 else batch
        tier.keys.difference_update(removed)
        tier.deleted.update(removed)
        return TierDeleteResult(removed, len(removed))

    monkeypatch.setattr(tier, "delete", partial_delete)
    expire_stored(manager, tier, clock, [b"a", b"b"])
    executor.run_next().result(timeout=1)
    step(manager, clock)
    assert tier.deleted == {b"a"}
    assert manager.residency.has_secondary_copy(b"b")
    assert manager.get_policy_snapshot()["pending_delete_keys"] == 1

    clock.now += manager.policy.retry_backoff_sec + 0.1
    step(manager, clock)
    assert executor.pending
    executor.run_next().result(timeout=1)
    step(manager, clock)
    assert calls == [{b"a", b"b"}, {b"b"}]
    assert tier.deleted == {b"a", b"b"}
    assert manager.get_policy_snapshot()["pending_delete_keys"] == 0
    assert not manager.residency.has_secondary_copy(b"a")
    assert not manager.residency.has_secondary_copy(b"b")


def test_failed_delete_tombstone_backlog_bounds_new_cache_metadata(
    make_manager, monkeypatch
):
    manager, tier, clock, executor = ttl_manager(make_manager, max_entries=2)

    def fail_delete(keys):
        raise OSError("persistent delete failure")

    monkeypatch.setattr(tier, "delete", fail_delete)
    expire_stored(manager, tier, clock, [b"a", b"b"])
    executor.run_next()
    step(manager, clock)
    tombstones = manager._delete_queue | manager._delete_active
    assert tombstones == {b"a", b"b"}
    assert len(manager.residency.tracked_keys() | tombstones) <= 2

    for index in range(4):
        context = request(manager, f"new-{index}", f"new-session-{index}")
        key = f"new-key-{index}".encode()
        assert manager.lookup(key, context) is LookupResult.MISS
        assert manager.prepare_store([key], context) is None
        manager.on_request_finished(context)
        tombstones = manager._delete_queue | manager._delete_active
        assert len(manager.residency.tracked_keys() | tombstones) <= 2
    assert manager.get_policy_snapshot()["decisions"]["metadata_capacity"] >= 1


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_fs_promotion_reconciles_reclaimed_cpu_residency(make_manager, cache_policy):
    manager, tier, clock = make_manager(
        capacity=1,
        cache_policy=cache_policy,
        policy=TieringPolicyConfig(secondary_pressure_aware=False),
    )
    original = request(manager, "original", "original-session")
    store(manager, original, [b"old"])
    tier.finish()
    step(manager, clock)
    assert manager._cpu.ready(b"old")
    assert manager.residency.snapshot()["cpu_blocks"] == 1
    assert b"old" in tier.keys  # A safe persisted reclaim victim exists.
    tier.keys.add(b"promoted")
    context = request(manager, "restore", "restore-session")

    assert manager.lookup(b"promoted", context) is LookupResult.RETRY
    assert manager._cpu.status(b"old") is None
    assert not manager.residency._blocks[b"old"].cpu_resident
    assert manager.residency.snapshot()["cpu_blocks"] == 0
    step(manager, clock)
    tier.finish()
    step(manager, clock)
    assert manager._cpu.ready(b"promoted")
    assert manager.residency._blocks[b"promoted"].cpu_resident
    assert not manager.residency._blocks[b"old"].cpu_resident
    assert manager.residency.snapshot()["cpu_blocks"] == 1


def test_touch_cannot_revive_or_forward_pending_delete_keys(make_manager, monkeypatch):
    manager, tier, clock, executor = ttl_manager(make_manager)
    original = request(manager, "original", "expired-session")
    later = request(manager, "later", "later-session")
    expire_stored(manager, tier, clock, [b"expired"], context=original)
    forwarded = []
    monkeypatch.setattr(tier, "touch", lambda keys, context: forwarded.extend(keys))
    assert b"expired" in manager._delete_active

    manager.touch([b"expired"], later)
    assert manager.residency.get_session_keys("later-session") == set()
    assert manager.lookup(b"expired", later) is LookupResult.MISS
    assert forwarded == []
    assert (
        b"expired" not in manager.residency._blocks
        or not manager.residency._blocks[b"expired"].session_ids
    )
    executor.run_next().result(timeout=1)
    step(manager, clock)


def test_shutdown_retries_failed_drain_before_releasing_primary(
    make_manager, monkeypatch
):
    manager, tier, _ = make_manager(
        policy=TieringPolicyConfig(secondary_pressure_aware=False)
    )
    context = request(manager)
    store(manager, context, [b"pinned"])
    original_drain = tier.drain_jobs
    calls = []

    def fail_once():
        calls.append("drain")
        if len(calls) == 1:
            raise RuntimeError("injected drain failure")
        original_drain()

    monkeypatch.setattr(tier, "drain_jobs", fail_once)
    with pytest.raises(RuntimeError, match="injected drain failure"):
        manager.shutdown()
    assert not tier.closed
    assert tier.pending
    assert manager._cpu.status(b"pinned").ref_cnt > 0
    assert manager._cpu.primary.get_kv_memoryview().nbytes > 0

    manager.shutdown()
    assert tier.closed
    assert not tier.pending
    manager.shutdown()
    assert calls == ["drain", "drain"]


@pytest.mark.parametrize("delete_secondary", [False, True])
def test_anonymous_request_keeps_ttl_after_async_finalization(
    make_manager, delete_secondary
):
    manager, tier, clock = make_manager(
        lifecycle=LifecycleConfig(
            residency_tracking_enabled=True,
            idle_ttl_sec=1,
            delete_expired_secondary=delete_secondary,
        ),
        policy=TieringPolicyConfig(secondary_pressure_aware=False),
    )
    executor = ManualDeleteExecutor()
    manager._delete_executor = executor
    context = request(manager, "anonymous", None)
    store(manager, context, [b"prefix"])
    manager.on_request_finished(context)
    clock.now += 2
    step(manager, clock)
    # An overdue TTL still cannot delete a copy with a pending secondary write.
    assert manager._lifecycle.pending_request_ids() == {"anonymous"}
    assert manager._cpu.ready(b"prefix")
    assert not executor.pending

    tier.finish()
    step(manager, clock)
    assert manager._lifecycle.pending_request_ids() == set()
    assert manager._lifecycle.snapshot()["sessions"] == 0
    assert manager._cpu.status(b"prefix") is None
    if delete_secondary:
        assert len(executor.pending) == 1
        executor.run_next().result(timeout=1)
        step(manager, clock)
        assert tier.deleted == {b"prefix"}
        assert b"prefix" not in tier.keys
    else:
        assert not executor.pending
        assert not tier.deleted
        assert b"prefix" in tier.keys


def test_repeated_anonymous_prefix_passes_calibrated_cost_gate(make_manager):
    # Synthetic calibration makes a second observation economical; it is not a
    # device-performance claim. The first rejected store must still teach history.
    manager, _, _ = make_manager(
        policy=TieringPolicyConfig(
            prefill_tokens_per_second=1,
            restore_bytes_per_second=100,
            store_bytes_per_second=100,
        )
    )
    first = request(manager, "first", None)
    assert manager.lookup(b"prefix", first) is LookupResult.MISS
    for _ in range(3):
        assert manager.prepare_store([b"prefix"], first) is None
    assert manager.get_policy_snapshot()["decisions"]["estimated_cost"] == 3
    manager.on_request_finished(first)
    assert manager._lifecycle.pending_request_ids() == set()

    second = request(manager, "second", None)
    # A mixed batch must not lend the hot prefix's observations to a cold key.
    assert manager.prepare_store([b"prefix", b"cold"], second) is None
    output = store(manager, second, [b"prefix"])
    assert list(output.keys_to_store) == [b"prefix"]
    assert manager._cpu.ready(b"prefix")


def test_shared_anonymous_reader_prevents_writer_ttl_deletion(make_manager):
    manager, tier, clock = make_manager(
        lifecycle=LifecycleConfig(
            residency_tracking_enabled=True,
            idle_ttl_sec=1,
            delete_expired_secondary=True,
        ),
        policy=TieringPolicyConfig(secondary_pressure_aware=False),
    )
    executor = ManualDeleteExecutor()
    manager._delete_executor = executor
    writer = request(manager, "writer", None)
    store(manager, writer, [b"shared"])
    tier.finish()
    step(manager, clock)
    manager.on_request_finished(writer)
    reader = request(manager, "reader", None)
    assert manager.lookup(b"shared", reader) is LookupResult.HIT
    clock.now += 2
    step(manager, clock)
    assert manager._cpu.ready(b"shared")
    assert not executor.pending
    assert b"shared" in tier.keys

    manager.on_request_finished(reader)
    clock.now += 2
    step(manager, clock)
    assert manager._cpu.status(b"shared") is None
    executor.run_next().result(timeout=1)
    step(manager, clock)
    assert tier.deleted == {b"shared"}


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
@pytest.mark.parametrize("seed", [0, 19, 57])
def test_seeded_cache_policy_preserves_allocation_and_pin_invariants(
    make_manager, cache_policy, seed
):
    rng = random.Random(seed)
    lifecycle = LifecycleConfig(
        residency_tracking_enabled=True,
        idle_ttl_sec=1.0,
        cpu_demote_after_sec=0.5,
        cpu_high_watermark=0.8,
        cpu_low_watermark=0.4,
        max_sessions=8,
        residency_max_entries=16,
    )
    policy = TieringPolicyConfig(max_inflight_store_jobs=3, max_spill_jobs=2)
    manager, tier, clock = make_manager(
        capacity=5, lifecycle=lifecycle, policy=policy, cache_policy=cache_policy
    )
    keys = [f"prefix-{index}".encode() for index in range(12)]
    active = []
    serial = 0
    for _ in range(500):
        action = rng.randrange(5)
        if action == 0 and len(active) < 6:
            context = request(manager, f"request-{serial}", f"session-{serial % 7}")
            serial += 1
            active.append(context)
        elif action == 1 and active:
            context = rng.choice(active)
            selected = rng.sample(keys, rng.randint(1, 3))
            output = manager.prepare_store(selected, context)
            if output is not None and output.keys_to_store:
                manager.complete_store(output.keys_to_store, context)
        elif action == 2 and active:
            manager.lookup(rng.choice(keys), rng.choice(active))
        elif action == 3 and active:
            context = active.pop(rng.randrange(len(active)))
            manager.on_request_finished(context)
        else:
            tier.finish(success=rng.random() > 0.15)
        clock.now += rng.random() / 10
        step(manager, clock)

        primary = manager._cpu.primary
        present = {
            key: manager._cpu.status(key)
            for key in keys
            if manager._cpu.status(key) is not None
        }
        assert manager._cpu.free_slots + len(present) == manager._cpu.capacity
        assert primary._num_evictable_cache_blocks == sum(
            block.ref_cnt == 0 for block in present.values()
        )
        assert all(block.ref_cnt >= -1 for block in present.values())
        allocated_ids = [block.block_id for block in present.values()]
        assert len(allocated_ids) == len(set(allocated_ids))
        assert not set(allocated_ids).intersection(primary._free_list)
        assert len(primary._free_list) == len(set(primary._free_list))
        assert len(manager.residency) <= lifecycle.residency_max_entries
        assert manager._lifecycle.snapshot()["sessions"] <= lifecycle.max_sessions
        assert manager._manual_jobs <= policy.max_spill_jobs
        assert all(count > 0 for count in manager._device_jobs.values())

    for context in active:
        manager.on_request_finished(context)
    tier.finish()
    step(manager, clock)
    # Draining/teardown may flush one last previously reserved promotion.
    manager.shutdown()
    assert manager._shutdown_done


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_capacity_gate_protects_existing_cpu_copies(make_manager, cache_policy):
    manager, _, _ = make_manager(
        capacity=4,
        secondary=False,
        cache_policy=cache_policy,
        policy=TieringPolicyConfig(working_set_aware=True, working_set_max_new_bytes=1),
    )
    first = request(manager, "first", "first-session")
    store(manager, first, [b"a", b"b", b"c", b"d"])
    manager.on_request_finished(first)
    second = request(manager, "second", "second-session")
    assert manager.prepare_store([b"new-a", b"new-b"], second) is None
    assert manager.store_request_skipped(second.req_id)
    assert all(manager._cpu.ready(key) for key in (b"a", b"b", b"c", b"d"))
    assert manager.get_policy_snapshot()["store_skipped_requests"] == 1
    assert (
        manager.get_policy_snapshot()["pipeline_blocks"]["store_completed_blocks"] == 4
    )
    manager.on_request_finished(second)
    assert not manager.store_request_skipped(second.req_id)
    assert manager.get_policy_snapshot()["store_skipped_requests"] == 0
    # A separate small append remains eligible under the same CPU pressure.
    third = request(manager, "third", "third-session")
    output = store(manager, third, [b"new-a"])
    assert output.keys_to_store == [b"new-a"]
    assert manager._cpu.ready(b"new-a")


def test_ready_copies_do_not_consume_an_exhausted_store_budget(make_manager):
    manager, _, _ = make_manager(
        secondary=False, policy=TieringPolicyConfig(max_store_blocks_per_step=1)
    )
    context = request(manager)
    store(manager, context, [b"cached"])
    assert manager._step_blocks == 1
    # Host owns no-op handling; the plugin must not reject before it is called.
    manager.prepare_store([b"cached"], context)
    assert manager.get_policy_snapshot()["decisions"].get("step_budget", 0) == 0
    assert manager._step_blocks == 1
    assert not manager._device_jobs
