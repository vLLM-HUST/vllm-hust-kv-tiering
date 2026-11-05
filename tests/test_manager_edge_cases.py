"""Frozen-host manager edge regressions; requires the real host CPU APIs.

The shared fixture builds actual CPUPrimaryTierOffloadingManager and
TieringOffloadingManager objects. Only the secondary I/O tier and delete
executor are deterministic fakes; no host implementation is stubbed.
"""

from collections import deque
from concurrent.futures import Future

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.tiering.manager")

from test_tiering_manager import make_manager as make_manager  # noqa: PLC0414 - pytest fixture re-export
from test_tiering_manager import request, step, store
from vllm.v1.kv_offload.base import LookupResult
from vllm_hust_kv_tiering.base import TierDeleteResult
from vllm_hust_kv_tiering.lifecycle import LifecycleConfig
from vllm_hust_kv_tiering.policy import TieringPolicyConfig

pytestmark = pytest.mark.host


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
