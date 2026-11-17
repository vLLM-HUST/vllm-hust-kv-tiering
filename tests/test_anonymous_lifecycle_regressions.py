"""Anonymous-request TTL and content-history regressions on the real CPU host."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.tiering.manager")

from test_manager_edge_cases import ManualDeleteExecutor
from test_tiering_manager import make_manager as make_manager  # noqa: PLC0414 - pytest fixture re-export
from test_tiering_manager import request, step, store
from vllm.v1.kv_offload.base import LookupResult

from vllm_hust_kv_tiering.lifecycle import LifecycleConfig
from vllm_hust_kv_tiering.policy import TieringPolicyConfig

pytestmark = pytest.mark.host


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
