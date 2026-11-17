"""Additional frozen-host CPU integration gates; intentionally skipped on Mac."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.tiering.manager")

from test_tiering_manager import make_manager as make_manager  # noqa: PLC0414
from test_tiering_manager import request, store

from vllm_hust_kv_tiering.policy import TieringPolicyConfig

pytestmark = pytest.mark.host


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
