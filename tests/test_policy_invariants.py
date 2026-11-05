"""Seeded scheduler simulations over the real host LRU/ARC bookkeeping."""

import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_offload.tiering.manager")

from test_tiering_manager import make_manager as make_manager  # noqa: PLC0414 - pytest fixture re-export
from test_tiering_manager import request, step
from vllm_hust_kv_tiering.lifecycle import LifecycleConfig
from vllm_hust_kv_tiering.policy import TieringPolicyConfig

pytestmark = pytest.mark.host


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
