"""Portable pressure, cost and bounded working-set admission tests."""

import pytest

from vllm_hust_kv_tiering.policy import (
    PressureGate,
    TieringPolicyConfig,
    WorkingSetAdmission,
    admission_reason,
)


def test_hysteresis_and_preemption():
    gate = PressureGate(0.8, 0.6)
    assert not gate.update(0.79)
    assert gate.update(0.8)
    assert gate.update(0.7)
    assert not gate.update(0.6)
    assert gate.update(0.1, preempted=True)
    assert not gate.update(0.1)


@pytest.mark.parametrize("usage", [float("nan"), float("inf"), -0.01, 1.01, None])
def test_invalid_pressure_is_not_silently_accepted(usage):
    with pytest.raises(ValueError):
        PressureGate(0.8, 0.6).update(usage)


@pytest.mark.parametrize(
    "extra",
    [
        {"tiering_hbm_pressure_aware": "false"},
        {"tiering_hbm_high_watermark": float("nan")},
        {"tiering_hbm_high_watermark": 0.4, "tiering_hbm_low_watermark": 0.6},
        {"tiering_min_reuse_probability": 1.1},
        {"tiering_max_spill_jobs": True},
        {"tiering_max_device_store_blocks_per_step": 0},
        {"tiering_max_inflight_device_store_jobs": 1.5},
        {"tiering_prefill_tokens_per_second": 1},
        {"tiering_store_bandwidth_bytes_per_second": -1},
        {"tiering_benefit_safety_factor": 0.9},
        {"tiering_maintenance_interval_sec": 0},
        {"tiering_retry_backoff_sec": float("inf")},
        {"tiering_reclaim_device_cache_after_store": True},
        {"tiering_typo": True},
    ],
)
def test_config_fails_closed(extra):
    with pytest.raises(ValueError):
        TieringPolicyConfig.from_extra_config(extra)


def reason(config, probability=0.0, requests=1):
    return admission_reason(
        config,
        requests=requests,
        reuse_probability=probability,
        num_blocks=10,
        bytes_per_block=1000,
        tokens_per_block=100,
    )


def test_default_policy_does_not_invent_cost_or_reuse_benefit():
    config = TieringPolicyConfig.from_extra_config({})
    assert config.pressure_aware
    assert reason(config) is None  # Pressure gating is separate.
    assert reason(TieringPolicyConfig(min_session_requests=2)) == "session_history"
    assert reason(TieringPolicyConfig(min_reuse_probability=0.5)) == "reuse_probability"


def test_cost_gate_requires_positive_net_expected_benefit():
    config = TieringPolicyConfig(
        prefill_tokens_per_second=100,
        restore_bytes_per_second=10000,
        store_bytes_per_second=10000,
    )
    assert reason(config, probability=0.0) == "estimated_cost"
    assert reason(config, probability=1.0) is None
    slow_restore = TieringPolicyConfig(
        prefill_tokens_per_second=10000,
        restore_bytes_per_second=100,
        store_bytes_per_second=10000,
    )
    assert reason(slow_restore, probability=1.0) == "estimated_cost"


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def gate(**kwargs):
    clock = Clock()
    config = TieringPolicyConfig(
        working_set_aware=True,
        working_set_window_sec=10,
        working_set_max_new_bytes=20,
        **kwargs,
    )
    return WorkingSetAdmission(config, clock=clock), clock


def working_set_reason(policy, *, free=0, new=3, written=0, capacity=4):
    return policy.reason(
        capacity_blocks=capacity,
        free_blocks=free,
        new_blocks=new,
        admitted_request_blocks=written,
        bytes_per_block=10,
    )


def test_large_cold_fill_declined_only_with_capacity_displacement():
    policy, _ = gate()
    policy.observe(range(4))
    assert (
        working_set_reason(policy) is None
    )  # Full cache does not imply an oversized set.
    policy.observe((4, 5))
    assert working_set_reason(policy, free=3) is None  # Real free capacity is usable.
    assert working_set_reason(policy) == "working_set_capacity"
    assert working_set_reason(policy, new=2) is None  # A small append remains eligible.


def test_split_prefill_cannot_bypass_request_write_budget():
    policy, _ = gate()
    policy.observe(range(6))
    assert working_set_reason(policy, new=1, written=1) is None
    assert working_set_reason(policy, new=1, written=2) == "working_set_capacity"


def test_repeated_callbacks_and_shared_prefixes_do_not_inflate_set():
    policy, _ = gate()
    for _ in range(50):
        policy.observe((b"shared", b"shared", b"tail"))
    assert policy.snapshot(bytes_per_block=10)["tracked_blocks"] == 2
    assert working_set_reason(policy) is None


def test_window_recovers_after_old_competing_prefixes_expire():
    policy, clock = gate()
    policy.observe(range(6))
    assert working_set_reason(policy) == "working_set_capacity"
    clock.now = 9
    policy.observe((0, 1))
    clock.now = 10
    assert working_set_reason(policy) is None
    assert policy.snapshot(bytes_per_block=10)["tracked_bytes"] == 20


def test_history_is_bounded_and_overflow_is_not_reported_as_capacity_fit():
    policy, clock = gate(working_set_max_entries=2)
    policy.observe(range(100))
    snapshot = policy.snapshot(bytes_per_block=10)
    assert snapshot["tracked_blocks"] == 2
    assert snapshot["censored"]
    assert working_set_reason(policy, capacity=1000) == "working_set_uncertain"
    clock.now = 10
    assert working_set_reason(policy, capacity=1000) is None
    assert not policy.snapshot(bytes_per_block=10)["censored"]


def test_salt_and_hybrid_group_identity_remain_distinct():
    policy, _ = gate()
    keys = [(b"same-prefix", salt, group) for salt in ("a", "b") for group in (0, 1)]
    policy.observe(keys)
    assert policy.snapshot(bytes_per_block=10)["tracked_blocks"] == 4
    assert working_set_reason(policy, capacity=3) == "working_set_capacity"


def test_disabled_policy_does_not_allocate_history_or_decline_stores():
    policy = WorkingSetAdmission(TieringPolicyConfig())
    policy.observe(range(100_000))
    assert policy.snapshot(bytes_per_block=10)["tracked_blocks"] == 0
    assert working_set_reason(policy) is None


def test_reset_drops_both_history_and_uncertainty():
    policy, _ = gate(working_set_max_entries=2)
    policy.observe(range(6))
    policy.reset()
    assert working_set_reason(policy) is None
    snapshot = policy.snapshot(bytes_per_block=10)
    assert snapshot["tracked_blocks"] == 0
    assert not snapshot["censored"]


@pytest.mark.parametrize(
    "extra",
    [
        {"tiering_working_set_aware": "true"},
        {"tiering_working_set_window_sec": True},
        {"tiering_working_set_window_sec": 0},
        {"tiering_working_set_window_sec": float("nan")},
        {"tiering_working_set_max_entries": 0},
        {"tiering_working_set_max_entries": 1.5},
        {"tiering_working_set_max_new_bytes": True},
        {"tiering_working_set_max_new_bytes": -1},
    ],
)
def test_invalid_working_set_config_fails_closed(extra):
    with pytest.raises(ValueError):
        TieringPolicyConfig.from_extra_config(extra)


def test_configuration_round_trip_and_byte_threshold():
    config = TieringPolicyConfig.from_extra_config(
        {
            "tiering_working_set_aware": True,
            "tiering_working_set_window_sec": 30,
            "tiering_working_set_max_entries": 100,
            "tiering_working_set_max_new_bytes": 4096,
        }
    )
    assert config.working_set_aware
    assert config.working_set_window_sec == 30
    assert config.working_set_max_entries == 100
    assert config.working_set_max_new_bytes == 4096
