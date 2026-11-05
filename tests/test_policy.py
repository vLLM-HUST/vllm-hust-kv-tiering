"""Admission logic is testable without torch, vLLM or an accelerator."""

import pytest

from vllm_hust_kv_tiering.policy import (
    PressureGate,
    TieringPolicyConfig,
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
