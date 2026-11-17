"""Candidate-set admission needs neither a device nor a host replacement."""

import pytest

from vllm_hust_kv_tiering.policy import TieringPolicyConfig, WorkingSetAdmission


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


def reason(policy, *, free=0, new=3, written=0, capacity=4):
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
    assert reason(policy) is None  # Full cache does not imply an oversized set.
    policy.observe((4, 5))
    assert reason(policy, free=3) is None  # Real free capacity is usable.
    assert reason(policy) == "working_set_capacity"
    assert reason(policy, new=2) is None  # A small append remains eligible.


def test_split_prefill_cannot_bypass_request_write_budget():
    policy, _ = gate()
    policy.observe(range(6))
    assert reason(policy, new=1, written=1) is None
    assert reason(policy, new=1, written=2) == "working_set_capacity"


def test_repeated_callbacks_and_shared_prefixes_do_not_inflate_set():
    policy, _ = gate()
    for _ in range(50):
        policy.observe((b"shared", b"shared", b"tail"))
    assert policy.snapshot(bytes_per_block=10)["tracked_blocks"] == 2
    assert reason(policy) is None


def test_window_recovers_after_old_competing_prefixes_expire():
    policy, clock = gate()
    policy.observe(range(6))
    assert reason(policy) == "working_set_capacity"
    clock.now = 9
    policy.observe((0, 1))
    clock.now = 10
    assert reason(policy) is None
    assert policy.snapshot(bytes_per_block=10)["tracked_bytes"] == 20


def test_history_is_bounded_and_overflow_is_not_reported_as_capacity_fit():
    policy, clock = gate(working_set_max_entries=2)
    policy.observe(range(100))
    snapshot = policy.snapshot(bytes_per_block=10)
    assert snapshot["tracked_blocks"] == 2
    assert snapshot["censored"]
    assert reason(policy, capacity=1000) == "working_set_uncertain"
    clock.now = 10
    assert reason(policy, capacity=1000) is None
    assert not policy.snapshot(bytes_per_block=10)["censored"]


def test_salt_and_hybrid_group_identity_remain_distinct():
    policy, _ = gate()
    keys = [(b"same-prefix", salt, group) for salt in ("a", "b") for group in (0, 1)]
    policy.observe(keys)
    assert policy.snapshot(bytes_per_block=10)["tracked_blocks"] == 4
    assert reason(policy, capacity=3) == "working_set_capacity"


def test_disabled_policy_does_not_allocate_history_or_decline_stores():
    policy = WorkingSetAdmission(TieringPolicyConfig())
    policy.observe(range(100_000))
    assert policy.snapshot(bytes_per_block=10)["tracked_blocks"] == 0
    assert reason(policy) is None


def test_reset_drops_both_history_and_uncertainty():
    policy, _ = gate(working_set_max_entries=2)
    policy.observe(range(6))
    policy.reset()
    assert reason(policy) is None
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
