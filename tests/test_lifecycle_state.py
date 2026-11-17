"""Deterministic, dependency-free lifecycle/residency state-machine tests."""

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


@dataclass
class Request:
    req_id: str
    kv_transfer_params: dict | None = None


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def request(req_id, session=None):
    return Request(req_id, {"session_id": session} if session else None)


@contextmanager
def loaded_state_modules():
    """Load actual implementation with temporary host stubs, never global fakes."""
    base = ModuleType("vllm.v1.kv_offload.base")
    base.OffloadKey = bytes
    base.ReqContext = Request
    for name in (
        "OffloadingCounterMetadata",
        "OffloadingGaugeMetadata",
        "OffloadingHistogramMetadata",
        "OffloadingMetricMetadata",
    ):
        setattr(base, name, SimpleNamespace)
    stubs = {
        name: ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.kv_offload")
    }
    stubs[base.__name__] = base
    package = "vllm_hust_kv_tiering"
    root = Path(__file__).resolve().parents[1] / "src" / package
    stubs[package] = ModuleType(package)
    modules = {}
    with patch.dict(sys.modules, stubs):
        for short_name in ("policy", "residency", "lifecycle"):
            name = f"{package}.{short_name}"
            spec = importlib.util.spec_from_file_location(
                name, root / f"{short_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[short_name] = module
    yield SimpleNamespace(**modules)


class LifecycleStateTests(unittest.TestCase):
    def setUp(self):
        self.loading = loaded_state_modules()
        self.modules = self.loading.__enter__()
        self.clock = Clock()
        self.life = self.modules.lifecycle
        self.res = self.modules.residency
        self.manager = self.make_manager()

    def tearDown(self):
        self.loading.__exit__(None, None, None)

    def make_manager(self, **kwargs):
        options = {"residency_tracking_enabled": True, **kwargs}
        return self.life.SessionLifecycleManager(
            self.life.LifecycleConfig(**options), clock=self.clock
        )

    def add(self, req, keys):
        self.manager.on_new_request(req)
        self.assertTrue(self.manager.record_request_keys(req, keys))

    def finish(self, req):
        self.manager.on_request_finished(req)
        self.manager.on_request_finalized(req)

    def test_empty_config_is_disabled_and_matches_dataclass_defaults(self):
        config = self.life.LifecycleConfig.from_extra_config({})
        self.assertEqual(config, self.life.LifecycleConfig())
        self.assertFalse(config.enabled)
        manager = self.life.SessionLifecycleManager(config, clock=self.clock)
        req = request("r")
        manager.on_new_request(req)
        self.assertFalse(manager.tracked(req))
        self.assertFalse(manager.record_request_keys(req, [b"a"]))
        self.assertEqual(manager.snapshot()["sessions"], 0)

    def test_strict_config_rejects_bool_string_nonfinite_fractional_and_unknown(self):
        bad = [
            {"lifecycle_idle_ttl_sec": "1"},
            {"lifecycle_idle_ttl_sec": True},
            {"lifecycle_idle_ttl_sec": float("nan")},
            {"lifecycle_idle_ttl_sec": -1},
            {"lifecycle_cpu_demote_after_sec": float("inf")},
            {"lifecycle_cpu_high_watermark": True},
            {"lifecycle_cpu_low_watermark": float("nan")},
            {"lifecycle_cpu_high_watermark": 0.8},
            {"lifecycle_delete_expired_secondary": "false"},
            {"residency_tracking_enabled": 1},
            {"lifecycle_reclaim_batch_size": 1.5},
            {"residency_max_entries": True},
            {"lifecycle_max_sessions": "2"},
            {"lifecycle_max_sessions": 0},
            {"lifecycle_idle_ttl_secs": 5},
            {"residency_max_entry": 5},
        ]
        for extra in bad:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.life.LifecycleConfig.from_extra_config(extra)
        config = self.life.LifecycleConfig.from_extra_config(
            {
                "lifecycle_idle_ttl_sec": 2,
                "residency_tracking_enabled": True,
                "lifecycle_cpu_high_watermark": 0.9,
                "lifecycle_cpu_low_watermark": 0.7,
                "unrelated_host_option": True,
            }
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.idle_ttl_sec, 2)

    def test_direct_config_is_also_strict(self):
        for kwargs in (
            {"idle_ttl_sec": True},
            {"delete_expired_secondary": 1},
            {"max_sessions": 1.2},
            {"residency_max_entries": False},
            {"cpu_low_watermark": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.life.LifecycleConfig(**kwargs)

    def test_session_id_aliases_and_request_fallback(self):
        for key in ("session_id", "conversation_id", "kv_session_id"):
            req = Request("r", {key: "stable"})
            self.assertEqual(self.life.get_session_id(req), "stable")
            self.assertTrue(self.life.has_stable_session_id(req))
        req = Request("fallback", {"session_id": 42})
        self.assertEqual(self.life.get_session_id(req), "fallback")
        self.assertFalse(self.life.has_stable_session_id(req))

    def test_clock_drives_heat_without_sleep_and_duplicate_new_is_idempotent(self):
        first = request("r1", "s")
        self.add(first, [b"a"])
        self.manager.on_new_request(first)
        self.assertEqual(self.manager.get_session_heat(first), (1, 0.0))
        self.finish(first)
        self.clock.advance(3)
        second = request("r2", "s")
        self.manager.on_new_request(second)
        self.manager.record_reuse_hit(second, 2)
        self.manager.record_reuse_hit(second, 3)
        self.assertEqual(self.manager.get_session_heat(second), (2, 0.5))
        state = self.manager._sessions["s"]
        self.assertEqual(state.last_reuse_at, 3)
        self.assertEqual(state.reuse_request_count, 1)
        self.assertEqual(state.reuse_hit_blocks, 5)

    def test_shared_prefix_not_unreferenced_until_last_session_released(self):
        first, second = request("r1", "s1"), request("r2", "s2")
        self.add(first, [b"shared", b"only1"])
        self.add(second, [b"shared"])
        tracker = self.manager.residency
        self.assertEqual(tracker.snapshot()["shared_blocks"], 1)
        self.assertEqual(tracker.release_session("s1"), {b"only1"})
        self.assertEqual(tracker.get_session_keys("s2"), {b"shared"})
        self.assertEqual(tracker.release_session("s2"), {b"shared"})

    def test_overlapping_requests_keep_session_active_until_last_finished(self):
        first, second = request("r1", "s"), request("r2", "s")
        self.add(first, [b"a"])
        self.add(second, [b"b"])
        self.finish(first)
        self.assertTrue(self.manager.has_active(second))
        self.assertEqual(self.manager.residency.snapshot()["active_blocks"], 2)
        self.assertEqual(self.manager.pending_request_ids(), {"r2"})
        self.manager.on_request_finished(second)
        self.assertFalse(self.manager.has_active(second))
        self.assertEqual(self.manager.residency.snapshot()["active_blocks"], 0)
        self.assertEqual(self.manager.snapshot()["idle_sessions"], 1)

    def test_finalized_ephemeral_removed_and_finalization_is_idempotent(self):
        req = request("ephemeral")
        self.add(req, [b"a"])
        self.manager.on_request_finished(req)
        self.assertTrue(self.manager.tracked(req))
        self.manager.on_request_finalized(req)
        self.manager.on_request_finalized(req)
        self.assertFalse(self.manager.tracked(req))
        self.assertEqual(self.manager.snapshot()["sessions"], 0)
        self.assertEqual(self.manager.pending_request_ids(), set())
        self.assertEqual(self.manager.residency.get_session_keys("ephemeral"), set())
        self.assertEqual(self.manager.residency._blocks[b"a"].session_ids, set())
        self.assertFalse(self.manager.has_pending_expiration())

    def test_finalized_anonymous_ownership_expires_at_ttl_not_finalization(self):
        self.manager = self.make_manager(idle_ttl_sec=5)
        req = request("ephemeral")
        self.add(req, [b"a"])
        self.finish(req)
        self.manager.on_request_finalized(req)
        self.assertFalse(self.manager.tracked(req))
        self.assertEqual(self.manager.pending_request_ids(), set())
        self.assertEqual(self.manager.snapshot()["idle_sessions"], 1)
        self.assertEqual(self.manager.residency.get_session_keys("ephemeral"), {b"a"})
        self.clock.advance(4)
        self.assertEqual(self.manager.expire_idle_sessions().expired_sessions, 0)
        self.clock.advance(1)
        expired = self.manager.expire_idle_sessions()
        self.assertEqual(expired.expired_sessions, 1)
        self.assertEqual(expired.unreferenced_keys, {b"a"})
        self.assertEqual(self.manager.snapshot()["sessions"], 0)
        self.assertFalse(self.manager.has_pending_expiration())
        self.assertEqual(self.manager.expire_idle_sessions().expired_sessions, 0)

    def test_session_cap_preserves_anonymous_ttl_obligation(self):
        self.manager = self.make_manager(max_sessions=1, idle_ttl_sec=5)
        first, second = request("first"), request("second")
        self.add(first, [b"a"])
        self.finish(first)
        self.manager.on_new_request(second)
        self.assertFalse(self.manager.tracked(second))
        self.assertEqual(self.manager.snapshot()["sessions"], 1)
        self.assertEqual(self.manager.residency.get_session_keys("first"), {b"a"})
        self.clock.advance(5)
        self.assertEqual(self.manager.expire_idle_sessions().unreferenced_keys, {b"a"})
        self.manager.on_new_request(second)
        self.assertTrue(self.manager.tracked(second))

    def test_prefix_history_counts_distinct_interleaved_requests_not_callbacks(self):
        first, second = request("first"), request("second")
        self.add(first, [b"shared"])
        self.assertEqual(self.manager.get_admission_heat(first, [b"shared"]), (1, 0.0))
        self.add(second, [b"shared"])
        for req in (first, second, first, second):
            self.manager.record_request_keys(req, [b"shared", b"shared"])
            self.assertEqual(
                self.manager.get_admission_heat(req, [b"shared"]), (2, 0.5)
            )
        self.finish(first)
        self.finish(second)
        self.assertEqual(self.manager.pending_request_ids(), set())
        self.assertEqual(self.manager._request_keys, {})
        third = request("third")
        self.add(third, [b"shared", b"cold"])
        self.assertEqual(
            self.manager.get_admission_heat(third, [b"shared"]), (3, 2 / 3)
        )
        self.assertEqual(
            self.manager.get_admission_heat(third, [b"shared", b"cold"]), (1, 0.0)
        )

    def test_prefix_history_is_bounded_and_forgotten_keys_start_cold(self):
        self.manager = self.make_manager(residency_max_entries=2)
        for index in range(20):
            req = request(f"req-{index}")
            self.add(req, [f"key-{index}".encode()])
            self.finish(req)
            self.assertLessEqual(len(self.manager._prefix_requests), 2)
            self.assertEqual(self.manager._request_keys, {})
            self.assertEqual(self.manager.snapshot()["sessions"], 0)
        req = request("repeat")
        self.add(req, [b"key-0", b"key-19"])
        self.assertEqual(self.manager.get_admission_heat(req, [b"key-0"]), (1, 0.0))
        self.assertEqual(self.manager.get_admission_heat(req, [b"key-19"]), (2, 0.5))

    def test_finalize_without_finished_does_not_leak_active_ephemeral(self):
        req = request("ephemeral")
        self.add(req, [b"a"])
        self.manager.on_request_finalized(req)
        self.assertEqual(self.manager.snapshot()["sessions"], 0)
        self.assertEqual(self.manager.residency._active_sessions, set())

    def test_idle_ttl_protects_keys_unfinalized_work_and_active_sessions(self):
        self.manager = self.make_manager(idle_ttl_sec=5)
        req = request("r", "s")
        self.add(req, [b"a"])
        self.clock.advance(10)
        self.assertFalse(self.manager.has_expiration_due())
        self.manager.on_request_finished(req)
        self.clock.advance(5)
        self.assertTrue(self.manager.has_expiration_due())
        self.assertEqual(
            self.manager.expire_idle_sessions(protected_req_ids={"r"}).expired_sessions,
            0,
        )
        self.assertEqual(self.manager.expire_idle_sessions().expired_sessions, 0)
        self.manager.on_request_finalized(req)
        self.assertEqual(
            self.manager.expire_idle_sessions(protected_keys={b"a"}).expired_sessions, 0
        )
        result = self.manager.expire_idle_sessions()
        self.assertEqual(result.expired_sessions, 1)
        self.assertEqual(result.unreferenced_keys, {b"a"})
        self.assertFalse(self.manager.has_pending_expiration())

    def test_returning_session_cancels_ttl_and_updates_clock(self):
        self.manager = self.make_manager(idle_ttl_sec=5)
        first = request("r1", "s")
        self.add(first, [b"a"])
        self.finish(first)
        self.clock.advance(4)
        second = request("r2", "s")
        self.manager.on_new_request(second)
        self.clock.advance(100)
        self.assertFalse(self.manager.has_pending_expiration())
        self.assertEqual(self.manager.expire_idle_sessions().expired_sessions, 0)
        self.assertEqual(self.manager.residency._blocks[b"a"].last_access_at, 4)

    def test_active_session_cap_refuses_new_tracking_without_assert_or_eviction(self):
        self.manager = self.make_manager(max_sessions=1)
        first, second = request("r1", "s1"), request("r2", "s2")
        self.add(first, [b"a"])
        self.manager.on_new_request(second)
        self.assertFalse(self.manager.tracked(second))
        self.assertEqual(self.manager.tracking_rejections, 1)
        self.assertFalse(self.manager.record_request_keys(second, [b"b"]))
        self.manager.on_request_finished(second)
        self.manager.on_request_finalized(second)
        self.assertEqual(self.manager.snapshot()["sessions"], 1)
        self.assertEqual(self.manager.residency.get_session_keys("s1"), {b"a"})
        self.assertEqual(self.manager.pending_request_ids(), {"r1"})

    def test_retained_session_cannot_be_pruned_before_async_finalize(self):
        self.manager = self.make_manager(max_sessions=1)
        first, second = request("r1", "s1"), request("r2", "s2")
        self.add(first, [b"a"])
        self.manager.on_request_finished(first)
        self.manager.on_new_request(second)
        self.assertFalse(self.manager.tracked(second))
        self.assertEqual(self.manager.prune_idle_sessions().pruned_sessions, 0)
        self.manager.on_request_finalized(first)
        self.manager.on_new_request(second)
        self.assertTrue(self.manager.tracked(second))
        self.assertEqual(set(self.manager._sessions), {"s2"})
        self.assertEqual(self.manager.residency.get_session_keys("s1"), set())

    def test_stable_session_lru_metadata_pressure_does_not_delete_secondary_copy(self):
        self.manager = self.make_manager(max_sessions=2)
        first, second = request("r1", "s1"), request("r2", "s2")
        self.add(first, [b"a"])
        self.manager.residency.mark_secondary_resident([b"a"], "fs")
        self.finish(first)
        self.clock.advance(1)
        self.add(second, [b"b"])
        self.finish(second)
        third = request("r3", "s3")
        self.manager.on_new_request(third)
        self.assertEqual(set(self.manager._sessions), {"s2", "s3"})
        self.assertEqual(self.manager.residency.get_session_keys("s1"), set())
        self.assertTrue(self.manager.residency.has_secondary_copy(b"a"))
        self.assertEqual(self.manager.residency._blocks[b"a"].session_ids, set())

    def test_metadata_prune_returns_no_file_deletion_authority(self):
        self.manager = self.make_manager(max_sessions=2)
        for req, key in ((request("r1", "s1"), b"a"), (request("r2", "s2"), b"b")):
            self.add(req, [key])
            self.manager.residency.mark_secondary_resident([key], "fs")
            self.finish(req)
            self.clock.advance(1)
        self.manager.config.max_sessions = 1
        result = self.manager.prune_idle_sessions()
        self.assertEqual(result.pruned_sessions, 1)
        self.assertEqual(result.unreferenced_keys, set())
        self.assertTrue(self.manager.residency.has_secondary_copy(b"a"))

    def test_many_active_and_stable_sessions_stay_within_hard_limit(self):
        self.manager = self.make_manager(max_sessions=3)
        for index in range(50):
            req = request(f"r{index}", f"s{index}")
            self.manager.on_new_request(req)
            if self.manager.tracked(req):
                self.assertTrue(
                    self.manager.record_request_keys(req, [str(index).encode()])
                )
                self.finish(req)
            self.assertLessEqual(self.manager.snapshot()["sessions"], 3)
        self.assertEqual(self.manager.snapshot()["sessions"], 3)
        self.assertEqual(self.manager.pending_request_ids(), set())

    def test_request_key_admission_is_atomic_and_keeps_existing_ownership(self):
        self.manager = self.make_manager(residency_max_entries=2)
        req = request("r", "s")
        self.add(req, [b"a"])
        self.assertFalse(self.manager.record_request_keys(req, [b"b", b"c"]))
        self.assertEqual(self.manager._sessions["s"].block_keys, {b"a"})
        self.assertEqual(self.manager.residency.get_session_keys("s"), {b"a"})
        self.assertEqual(set(self.manager.residency._blocks), {b"a"})
        self.assertTrue(self.manager.record_request_keys(req, [b"a", b"b", b"b"]))
        self.assertEqual(self.manager.snapshot()["retained_blocks"], 2)
        self.assertEqual(self.manager.tracking_rejections, 1)

    def test_idle_demotion_requires_secondary_idle_age_no_active_shared_or_inflight(
        self,
    ):
        self.manager = self.make_manager(cpu_demote_after_sec=5)
        req = request("r", "s")
        keys = [b"candidate", b"inflight", b"no_secondary", b"shared"]
        self.add(req, keys)
        tracker = self.manager.residency
        tracker.mark_cpu_resident(keys)
        tracker.mark_secondary_resident([b"candidate", b"inflight", b"shared"], "fs")
        tracker.start_transfer([b"inflight"], "cpu", "fs")
        other = request("r2", "s2")
        self.add(other, [b"shared"])
        self.assertEqual(
            self.manager.get_idle_cpu_candidates(limit=20, require_idle_age=False), []
        )
        self.finish(req)
        self.assertEqual(
            self.manager.get_idle_cpu_candidates(limit=20, require_idle_age=True), []
        )
        self.clock.advance(5)
        self.assertEqual(
            self.manager.get_idle_cpu_candidates(limit=20, require_idle_age=True),
            [b"candidate"],
        )
        self.assertTrue(self.manager.has_pending_cpu_demotion())
        self.assertEqual(
            tracker.get_cpu_pressure_candidates(limit=20, include_active=False),
            [b"candidate", b"no_secondary"],
        )

    def test_heat_only_mode_honors_session_bound_and_finalizes_ephemeral(self):
        manager = self.life.SessionLifecycleManager(
            self.life.LifecycleConfig(max_sessions=1), clock=self.clock
        )
        first, second = request("a"), request("b")
        manager.on_new_request(first, track_heat=True)
        manager.on_new_request(second, track_heat=True)
        self.assertFalse(manager.tracked(second))
        manager.on_request_finished(first, track_heat=True)
        manager.on_request_finalized(first)
        self.assertEqual(manager.snapshot()["sessions"], 0)


class ResidencyStateTests(unittest.TestCase):
    def setUp(self):
        self.loading = loaded_state_modules()
        self.modules = self.loading.__enter__()
        self.res = self.modules.residency
        self.clock = Clock()
        self.tracker = self.res.BlockResidencyTracker(2, clock=self.clock)

    def tearDown(self):
        self.loading.__exit__(None, None, None)

    def test_cheap_presence_and_count_include_nonresident_metadata(self):
        self.assertEqual(len(self.tracker), 0)
        self.assertFalse(self.tracker.contains_entry(b"a"))
        self.tracker.record_access("s", [b"a"])
        self.assertEqual(len(self.tracker), 1)
        self.assertTrue(self.tracker.contains_entry(b"a"))
        self.assertFalse(self.tracker.is_known(b"a"))
        keys = self.tracker.tracked_keys()
        keys.clear()
        self.assertEqual(self.tracker.tracked_keys(), {b"a"})
        self.tracker.mark_cpu_resident([b"a"])
        self.assertTrue(self.tracker.is_known(b"a"))
        self.assertEqual(len(self.tracker), 1)

    def test_capacity_prunes_only_unowned_noncpu_noninflight_lru(self):
        self.tracker.mark_secondary_resident([b"old"], "fs")
        self.clock.advance(1)
        self.tracker.mark_secondary_resident([b"newer"], "fs")
        self.assertTrue(self.tracker.record_access("s", [b"needed"]))
        self.assertEqual(set(self.tracker._blocks), {b"newer", b"needed"})
        self.assertEqual(self.tracker.get_session_keys("s"), {b"needed"})

    def test_capacity_never_prunes_requested_existing_keys(self):
        self.tracker.mark_secondary_resident([b"existing"], "fs")
        self.tracker.mark_secondary_resident([b"other"], "fs")
        self.assertTrue(self.tracker.record_access("s", [b"existing", b"new"]))
        self.assertEqual(set(self.tracker._blocks), {b"existing", b"new"})

    def test_all_owned_capacity_failure_has_no_partial_pruning_or_ownership(self):
        self.assertTrue(self.tracker.record_access("s", [b"a", b"b"]))
        self.assertFalse(self.tracker.ensure_capacity([b"c"]))
        self.assertFalse(self.tracker.record_access("other", [b"a", b"c"]))
        self.assertNotIn("other", self.tracker._session_keys)
        self.assertEqual(self.tracker._blocks[b"a"].session_ids, {"s"})
        self.assertEqual(set(self.tracker._blocks), {b"a", b"b"})

    def test_failed_batch_does_not_even_prune_available_hint(self):
        self.tracker.mark_secondary_resident([b"hint"], "fs")
        self.tracker.mark_cpu_resident([b"cpu"])
        self.assertFalse(self.tracker.ensure_capacity([b"new1", b"new2"]))
        self.assertEqual(set(self.tracker._blocks), {b"hint", b"cpu"})

    def test_positive_mark_and_start_raise_explicit_capacity_without_partial_mutation(
        self,
    ):
        self.tracker.record_access("s", [b"a", b"b"])
        operations = (
            lambda: self.tracker.mark_cpu_resident([b"a", b"c"]),
            lambda: self.tracker.mark_secondary_resident([b"a", b"c"], "fs"),
            lambda: self.tracker.start_transfer([b"a", b"c"], "cpu", "fs"),
        )
        for operation in operations:
            with (
                self.subTest(operation=operation),
                self.assertRaises(self.res.ResidencyCapacityError),
            ):
                operation()
        state = self.tracker._blocks[b"a"]
        self.assertFalse(state.cpu_resident)
        self.assertFalse(state.secondary_tiers)
        self.assertFalse(state.inflight_transfers)
        self.assertEqual(len(self.tracker._blocks), 2)

    def test_cpu_and_inflight_ownership_cannot_be_evicted(self):
        self.tracker.mark_cpu_resident([b"cpu"])
        self.tracker.start_transfer([b"inflight"], "cpu", "fs")
        self.assertFalse(self.tracker.ensure_capacity([b"new"]))
        self.tracker.finish_transfer([b"inflight"], "cpu", "fs")
        self.assertTrue(self.tracker.record_access("s", [b"new"]))
        self.assertEqual(set(self.tracker._blocks), {b"cpu", b"new"})

    def test_unknown_negative_marks_and_finish_do_not_allocate_or_claim_proof(self):
        self.tracker.mark_cpu_resident([b"unknown"], False)
        self.tracker.mark_secondary_resident([b"unknown"], "fs", False)
        self.tracker.finish_transfer([b"unknown"], "cpu", "fs")
        self.assertEqual(self.tracker.snapshot()["tracked_blocks"], 0)
        self.assertFalse(self.tracker.is_known(b"unknown"))
        self.assertFalse(self.tracker.has_secondary_copy(b"unknown"))
        self.assertEqual(self.tracker.release_session("unknown"), set())

    def test_release_missing_state_never_claims_unreferenced(self):
        # Simulate an incomplete legacy hint: absence is not ownership proof.
        self.tracker._session_keys["s"] = {b"missing"}
        self.assertEqual(self.tracker.release_session("s"), set())

    def test_transfer_counter_handles_overlap_and_duplicate_keys_exactly_once(self):
        self.tracker.start_transfer([b"a", b"a"], "cpu", "fs")
        self.tracker.start_transfer([b"a"], "cpu", "fs")
        self.assertEqual(
            self.tracker._blocks[b"a"].inflight_transfers[("cpu", "fs")], 2
        )
        self.tracker.finish_transfer([b"a", b"a"], "cpu", "fs")
        self.assertEqual(self.tracker.get_inflight_keys(), {b"a"})
        self.tracker.finish_transfer([b"a"], "cpu", "fs")
        self.tracker.finish_transfer([b"a"], "cpu", "fs")
        self.assertEqual(self.tracker.get_inflight_keys(), set())

    def test_zero_clock_idle_timestamp_does_not_reset_on_release(self):
        self.tracker.record_access("s", [b"a"])
        self.assertEqual(self.tracker._blocks[b"a"].idle_since, 0)
        self.clock.advance(10)
        self.tracker.release_session("s")
        self.assertEqual(self.tracker._blocks[b"a"].idle_since, 0)

    def test_repeated_idle_transition_preserves_idle_age_and_nonpositive_limits(self):
        self.tracker.on_session_active("s")
        self.tracker.record_access("s", [b"a"])
        self.tracker.mark_cpu_resident([b"a"])
        self.tracker.mark_secondary_resident([b"a"], "fs")
        self.tracker.on_session_idle("s")
        self.clock.advance(10)
        self.tracker.on_session_idle("s")
        self.assertEqual(self.tracker._blocks[b"a"].idle_since, 0)
        self.assertEqual(self.tracker.get_idle_cpu_candidates(limit=-1), [])
        self.assertEqual(self.tracker.get_cpu_pressure_candidates(limit=0), [])

    def test_direct_tracker_limit_strict_and_hard_bound_under_repeated_hints(self):
        for bad in (True, 0, -1, 1.5, "2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.res.BlockResidencyTracker(bad)
        for index in range(100):
            self.tracker.mark_secondary_resident([str(index).encode()], "fs")
            self.assertLessEqual(len(self.tracker._blocks), 2)

    def test_import_fixture_restores_host_and_implementation_modules(self):
        names = [
            "vllm.v1.kv_offload.base",
            "vllm_hust_kv_tiering.lifecycle",
            "vllm_hust_kv_tiering.residency",
        ]
        before = {name: sys.modules.get(name) for name in names}
        with loaded_state_modules() as modules:
            self.assertTrue(hasattr(modules.lifecycle, "SessionLifecycleManager"))
            self.assertEqual({name: sys.modules.get(name) for name in names}, before)
        self.assertEqual({name: sys.modules.get(name) for name in names}, before)


if __name__ == "__main__":
    unittest.main()
