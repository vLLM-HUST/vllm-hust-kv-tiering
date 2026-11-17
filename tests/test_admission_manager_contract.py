"""Actual plugin manager with small explicit host-interface fixtures.

These are portable contract tests, NOT frozen-vLLM integration, DMA tests or
model correctness evidence. Imports are restored before running each test.
"""

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from test_lifecycle_state import Clock, Request, loaded_state_modules


class Lookup(Enum):
    MISS = 0
    HIT = 1


@dataclass
class ScheduleEnd:
    scheduled: tuple = ()
    finished: tuple = ()


@contextmanager
def loaded_manager():
    with loaded_state_modules() as state:
        base = ModuleType("vllm.v1.kv_offload.base")
        base.LookupResult = Lookup
        base.OffloadKey = bytes
        base.ReqContext = Request
        base.ScheduleEndContext = ScheduleEnd
        base.OffloadingEvent = SimpleNamespace
        tier = ModuleType("vllm.v1.kv_offload.tiering.base")
        tier.JobMetadata = SimpleNamespace
        tier.JobResult = SimpleNamespace
        stats_module = ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics"
        )
        stats_module.OffloadingConnectorStats = Stats
        package = "vllm_hust_kv_tiering"
        stubs = {
            base.__name__: base,
            tier.__name__: tier,
            stats_module.__name__: stats_module,
        }
        for short in ("policy", "residency", "lifecycle"):
            stubs[f"{package}.{short}"] = getattr(state, short)
        modules = {}
        with patch.dict(sys.modules, stubs):
            for short in ("host_adapter", "lifecycle_manager"):
                name = f"{package}.{short}"
                path = (
                    Path(__file__).resolve().parents[1]
                    / "src"
                    / package
                    / f"{short}.py"
                )
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[short] = module
            # get_stats imports its host type at call time. Bind the temporary
            # fixture only during calls; never leak it to host-test collection.
        yield SimpleNamespace(**vars(state), **modules, stats_module=stats_module)


class Stats:
    def __init__(self):
        self.counters = {}
        self.gauges = {}

    def increase_counter(self, name, value, labels=()):
        key = (name, labels)
        self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name, value, labels=()):
        self.gauges[(name, labels)] = value


class Policy(dict):
    def remove(self, key):
        self.pop(key)


class Primary:
    def __init__(self, capacity):
        self._policy = Policy()
        self._num_blocks = capacity
        self._num_allocated_blocks = 0
        self._num_evictable_cache_blocks = 0
        self._free_list = []
        self.events = None

    def put(self, key, ready=True):
        if key not in self._policy:
            self._num_allocated_blocks += 1
            self._num_evictable_cache_blocks += int(ready)
        self._policy[key] = SimpleNamespace(is_ready=ready, ref_cnt=0)

    def _free_block(self, block):
        self._free_list.append(block)

    def prepare_read(self, keys, context):
        return SimpleNamespace(block_ids=list(range(len(keys))))

    def complete_read(self, keys, context):
        pass


class Delegate:
    def __init__(self, capacity):
        self.primary_tier = Primary(capacity)
        self.secondary_tiers = []
        self.prepare_calls = []
        self.polls = 0

    def _process_finished_jobs(self):
        self.polls += 1

    def on_new_request(self, context):
        pass

    def on_request_finished(self, context):
        pass

    def lookup(self, key, context):
        value = self.primary_tier._policy.get(key)
        return Lookup.HIT if value is not None and value.is_ready else Lookup.MISS

    def prepare_store(self, keys, context):
        self.prepare_calls.append(list(keys))
        fresh = [
            key for key in dict.fromkeys(keys) if key not in self.primary_tier._policy
        ]
        if any(
            not self.primary_tier._policy[key].is_ready
            for key in keys
            if key not in fresh
        ):
            return None
        for key in fresh:
            self.primary_tier.put(key, ready=False)
        return SimpleNamespace(keys_to_store=fresh, evicted_keys=[])

    def complete_store(self, keys, context, success=True):
        for key in keys:
            if success:
                self.primary_tier._policy[key].is_ready = True
            else:
                self.primary_tier._policy.pop(key)
        self.primary_tier._num_evictable_cache_blocks += len(keys) * int(success)

    def prepare_load(self, keys, context):
        return SimpleNamespace(block_ids=list(range(len(keys))))

    def complete_load(self, keys, context):
        pass

    def reset_cache(self):
        self.primary_tier._policy.clear()
        self.primary_tier._num_allocated_blocks = 0
        self.primary_tier._num_evictable_cache_blocks = 0
        self.primary_tier._free_list.clear()

    def get_stats(self):
        return None


@pytest.fixture
def environment():
    with loaded_manager() as modules:
        yield modules


def manager(modules, *, capacity=8, **options):
    clock, delegate = Clock(), Delegate(capacity)
    lifecycle = modules.lifecycle.SessionLifecycleManager(
        modules.lifecycle.LifecycleConfig(residency_tracking_enabled=True), clock=clock
    )
    owner = modules.lifecycle_manager.LifecycleAwareManager(
        delegate,
        lifecycle,
        modules.policy.TieringPolicyConfig(**options),
        bytes_per_block=10,
        tokens_per_block=100,
        clock=clock,
    )
    owner.bind_pressure_provider(lambda: 0.9)
    owner.begin_step()
    return owner, delegate, clock


def request(owner, req_id="r", session=None):
    context = Request(req_id, {"session_id": session} if session else None)
    owner.on_new_request(context)
    return context


def fill(primary, keys):
    for key in keys:
        primary.put(key)


def test_existing_copies_bypass_new_write_budgets_and_cost_gate(environment):
    owner, delegate, _ = manager(
        environment, min_session_requests=2, max_store_blocks_per_step=1
    )
    fill(delegate.primary_tier, [b"existing"])
    owner._step_blocks = 1
    context = request(owner)
    result = owner.prepare_store([b"existing"], context)
    assert result.keys_to_store == []
    assert owner.get_policy_snapshot()["decisions"]["already_present"] == 1
    assert not owner._device_jobs


def test_existing_cold_copy_does_not_hide_repeated_missing_prefix(environment):
    owner, delegate, _ = manager(environment, min_session_requests=2)
    first = request(owner, "first")
    assert owner.lookup(b"repeat", first) == Lookup.MISS
    owner.on_request_finished(first)
    fill(delegate.primary_tier, [b"existing"])
    second = request(owner, "second")
    result = owner.prepare_store([b"existing", b"repeat"], second)
    assert result.keys_to_store == [b"repeat"]
    assert delegate.prepare_calls[-1] == [b"existing", b"repeat"]


def test_mixed_hot_and_cold_missing_keys_do_not_borrow_content_heat(environment):
    owner, _, _ = manager(environment, min_session_requests=2)
    first = request(owner, "first")
    owner.lookup(b"hot", first)
    owner.on_request_finished(first)
    second = request(owner, "second")
    assert owner.prepare_store([b"hot", b"cold"], second) is None
    assert owner.get_policy_snapshot()["decisions"]["session_history"] == 1


def test_large_checkpoint_bundle_is_declined_without_partial_host_mutation(environment):
    owner, delegate, _ = manager(
        environment,
        capacity=4,
        working_set_aware=True,
        working_set_max_new_bytes=10,
    )
    fill(delegate.primary_tier, [b"old0", b"old1", b"old2", b"old3"])
    context = request(owner)
    for key in (b"old0", b"old1", b"old2", b"old3"):
        owner.lookup(key, context)
    keys = [b"attention-group", b"recurrent-group"]
    assert owner.prepare_store(keys, context) is None
    assert not delegate.prepare_calls
    assert not owner._device_jobs
    assert owner.get_policy_snapshot()["store_skipped_requests"] == 1
    assert owner.get_policy_snapshot()["decisions"]["working_set_capacity"] == 1


def test_request_skip_survives_window_expiry_and_reset_but_not_finalization(
    environment,
):
    owner, delegate, clock = manager(
        environment,
        capacity=2,
        working_set_aware=True,
        working_set_window_sec=1,
        working_set_max_new_bytes=10,
    )
    fill(delegate.primary_tier, [b"a", b"b"])
    context = request(owner)
    assert owner.prepare_store([b"c", b"d", b"e"], context) is None
    clock.advance(2)
    owner.reset_cache()
    owner.begin_step()
    assert owner.prepare_store([b"c"], context) is None
    owner.on_request_finished(context)
    assert owner.get_policy_snapshot()["store_skipped_requests"] == 0
    second = request(owner, "next")
    assert owner.prepare_store([b"c"], second) is not None


def test_noop_host_progress_is_allowed_even_for_a_skipped_request(environment):
    owner, delegate, _ = manager(
        environment, capacity=2, working_set_aware=True, working_set_max_new_bytes=10
    )
    fill(delegate.primary_tier, [b"cached0", b"cached1"])
    context = request(owner)
    assert owner.prepare_store([b"new0", b"new1", b"new2"], context) is None
    assert owner.prepare_store([b"cached0"], context).keys_to_store == []


def test_chunked_request_uses_cumulative_submitted_bytes(environment):
    owner, delegate, _ = manager(
        environment, capacity=2, working_set_aware=True, working_set_max_new_bytes=10
    )
    context = request(owner)
    assert owner.prepare_store([b"a"], context) is not None
    owner.complete_store([b"a"], context)
    fill(delegate.primary_tier, [b"other"])
    owner.lookup(b"other", context)
    assert owner.prepare_store([b"b"], context) is None
    assert owner.get_policy_snapshot()["decisions"]["working_set_capacity"] == 1


def test_lookup_and_load_submission_are_not_reported_as_completed_restore(environment):
    owner, delegate, _ = manager(environment)
    fill(delegate.primary_tier, [b"cached"])
    context = request(owner)
    assert owner.lookup(b"cached", context) == Lookup.HIT
    owner.prepare_load([b"cached"], context)
    before = owner.get_policy_snapshot()
    assert before["pipeline_blocks"]["lookup_hit_blocks"] == 1
    assert before["pipeline_blocks"]["load_submitted_blocks"] == 1
    assert before["pipeline_blocks"].get("load_completed_blocks", 0) == 0
    assert not any(stage.startswith("lookup_") for stage in before["pipeline_bytes"])
    owner.complete_load([b"cached"], context)
    assert owner.get_policy_snapshot()["pipeline_bytes"]["load_completed_blocks"] == 10


def test_lookups_have_separate_miss_and_metadata_rejection_outcomes(environment):
    owner, _, _ = manager(environment)
    context = request(owner)
    assert owner.lookup(b"absent", context) == Lookup.MISS
    owner._delete_queue.add(b"deleting")
    assert owner.lookup(b"deleting", context) == Lookup.MISS
    stages = owner.get_policy_snapshot()["pipeline_blocks"]
    assert stages["lookup_attempt_blocks"] == 2
    assert stages["lookup_miss_blocks"] == 1
    assert stages["lookup_rejected_blocks"] == 1


@pytest.mark.parametrize("geometry", [0, -1, True, 1.5])
def test_invalid_payload_geometry_fails_before_admission(environment, geometry):
    delegate, clock = Delegate(2), Clock()
    lifecycle = environment.lifecycle.SessionLifecycleManager(
        environment.lifecycle.LifecycleConfig(residency_tracking_enabled=True),
        clock=clock,
    )
    with pytest.raises(ValueError, match="bytes_per_block"):
        environment.lifecycle_manager.LifecycleAwareManager(
            delegate, lifecycle, bytes_per_block=geometry, clock=clock
        )


def test_failed_store_is_not_counted_as_success_and_finalization_waits(environment):
    owner, _, _ = manager(environment, working_set_aware=True)
    context = request(owner)
    owner.prepare_store([b"a"], context)
    owner.on_request_finished(context)
    assert owner._request_store_blocks[context.req_id] == 1
    owner.complete_store([b"a"], context, success=False)
    snapshot = owner.get_policy_snapshot()
    assert snapshot["pipeline_blocks"]["store_failed_blocks"] == 1
    assert snapshot["pipeline_blocks"].get("store_completed_blocks", 0) == 0
    assert context.req_id not in owner._request_store_blocks
    with pytest.raises(RuntimeError, match="untracked"):
        owner.complete_load([b"a"], context)
    assert snapshot["pipeline_blocks"].get("load_completed_blocks", 0) == 0


def test_request_arrivals_do_not_force_full_cache_scans(environment):
    owner, _, clock = manager(environment)
    calls = []
    owner._reconcile_cpu = lambda: calls.append(True)
    for index in range(20):
        request(owner, str(index))
    assert calls == []
    clock.advance(owner.policy.maintenance_interval_sec)
    request(owner, "after-interval")
    assert calls == [True]


def test_stage_metrics_are_deltas_and_have_registered_definitions(environment):
    owner, _, _ = manager(environment)
    context = request(owner)
    owner.prepare_store([b"a"], context)
    owner.complete_store([b"a"], context)
    module = environment.stats_module
    with patch.dict(sys.modules, {module.__name__: module}):
        first, second = owner.get_stats(), owner.get_stats()
    metric = environment.residency.TieringMetrics.PIPELINE_BLOCKS
    assert first.counters[(metric, ("store_completed_blocks",))] == 1
    assert second.counters[(metric, ("store_completed_blocks",))] == 0
    definitions = environment.residency.build_tiering_metric_definitions()
    assert metric in definitions
    assert environment.residency.TieringMetrics.WORKING_SET_CENSORED in definitions


def test_fixture_does_not_leak_fake_host_modules(environment):
    assert (
        getattr(sys.modules.get("vllm.v1.kv_offload.base"), "LookupResult", None)
        is not Lookup
    )
