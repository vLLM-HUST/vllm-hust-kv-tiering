# SPDX-License-Identifier: Apache-2.0
"""Policy/lifecycle adapter; the host still owns transfer scheduling and pins."""

from __future__ import annotations

import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult

from vllm_hust_kv_tiering.host_adapter import CPUPrimaryControl
from vllm_hust_kv_tiering.lifecycle import SessionLifecycleManager
from vllm_hust_kv_tiering.policy import (
    PressureGate,
    TieringPolicyConfig,
    admission_reason,
)
from vllm_hust_kv_tiering.residency import TieringMetrics


class _ObservedTier:
    """Observe actual secondary completions, without copying the host manager.

    Host jobs retain their IDs and pins. Negative IDs are plugin maintenance
    jobs: their completion is consumed here and releases a plugin-owned read pin.
    Skipped normal cascades still report completion to release the host's pin;
    they are never counted as a stored secondary copy.
    """

    def __init__(self, delegate: Any, owner: LifecycleAwareManager, index: int):
        self._delegate, self._owner = delegate, owner
        self.tier_type = delegate.tier_type
        self.identity = f"{self.tier_type}:{index}"
        self._jobs: dict[int, tuple[JobMetadata, bool, bool]] = {}
        self._synthetic: deque[JobResult] = deque()
        self._request_finalizations: dict[str, ReqContext] = {}

    def on_request_finished(self, context):
        self._request_finalizations[context.req_id] = context
        self._finish_requests()

    def _finish_requests(self):
        active = {entry[0].req_context.req_id for entry in self._jobs.values()}
        active.update(req_id for req_id, _ in self._owner._deferred_promotions)
        for req_id in list(self._request_finalizations):
            if req_id not in active:
                self._delegate.on_request_finished(
                    self._request_finalizations.pop(req_id)
                )

    def on_schedule_end(self, context):
        self._delegate.on_schedule_end(context)
        self._finish_requests()

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def submit_store(self, metadata: JobMetadata) -> None:
        self._submit(metadata, store=True, manual=False)

    def submit_load(self, metadata: JobMetadata) -> None:
        self._submit(metadata, store=False, manual=False)

    def force_store(self, metadata: JobMetadata) -> None:
        self._submit(metadata, store=True, manual=True)

    def _submit(self, metadata: JobMetadata, *, store: bool, manual: bool):
        if metadata.job_id in self._jobs:
            raise RuntimeError("duplicate secondary transfer job")
        forwarded = not store or manual or self._owner.allow_cascade()
        self._jobs[metadata.job_id] = (metadata, forwarded, manual)
        self._owner._secondary_started(self, metadata, manual)
        if not forwarded:
            self._synthetic.append(JobResult(metadata.job_id, True))
            return
        # An unknown partial enqueue must propagate; never manufacture a
        # completion that authorizes reuse. Keep the job/pin for safe teardown.
        method = self._delegate.submit_store if store else self._delegate.submit_load
        method(metadata)

    def get_finished_jobs(self):
        results = list(self._delegate.get_finished_jobs())
        results.extend(self._synthetic)
        self._synthetic.clear()
        visible = []
        for result in results:
            entry = self._jobs.pop(result.job_id, None)
            if entry is None:
                raise RuntimeError(f"untracked secondary completion {result.job_id}")
            metadata, forwarded, manual = entry
            self._owner._secondary_finished(
                self, metadata, result.success, forwarded, manual
            )
            if manual:
                self._owner._cpu.primary.complete_read(
                    metadata.keys, metadata.req_context
                )
            else:
                visible.append(result)
        self._finish_requests()
        return visible

    def has_pending_work(self):
        return bool(self._jobs) or self._delegate.has_pending_work()

    def drain_jobs(self):
        self._delegate.drain_jobs()


class LifecycleAwareManager:
    """Bounded pressure-aware cache policy around a matched host manager.

    All lifecycle mutations run on the scheduler thread. Only filesystem delete
    work runs in an executor. Inference is never blocked waiting for maintenance
    I/O: admission may decline caching and the host continues/recomputes.
    """

    def __init__(
        self,
        delegate: Any,
        lifecycle: SessionLifecycleManager,
        policy: TieringPolicyConfig | None = None,
        *,
        bytes_per_block: int = 1,
        tokens_per_block: int = 1,
        clock=time.monotonic,
    ):
        if not callable(getattr(delegate, "_process_finished_jobs", None)):
            raise TypeError(
                "incompatible HUST tiering manager: missing completion-poll contract"
            )
        self._delegate, self._lifecycle = delegate, lifecycle
        self.policy = policy or TieringPolicyConfig()
        self._clock = clock
        self._cpu = CPUPrimaryControl(delegate.primary_tier)
        self._hbm_gate = PressureGate(self.policy.hbm_high, self.policy.hbm_low)
        config = lifecycle.config
        self._cpu_gate = PressureGate(
            config.cpu_high_watermark, config.cpu_low_watermark
        )
        self._pressure_provider = None
        self._hbm_usage = 0.0
        self._step_blocks = 0
        self._pending_requests: Counter[str] = Counter()
        self._finished_requests: dict[str, ReqContext] = {}
        self._device_jobs: dict[tuple[str, bool, frozenset], int] = {}
        self._deferred_promotions: set[tuple[str, OffloadKey]] = set()
        self._known_cpu_keys: set[OffloadKey] = set()
        self._step_initialized = False
        self._manual_id = -1
        self._manual_jobs = 0
        self._spill_retry_after: dict[OffloadKey, float] = {}
        self._maintenance_context = ReqContext("__hust_tiering_maintenance__")
        self._delete_queue: set[OffloadKey] = set()
        self._delete_future: Future | None = None
        self._delete_active: set[OffloadKey] = set()
        self._delete_retry_after = 0.0
        self._delete_executor: ThreadPoolExecutor | None = None
        self._next_maintenance = 0.0
        self._closed = False
        self._shutdown_done = False
        self._counts: Counter[str] = Counter()
        self._metric_counts: Counter[str] = Counter()
        self._lifecycle_metric_totals: Counter[str] = Counter()
        self._bytes_per_block, self._tokens_per_block = (
            bytes_per_block,
            tokens_per_block,
        )
        tiers = [
            _ObservedTier(tier, self, i)
            for i, tier in enumerate(delegate.secondary_tiers)
        ]
        if config.delete_expired_secondary and any(
            not callable(getattr(tier, "delete", None)) for tier in tiers
        ):
            raise ValueError(
                "secondary TTL deletion requires every tier to implement delete(keys)"
            )
        delegate.secondary_tiers = tiers
        self.secondary_tiers = tiers

    def _mark_cpu(self, keys, resident=True):
        keys = list(keys)
        self.residency.mark_cpu_resident(keys, resident)
        if resident:
            self._known_cpu_keys.update(keys)
        else:
            self._known_cpu_keys.difference_update(keys)

    def _reconcile_cpu(self):
        absent = [key for key in self._known_cpu_keys if self._cpu.status(key) is None]
        if absent:
            self._mark_cpu(absent, False)

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @property
    def residency(self):
        return self._lifecycle.residency

    def bind_pressure_provider(self, provider) -> None:
        if not callable(provider):
            raise TypeError("pressure provider must be callable")
        self._pressure_provider = provider

    def begin_step(self, *, preempted: bool = False) -> None:
        self._step_blocks = 0
        self._step_initialized = True
        if self.policy.pressure_aware:
            if self._pressure_provider is None:
                raise RuntimeError(
                    "pressure-aware tiering requires host bind_gpu_block_pool/get_usage support"
                )
            self._hbm_usage = self._pressure_provider()
            self._hbm_gate.update(self._hbm_usage, preempted=preempted)
        # Poll unconditionally at this explicit new-step boundary: the host's
        # once-per-step flag can still be set by store construction *after*
        # its previous on_schedule_end hook. Reusing that flag here would
        # leave completed pins held for an extra scheduler step.
        self._delegate._process_finished_jobs()
        self._maintenance()

    def should_build_store_jobs(self) -> bool:
        if not self._step_initialized:
            raise RuntimeError(
                "tiering admission requires HustTieringConnector (scheduler begin_step hook)"
            )
        allowed = not self.policy.pressure_aware or self._hbm_gate.active
        if not allowed:
            self._decision("low_hbm_pressure")
        return allowed

    def allow_cascade(self) -> bool:
        return not self.policy.secondary_pressure_aware or self._cpu_gate.update(
            self._cpu.occupancy
        )

    def _decision(self, reason: str) -> None:
        self._counts[reason] += 1
        self._metric_counts[reason] += 1

    def on_new_request(self, req_context):
        self._maintenance(force=True)
        self._lifecycle.on_new_request(req_context, track_heat=True)
        return self._delegate.on_new_request(req_context)

    def on_request_finished(self, req_context):
        self._lifecycle.on_request_finished(req_context, track_heat=True)
        self._finished_requests[req_context.req_id] = req_context
        self._delegate.on_request_finished(req_context)
        self._maybe_finalize(req_context.req_id)

    def _maybe_finalize(self, req_id):
        if self._pending_requests.get(req_id, 0):
            return
        context = self._finished_requests.pop(req_id, None)
        if context is not None:
            self._lifecycle.on_request_finalized(context)
        self._pending_requests.pop(req_id, None)

    def _owned(self, keys, req_context) -> bool:
        keys = list(keys)
        tombstones = self._delete_queue | self._delete_active
        fresh = sum(not self.residency.contains_entry(key) for key in set(keys))
        if (
            fresh
            and tombstones
            and len(self.residency) + len(tombstones) + fresh
            > self.residency.max_entries
        ):
            self._decision("delete_backlog_capacity")
            self._decision("metadata_capacity")
            return False
        if not self._lifecycle.record_request_keys(req_context, keys):
            self._decision("metadata_capacity")
            return False
        return True

    def lookup(self, key, req_context):
        if key in self._delete_queue or key in self._delete_active:
            return LookupResult.MISS
        if not self._owned([key], req_context):
            return LookupResult.MISS
        prior = self._cpu.status(key)
        if prior is None and self._cpu.free_slots == 0:
            self._reclaim(1, protected={key}, pressure=True)
            if self._cpu.free_slots == 0:
                self._decision("promotion_no_free_slot")
                return LookupResult.MISS
        result = self._delegate.lookup(key, req_context)
        current = self._cpu.status(key)
        if prior is None and current is not None and not current.is_ready:
            token = (req_context.req_id, key)
            if token not in self._deferred_promotions:
                self._deferred_promotions.add(token)
                self._pending_requests[req_context.req_id] += 1
                self.residency.start_transfer([key], "secondary_pending", "cpu")
        if result == LookupResult.HIT:
            self._mark_cpu([key], self._cpu.ready(key))
            self._lifecycle.record_reuse_hit(req_context)
        return result

    def touch(self, keys, req_context):
        keys = [
            key
            for key in keys
            if key not in self._delete_queue and key not in self._delete_active
        ]
        known = [key for key in keys if self.residency.is_known(key)]
        if known:
            self._owned(known, req_context)
        self._delegate.touch(keys, req_context)

    def prepare_store(self, keys, req_context):
        keys = list(keys)
        if self._closed or not self.should_build_store_jobs():
            return None
        if any(key in self._delete_queue or key in self._delete_active for key in keys):
            self._decision("ttl_delete_pending")
            return None
        stores = sum(
            count for (_, store, _), count in self._device_jobs.items() if store
        )
        if stores >= self.policy.max_inflight_store_jobs:
            self._decision("inflight_budget")
            return None
        if self._step_blocks >= self.policy.max_store_blocks_per_step:
            self._decision("step_budget")
            return None
        if not self._owned(keys, req_context):
            return None
        requests, probability = self._lifecycle.get_session_heat(req_context)
        reason = admission_reason(
            self.policy,
            requests=requests,
            reuse_probability=probability,
            num_blocks=len(keys),
            bytes_per_block=self._bytes_per_block,
            tokens_per_block=self._tokens_per_block,
        )
        if reason:
            self._decision(reason)
            return None
        fresh = [key for key in keys if self._cpu.status(key) is None]
        if (
            self._step_blocks
            and self._step_blocks + len(fresh) > self.policy.max_store_blocks_per_step
        ):
            self._decision("step_budget")
            return None
        if len(fresh) > self._cpu.capacity:
            self._decision("request_exceeds_cpu_capacity")
            return None
        if self.secondary_tiers and self._cpu.free_slots < len(fresh):
            self._reclaim(
                len(fresh) - self._cpu.free_slots, protected=set(keys), pressure=True
            )
            if self._cpu.free_slots < len(fresh):
                self._decision("waiting_for_safe_spill")
                return None
        output = self._delegate.prepare_store(keys, req_context)
        if output is None:
            self._decision("primary_busy")
            return None
        self._mark_cpu(output.evicted_keys, False)
        accepted = list(output.keys_to_store)
        if accepted:
            self._start_device(accepted, req_context, store=True)
            self._step_blocks += len(accepted)
            self._decision("admitted")
        return output

    def prepare_load(self, keys, req_context):
        keys = list(keys)
        if not self._owned(keys, req_context):
            raise RuntimeError("load admission lost tracked ownership")
        spec = self._delegate.prepare_load(keys, req_context)
        self._start_device(keys, req_context, store=False)
        return spec

    def _start_device(self, keys, context, *, store):
        token = (context.req_id, store, frozenset(keys))
        self._device_jobs[token] = self._device_jobs.get(token, 0) + 1
        self._pending_requests[context.req_id] += 1
        source, target = ("device", "cpu") if store else ("cpu", "device")
        self.residency.start_transfer(keys, source, target)

    def _finish_device(self, keys, context, *, store):
        token = (context.req_id, store, frozenset(keys))
        count = self._device_jobs.get(token, 0)
        if not count:
            raise RuntimeError("untracked device transfer completion")
        if count == 1:
            del self._device_jobs[token]
        else:
            self._device_jobs[token] = count - 1
        self._pending_requests[context.req_id] -= 1
        source, target = ("device", "cpu") if store else ("cpu", "device")
        self.residency.finish_transfer(keys, source, target)

    def complete_store(self, keys, req_context, success=True):
        keys = list(keys)
        self._delegate.complete_store(keys, req_context, success)
        self._mark_cpu(keys, success)
        self._finish_device(keys, req_context, store=True)
        self._maybe_finalize(req_context.req_id)

    def complete_load(self, keys, req_context):
        keys = list(keys)
        self._delegate.complete_load(keys, req_context)
        self._finish_device(keys, req_context, store=False)
        self._maybe_finalize(req_context.req_id)

    def _secondary_started(self, tier, metadata, manual):
        if metadata.is_promotion:
            for key in metadata.keys:
                token = (metadata.req_context.req_id, key)
                if token in self._deferred_promotions:
                    self._deferred_promotions.remove(token)
                    self._pending_requests[metadata.req_context.req_id] -= 1
                    self.residency.finish_transfer([key], "secondary_pending", "cpu")
        source, target = (
            (tier.identity, "cpu") if metadata.is_promotion else ("cpu", tier.identity)
        )
        self.residency.start_transfer(metadata.keys, source, target)
        if manual:
            self._manual_jobs += 1
        else:
            self._pending_requests[metadata.req_context.req_id] += 1

    def _secondary_finished(self, tier, metadata, success, forwarded, manual):
        # A completed spill may now authorize reclamation; don't postpone it
        # behind the periodic scan interval after releasing its host read pin.
        self._next_maintenance = 0.0
        source, target = (
            (tier.identity, "cpu") if metadata.is_promotion else ("cpu", tier.identity)
        )
        self.residency.finish_transfer(metadata.keys, source, target)
        if forwarded and success:
            self.residency.mark_secondary_resident(metadata.keys, tier.identity)
            if metadata.is_promotion:
                self._mark_cpu(metadata.keys)
            self._decision(
                "secondary_promotion_completed"
                if metadata.is_promotion
                else "secondary_store_completed"
            )
        elif forwarded:
            if metadata.is_promotion:
                self.residency.mark_secondary_resident(
                    metadata.keys, tier.identity, False
                )
            else:
                for key in metadata.keys:
                    self._spill_retry_after[key] = (
                        self._clock() + self.policy.retry_backoff_sec
                    )
            self._decision("secondary_failure")
        if manual:
            self._manual_jobs -= 1
        else:
            self._pending_requests[metadata.req_context.req_id] -= 1
            self._maybe_finalize(metadata.req_context.req_id)

    def _reclaim(self, count: int, *, protected: set, pressure: bool) -> None:
        if count <= 0:
            return
        config = self._lifecycle.config
        idle_before = None if pressure else self._clock() - config.cpu_demote_after_sec
        candidates = self.residency.get_cpu_pressure_candidates(
            limit=config.reclaim_batch_size,
            idle_before=idle_before,
            include_active=pressure,
        )
        candidates = [
            key
            for key in candidates
            if key not in protected and self._cpu.removable(key)
        ]
        removable = [
            key for key in candidates if self.residency.has_secondary_copy(key)
        ]
        removed = self._cpu.remove(removable[:count])
        if removed:
            self._mark_cpu(removed, False)
            self._counts["cpu_reclaimed"] += len(removed)
        remaining = count - len(removed)
        if (
            remaining <= 0
            or not self.secondary_tiers
            or self._manual_jobs >= self.policy.max_spill_jobs
        ):
            return
        now = self._clock()
        cold = [
            key
            for key in candidates
            if key not in removed
            and not self.residency.has_secondary_copy(key)
            and self._spill_retry_after.get(key, 0) <= now
        ][:remaining]
        if not cold:
            return
        # One known backend is enough for a recoverable CPU cache copy. A
        # maintenance spill is explicit and bypasses the normal cascade gate.
        tier = self.secondary_tiers[0]
        spec = self._cpu.primary.prepare_read(cold, self._maintenance_context)
        metadata = JobMetadata(
            self._manual_id, cold, spec.block_ids, False, self._maintenance_context
        )
        self._manual_id -= 1
        tier.force_store(metadata)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._delegate.on_schedule_end(context)
        self._maintenance()

    def _maintenance(self, *, force=False):
        now = self._clock()
        self._poll_delete()
        if not force and now < self._next_maintenance:
            return
        self._next_maintenance = now + self.policy.maintenance_interval_sec
        self._reconcile_cpu()
        protected = self.residency.get_inflight_keys()
        expired = self._lifecycle.expire_idle_sessions(
            protected_keys=protected, protected_req_ids=set(self._pending_requests)
        )
        self._counts["expired_sessions"] += expired.expired_sessions
        if expired.unreferenced_keys:
            removed = self._cpu.remove(expired.unreferenced_keys)
            self._mark_cpu(removed, False)
            if self._lifecycle.config.delete_expired_secondary:
                fresh = (
                    expired.unreferenced_keys - self._delete_queue - self._delete_active
                )
                room = (
                    self.residency.max_entries
                    - len(self._delete_queue)
                    - len(self._delete_active)
                )
                accepted = set(list(fresh)[: max(0, room)])
                self._delete_queue.update(accepted)
                if len(accepted) < len(fresh):
                    # TTL is best-effort cache retention, never a secure-erasure
                    # SLA. Keep inference safe and memory bounded under failure.
                    self._decision("delete_backlog_overflow")
        self._submit_delete()
        pressure = self._cpu_gate.update(self._cpu.occupancy)
        config = self._lifecycle.config
        if pressure:
            target_free = int(
                self._cpu.capacity * (1 - config.cpu_low_watermark) + 0.999999
            )
            self._reclaim(
                max(0, target_free - self._cpu.free_slots),
                protected=protected,
                pressure=True,
            )
        elif config.cpu_demote_after_sec > 0:
            self._reclaim(
                config.reclaim_batch_size, protected=protected, pressure=False
            )
        self.residency.prune()
        for key in list(self._spill_retry_after):
            if self._spill_retry_after[key] <= now:
                del self._spill_retry_after[key]

    def _submit_delete(self):
        if (
            self._delete_future is not None
            or not self._delete_queue
            or self._clock() < self._delete_retry_after
        ):
            return
        protected = self.residency.get_inflight_keys()
        batch = {key for key in self._delete_queue if key not in protected}
        batch = set(list(batch)[: self._lifecycle.config.reclaim_batch_size])
        if not batch:
            return
        self._delete_queue.difference_update(batch)
        self._delete_active = batch
        if self._delete_executor is None:
            self._delete_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hust-kv-expiry"
            )
        tiers = list(self.secondary_tiers)

        def delete_batch():
            outcomes = []
            for tier in tiers:
                delete = getattr(tier, "delete", None)
                if delete is None:
                    outcomes.append((tier.identity, set()))
                else:
                    result = delete(batch)
                    outcomes.append((tier.identity, set(result.removed_keys)))
            return outcomes

        self._delete_future = self._delete_executor.submit(delete_batch)

    def _poll_delete(self):
        if self._delete_future is None or not self._delete_future.done():
            return
        batch = self._delete_active
        try:
            outcomes = self._delete_future.result()
        except Exception:  # noqa: BLE001 - cache maintenance retries, never inference data
            self._decision("delete_failure")
            self._delete_queue.update(batch)
            self._delete_retry_after = self._clock() + self.policy.retry_backoff_sec
        else:
            confirmed = set(batch)
            for identity, removed in outcomes:
                self.residency.mark_secondary_resident(removed, identity, False)
                confirmed.intersection_update(removed)
            self._counts["deleted_keys"] += len(confirmed)
            if remaining := batch - confirmed:
                self._delete_queue.update(remaining)
                self._delete_retry_after = self._clock() + self.policy.retry_backoff_sec
                self._decision("delete_partial_failure")
        self._delete_active = set()
        self._delete_future = None

    def has_pending_work(self):
        # Future TTL/retry deadlines do not keep an idle engine spinning.
        return (
            self._delegate.has_pending_work()
            or self._delete_future is not None
            or bool(self._delete_queue and self._clock() >= self._delete_retry_after)
        )

    def reset_cache(self):
        if self._delete_future is not None:
            wait((self._delete_future,))
            self._poll_delete()
        self._delegate.reset_cache()
        for req_id, key in self._deferred_promotions:
            self.residency.finish_transfer([key], "secondary_pending", "cpu")
            self._pending_requests[req_id] -= 1
            self._maybe_finalize(req_id)
        self._deferred_promotions.clear()
        for (req_id, store, keys), count in list(self._device_jobs.items()):
            source, target = ("device", "cpu") if store else ("cpu", "device")
            for _ in range(count):
                self.residency.finish_transfer(keys, source, target)
                self._pending_requests[req_id] -= 1
            self._maybe_finalize(req_id)
        self._device_jobs.clear()
        self.residency.clear_cpu_residency()
        self._known_cpu_keys.clear()
        for tier in self.secondary_tiers:
            tier._finish_requests()
        self._hbm_gate.active = self._cpu_gate.active = False

    def get_policy_snapshot(self):
        return {
            "hbm_pressure": self._hbm_usage,
            "hbm_gate": self._hbm_gate.active,
            "cpu_occupancy": self._cpu.occupancy,
            "cpu_gate": self._cpu_gate.active,
            "step_store_blocks": self._step_blocks,
            "manual_spill_jobs": self._manual_jobs,
            "pending_device_jobs": sum(self._device_jobs.values()),
            "pending_delete_keys": len(self._delete_queue) + len(self._delete_active),
            "decisions": dict(self._counts),
            "sessions": self._lifecycle.snapshot(),
            "residency": self.residency.snapshot(),
        }

    def get_stats(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        stats = self._delegate.get_stats() or OffloadingConnectorStats()
        # Frozen-host transfer series are label-lazy. Seed zero without
        # inventing I/O so the first real restore has a visible baseline.
        for name in ("vllm:kv_offload_load_bytes", "vllm:kv_offload_store_bytes"):
            stats.increase_counter(name, 0)
        stats.set_gauge(TieringMetrics.DEVICE_CACHE_USAGE, self._hbm_usage)
        stats.set_gauge(
            TieringMetrics.PRESSURE_STATE, int(self._hbm_gate.active), ("device",)
        )
        stats.set_gauge(
            TieringMetrics.PRESSURE_STATE, int(self._cpu_gate.active), ("cpu",)
        )
        snapshot = self.residency.snapshot()
        stats.set_gauge(TieringMetrics.TRACKED_BLOCKS, snapshot["tracked_blocks"])
        stats.set_gauge(
            TieringMetrics.RESIDENT_BLOCKS, snapshot["cpu_blocks"], ("cpu",)
        )
        stats.set_gauge(TieringMetrics.SHARED_BLOCKS, snapshot["shared_blocks"])
        stats.set_gauge(TieringMetrics.ACTIVE_BLOCKS, snapshot["active_blocks"])
        sessions = self._lifecycle.snapshot()
        stats.set_gauge(
            TieringMetrics.SESSION_STATES, sessions["active_sessions"], ("active",)
        )
        stats.set_gauge(
            TieringMetrics.SESSION_STATES, sessions["idle_sessions"], ("idle",)
        )
        stats.set_gauge(TieringMetrics.MIGRATION_BUDGET, self._step_blocks, ("step",))
        for key, metric, labels in (
            ("expired_sessions", TieringMetrics.EXPIRED_SESSIONS, ()),
            ("cpu_reclaimed", TieringMetrics.PRIMARY_RECLAIMED_BLOCKS, ("policy",)),
        ):
            total = self._counts[key]
            delta = total - self._lifecycle_metric_totals[key]
            stats.increase_counter(metric, delta, labels)
            self._lifecycle_metric_totals[key] = total
        for reason, count in self._metric_counts.items():
            stats.increase_counter(
                TieringMetrics.STORE_DECISIONS, count, ("observed", reason)
            )
        self._metric_counts.clear()
        return stats

    def shutdown(self):
        if self._shutdown_done:
            return
        self._closed = True
        if self._delete_executor is not None:
            self._delete_executor.shutdown(wait=True)
            self._poll_delete()
        # Submit any previously reserved promotions before draining. If drain
        # fails, leave the manager/resources retryable, not falsely closed.
        self._delegate.on_schedule_end(ScheduleEndContext((), ()))
        for tier in self.secondary_tiers:
            tier.drain_jobs()
        self._delegate.on_schedule_end(ScheduleEndContext((), ()))
        self._delegate.shutdown()
        self._shutdown_done = True
