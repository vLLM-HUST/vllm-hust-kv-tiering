# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-level residency and ownership metadata for KV tiering."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
)


class TieringMetrics:
    RESIDENT_BLOCKS = "vllm:kv_tiering_resident_blocks"
    SESSION_STATES = "vllm:kv_tiering_sessions"
    TRACKED_BLOCKS = "vllm:kv_tiering_tracked_blocks"
    SHARED_BLOCKS = "vllm:kv_tiering_shared_blocks"
    ACTIVE_BLOCKS = "vllm:kv_tiering_active_blocks"
    LOOKUPS = "vllm:kv_tiering_lookups"
    MIGRATION_BLOCKS = "vllm:kv_tiering_migration_blocks"
    MIGRATION_BYTES = "vllm:kv_tiering_migration_bytes"
    MIGRATION_LATENCY = "vllm:kv_tiering_migration_latency_seconds"
    EXPIRED_SESSIONS = "vllm:kv_tiering_expired_sessions"
    PRUNED_SESSIONS = "vllm:kv_tiering_pruned_sessions"
    DELETED_BLOCKS = "vllm:kv_tiering_deleted_blocks"
    PRIMARY_RECLAIMED_BLOCKS = "vllm:kv_tiering_primary_reclaimed_blocks"
    PRUNED_TRACKING_ENTRIES = "vllm:kv_tiering_pruned_tracking_entries"
    DEVICE_CACHE_USAGE = "vllm:kv_tiering_device_cache_usage"
    PRESSURE_STATE = "vllm:kv_tiering_pressure_state"
    STORE_DECISIONS = "vllm:kv_tiering_store_decisions"
    MIGRATION_BUDGET = "vllm:kv_tiering_migration_budget_blocks"
    REUSE_SIGNALS = "vllm:kv_tiering_reuse_signals"
    TRANSFER_SUBMIT_TO_COMPLETE = "vllm:kv_tiering_transfer_submit_to_complete_seconds"
    TRANSFER_OPS = "vllm:kv_tiering_transfer_ops"
    WAIT_SECONDS = "vllm:kv_tiering_worker_wait_seconds"
    H2D_ADMISSION = "vllm:kv_tiering_h2d_submit_to_admitted_seconds"


def build_tiering_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        TieringMetrics.RESIDENT_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of KV blocks observed in each offload tier.",
            labelnames=("tier",),
        ),
        TieringMetrics.SESSION_STATES: OffloadingGaugeMetadata(
            documentation="Number of lifecycle sessions in each state.",
            labelnames=("state",),
        ),
        TieringMetrics.TRACKED_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of block residency entries tracked by tiering.",
        ),
        TieringMetrics.SHARED_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of KV blocks referenced by multiple sessions.",
        ),
        TieringMetrics.ACTIVE_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of tracked KV blocks referenced by active sessions.",
        ),
        TieringMetrics.LOOKUPS: OffloadingCounterMetadata(
            documentation="Tiering lookup outcomes grouped by tier and result.",
            labelnames=("tier", "result"),
        ),
        TieringMetrics.MIGRATION_BLOCKS: OffloadingCounterMetadata(
            documentation="Number of KV blocks migrated between storage tiers.",
            labelnames=("source", "target"),
        ),
        TieringMetrics.MIGRATION_BYTES: OffloadingCounterMetadata(
            documentation="Bytes of KV data migrated between storage tiers.",
            labelnames=("source", "target"),
        ),
        TieringMetrics.MIGRATION_LATENCY: OffloadingHistogramMetadata(
            documentation="KV migration latency between storage tiers in seconds.",
            labelnames=("source", "target"),
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        ),
        TieringMetrics.EXPIRED_SESSIONS: OffloadingCounterMetadata(
            documentation="Number of idle KV lifecycle sessions that expired.",
        ),
        TieringMetrics.PRUNED_SESSIONS: OffloadingCounterMetadata(
            documentation=(
                "Number of oldest idle lifecycle sessions removed to bound metadata."
            ),
        ),
        TieringMetrics.DELETED_BLOCKS: OffloadingCounterMetadata(
            documentation="Number of expired KV block copies deleted from a tier.",
            labelnames=("tier",),
        ),
        TieringMetrics.PRIMARY_RECLAIMED_BLOCKS: OffloadingCounterMetadata(
            documentation=(
                "Number of idle CPU-primary blocks reclaimed after a secondary "
                "copy became durable."
            ),
            labelnames=("reason",),
        ),
        TieringMetrics.PRUNED_TRACKING_ENTRIES: OffloadingCounterMetadata(
            documentation="Number of stale residency metadata entries pruned.",
        ),
        TieringMetrics.DEVICE_CACHE_USAGE: OffloadingGaugeMetadata(
            documentation=(
                "Fraction of scheduler-managed device KV cache blocks in use."
            ),
        ),
        TieringMetrics.PRESSURE_STATE: OffloadingGaugeMetadata(
            documentation="Whether a tier is currently above its pressure threshold.",
            labelnames=("tier",),
        ),
        TieringMetrics.STORE_DECISIONS: OffloadingCounterMetadata(
            documentation="Pressure-policy decisions for newly computed KV blocks.",
            labelnames=("decision", "reason"),
        ),
        TieringMetrics.MIGRATION_BUDGET: OffloadingGaugeMetadata(
            documentation="Current device-to-CPU migration budget consumption.",
            labelnames=("scope",),
        ),
        TieringMetrics.REUSE_SIGNALS: OffloadingGaugeMetadata(
            documentation="Current lifecycle heat and external reuse observations.",
            labelnames=("signal",),
        ),
        TieringMetrics.TRANSFER_SUBMIT_TO_COMPLETE: OffloadingHistogramMetadata(
            documentation=(
                "Wall-clock time from asynchronous KV transfer submission to "
                "completion observation."
            ),
            labelnames=("direction", "phase", "path"),
            buckets=(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        ),
        TieringMetrics.TRANSFER_OPS: OffloadingCounterMetadata(
            documentation="Number of DMA/D2D operations used by KV transfers.",
            labelnames=("direction", "phase", "path", "op"),
        ),
        TieringMetrics.WAIT_SECONDS: OffloadingHistogramMetadata(
            documentation="Time spent blocked in worker.wait() fences.",
            buckets=(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        ),
        TieringMetrics.H2D_ADMISSION: OffloadingHistogramMetadata(
            documentation="Time from H2D submission until request admission.",
            buckets=(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        ),
    }


@dataclass(slots=True)
class BlockResidencyState:
    key: OffloadKey
    cpu_resident: bool = False
    secondary_tiers: set[str] = field(default_factory=set)
    session_ids: set[str] = field(default_factory=set)
    inflight_transfers: Counter[tuple[str, str]] = field(default_factory=Counter)
    created_at: float = field(default_factory=time.monotonic)
    last_access_at: float = field(default_factory=time.monotonic)
    idle_since: float | None = None
    access_count: int = 0

    @property
    def is_shared(self) -> bool:
        return len(self.session_ids) > 1

    @property
    def has_secondary_copy(self) -> bool:
        return bool(self.secondary_tiers)

    @property
    def has_inflight_transfer(self) -> bool:
        return bool(self.inflight_transfers)


class ResidencyCapacityError(RuntimeError):
    """A residency mutation cannot be tracked without losing live ownership."""


class BlockResidencyTracker:
    """Track residency, transfer state, and Session-to-Block ownership."""

    def __init__(
        self, max_entries: int = 64_000, clock: Callable[[], float] = time.monotonic
    ):
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("residency_max_entries must be a positive integer")
        self.max_entries = max_entries
        self._clock = clock
        self._blocks: dict[OffloadKey, BlockResidencyState] = {}
        self._session_keys: dict[str, set[OffloadKey]] = {}
        self._active_sessions: set[str] = set()

    def __len__(self) -> int:
        """Return metadata entry count without scanning residency state."""
        return len(self._blocks)

    def contains_entry(self, key: OffloadKey) -> bool:
        """Test metadata presence, including entries without a resident copy."""
        return key in self._blocks

    def tracked_keys(self) -> set[OffloadKey]:
        """Return a detached key snapshot for maintenance reconciliation."""
        return set(self._blocks)

    def on_session_active(self, session_id: str) -> None:
        self._active_sessions.add(session_id)
        now = self._clock()
        for key in self._session_keys.get(session_id, ()):
            state = self._blocks.get(key)
            if state is not None:
                state.idle_since = None
                state.last_access_at = now

    def on_session_idle(self, session_id: str) -> None:
        self._active_sessions.discard(session_id)
        now = self._clock()
        for key in self._session_keys.get(session_id, ()):
            state = self._blocks.get(key)
            if (
                state is not None
                and not self._has_active_session(state)
                and state.idle_since is None
            ):
                state.idle_since = now

    def record_access(self, session_id: str, keys: Iterable[OffloadKey]) -> bool:
        key_list = list(dict.fromkeys(keys))
        if not self.ensure_capacity(key_list):
            return False
        now = self._clock()
        session_keys = self._session_keys.setdefault(session_id, set())
        for key in key_list:
            state = self._get_or_create(key, now)
            state.session_ids.add(session_id)
            state.last_access_at = now
            state.access_count += 1
            if self._has_active_session(state):
                state.idle_since = None
            elif state.idle_since is None:
                state.idle_since = now
            session_keys.add(key)
        return True

    def release_session(self, session_id: str) -> set[OffloadKey]:
        self._active_sessions.discard(session_id)
        keys = self._session_keys.pop(session_id, set())
        unreferenced: set[OffloadKey] = set()
        now = self._clock()
        for key in keys:
            state = self._blocks.get(key)
            if state is None:
                continue
            state.session_ids.discard(session_id)
            if not state.session_ids:
                state.idle_since = now if state.idle_since is None else state.idle_since
                unreferenced.add(key)
            elif not self._has_active_session(state):
                state.idle_since = now if state.idle_since is None else state.idle_since
        return unreferenced

    def mark_cpu_resident(
        self, keys: Iterable[OffloadKey], resident: bool = True
    ) -> None:
        key_list = list(dict.fromkeys(keys))
        if resident:
            self._require_capacity(key_list)
        now = self._clock()
        for key in key_list:
            state = self._get_or_create(key, now) if resident else self._blocks.get(key)
            if state is not None:
                state.cpu_resident = resident

    def mark_secondary_resident(
        self, keys: Iterable[OffloadKey], tier: str, resident: bool = True
    ) -> None:
        key_list = list(dict.fromkeys(keys))
        if resident:
            self._require_capacity(key_list)
        now = self._clock()
        for key in key_list:
            state = self._get_or_create(key, now) if resident else self._blocks.get(key)
            if state is None:
                continue
            if resident:
                state.secondary_tiers.add(tier)
            else:
                state.secondary_tiers.discard(tier)

    def start_transfer(
        self, keys: Iterable[OffloadKey], source: str, target: str
    ) -> None:
        key_list = list(dict.fromkeys(keys))
        self._require_capacity(key_list)
        now = self._clock()
        transfer = (source, target)
        for key in key_list:
            state = self._get_or_create(key, now)
            state.inflight_transfers[transfer] += 1

    def finish_transfer(
        self, keys: Iterable[OffloadKey], source: str, target: str
    ) -> None:
        transfer = (source, target)
        for key in dict.fromkeys(keys):
            state = self._blocks.get(key)
            if state is None:
                continue
            count = state.inflight_transfers.get(transfer, 0)
            if count <= 1:
                state.inflight_transfers.pop(transfer, None)
            else:
                state.inflight_transfers[transfer] = count - 1

    def get_idle_cpu_candidates(
        self,
        *,
        limit: int,
        idle_before: float | None = None,
    ) -> list[OffloadKey]:
        if limit <= 0:
            return []
        candidates = [
            state
            for state in self._blocks.values()
            if state.cpu_resident
            and state.has_secondary_copy
            and not state.has_inflight_transfer
            and not self._has_active_session(state)
            and not state.is_shared
            and state.idle_since is not None
            and (idle_before is None or state.idle_since <= idle_before)
        ]
        candidates.sort(
            key=lambda state: (state.idle_since or 0.0, state.last_access_at)
        )
        return [state.key for state in candidates[:limit]]

    def get_cpu_pressure_candidates(
        self,
        *,
        limit: int,
        idle_before: float | None = None,
        include_active: bool = True,
    ) -> list[OffloadKey]:
        """Return CPU-resident blocks ordered from coldest to hottest.

        Idle, unshared, low-access blocks are preferred. Active-session blocks
        are considered only as a last resort when CPU pressure cannot be
        relieved using idle state alone.
        """
        if limit <= 0:
            return []
        candidates = [
            state
            for state in self._blocks.values()
            if state.cpu_resident
            and not state.has_inflight_transfer
            and (
                idle_before is None
                or (state.idle_since is not None and state.idle_since <= idle_before)
            )
            and (include_active or not self._has_active_session(state))
        ]
        candidates.sort(
            key=lambda state: (
                self._has_active_session(state),
                state.is_shared,
                state.access_count,
                state.idle_since is None,
                state.last_access_at if state.idle_since is None else state.idle_since,
                state.last_access_at,
            )
        )
        return [state.key for state in candidates[:limit]]

    def has_secondary_copy(self, key: OffloadKey) -> bool:
        state = self._blocks.get(key)
        return state is not None and state.has_secondary_copy

    def is_known(self, key: OffloadKey) -> bool:
        state = self._blocks.get(key)
        return state is not None and (
            state.cpu_resident
            or state.has_secondary_copy
            or state.has_inflight_transfer
        )

    def has_idle_cpu_blocks(self) -> bool:
        return any(
            state.cpu_resident
            and state.has_secondary_copy
            and not state.has_inflight_transfer
            and not self._has_active_session(state)
            and not state.is_shared
            and state.idle_since is not None
            for state in self._blocks.values()
        )

    def get_inflight_keys(self) -> set[OffloadKey]:
        return {
            state.key for state in self._blocks.values() if state.has_inflight_transfer
        }

    def clear_cpu_residency(self) -> None:
        for state in self._blocks.values():
            state.cpu_resident = False

    def get_session_keys(self, session_id: str) -> set[OffloadKey]:
        return set(self._session_keys.get(session_id, ()))

    def snapshot(self) -> dict[str, object]:
        secondary_counts: Counter[str] = Counter()
        cpu_blocks = shared_blocks = active_blocks = 0
        for state in self._blocks.values():
            cpu_blocks += int(state.cpu_resident)
            shared_blocks += int(state.is_shared)
            active_blocks += int(self._has_active_session(state))
            secondary_counts.update(state.secondary_tiers)
        return {
            "tracked_blocks": len(self._blocks),
            "cpu_blocks": cpu_blocks,
            "secondary_blocks": dict(secondary_counts),
            "shared_blocks": shared_blocks,
            "active_blocks": active_blocks,
        }

    def _prunable(self, protected: set[OffloadKey]) -> list[BlockResidencyState]:
        return sorted(
            (
                state
                for state in self._blocks.values()
                if state.key not in protected
                and not state.session_ids
                and not state.cpu_resident
                and not state.has_inflight_transfer
            ),
            key=lambda state: state.last_access_at,
        )

    def ensure_capacity(self, keys: Iterable[OffloadKey]) -> bool:
        """Reserve room atomically without forgetting active ownership.

        Secondary-only, unowned hints can be forgotten; this is metadata
        pruning, never evidence authorizing deletion of their actual files.
        The caller must serialize ensure + mutation (all APIs are host-thread
        confined), and supplied existing keys are protected from pruning.
        """
        requested = set(keys)
        missing = requested.difference(self._blocks)
        excess = len(self._blocks) + len(missing) - self.max_entries
        if excess <= 0:
            return True
        candidates = self._prunable(requested)
        if len(candidates) < excess:
            return False
        for state in candidates[:excess]:
            del self._blocks[state.key]
        return True

    def _require_capacity(self, keys: Iterable[OffloadKey]) -> None:
        if not self.ensure_capacity(keys):
            raise ResidencyCapacityError("residency metadata capacity exhausted")

    def prune(self) -> int:
        excess = max(0, len(self._blocks) - self.max_entries)
        candidates = self._prunable(set())[:excess]
        for state in candidates:
            del self._blocks[state.key]
        return len(candidates)

    def _get_or_create(self, key: OffloadKey, now: float) -> BlockResidencyState:
        state = self._blocks.get(key)
        if state is None:
            self._require_capacity((key,))
            state = BlockResidencyState(
                key=key,
                created_at=now,
                last_access_at=now,
            )
            self._blocks[key] = state
        return state

    def _has_active_session(self, state: BlockResidencyState) -> bool:
        return bool(state.session_ids.intersection(self._active_sessions))
