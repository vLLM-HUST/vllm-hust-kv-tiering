# SPDX-License-Identifier: Apache-2.0
"""Pure-Python, deterministic admission policy (no accelerator dependencies)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def boolean(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")  # noqa: TRY004 - uniform configuration errors
    return value


def positive_int(config: dict[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite(config: dict[str, Any], name: str, default: float) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004 - uniform configuration errors
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class TieringPolicyConfig:
    pressure_aware: bool = True
    hbm_high: float = 0.8
    hbm_low: float = 0.6
    secondary_pressure_aware: bool = True
    max_store_blocks_per_step: int = 128
    max_inflight_store_jobs: int = 8
    max_spill_jobs: int = 4
    min_session_requests: int = 1
    min_reuse_probability: float = 0.0
    prefill_tokens_per_second: float = 0.0
    restore_bytes_per_second: float = 0.0
    store_bytes_per_second: float = 0.0
    benefit_safety_factor: float = 1.2
    maintenance_interval_sec: float = 0.05
    retry_backoff_sec: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.hbm_low <= self.hbm_high <= 1:
            raise ValueError("HBM watermarks must satisfy 0 < low <= high <= 1")
        if not 0 <= self.min_reuse_probability <= 1:
            raise ValueError("tiering_min_reuse_probability must be in [0, 1]")
        rates = (
            self.prefill_tokens_per_second,
            self.restore_bytes_per_second,
            self.store_bytes_per_second,
        )
        if any(rate < 0 or not math.isfinite(rate) for rate in rates):
            raise ValueError(
                "calibrated throughput rates must be finite and non-negative"
            )
        if any(rates) and not all(rates):
            raise ValueError(
                "cost admission requires all three calibrated throughput rates"
            )
        if (
            not math.isfinite(self.benefit_safety_factor)
            or self.benefit_safety_factor < 1
        ):
            raise ValueError("tiering_benefit_safety_factor must be finite and >= 1")
        for value in (self.maintenance_interval_sec, self.retry_backoff_sec):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "maintenance interval and retry backoff must be positive"
                )
        for value in (
            self.max_store_blocks_per_step,
            self.max_inflight_store_jobs,
            self.max_spill_jobs,
            self.min_session_requests,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("policy limits must be positive integers")

    @classmethod
    def from_extra_config(cls, extra: dict[str, Any]) -> TieringPolicyConfig:
        known = {
            "tiering_hbm_pressure_aware",
            "tiering_hbm_high_watermark",
            "tiering_hbm_low_watermark",
            "tiering_secondary_pressure_aware",
            "tiering_max_device_store_blocks_per_step",
            "tiering_max_inflight_device_store_jobs",
            "tiering_max_spill_jobs",
            "tiering_min_session_requests_for_offload",
            "tiering_min_reuse_probability",
            "tiering_prefill_tokens_per_second",
            "tiering_restore_bandwidth_bytes_per_second",
            "tiering_store_bandwidth_bytes_per_second",
            "tiering_benefit_safety_factor",
            "tiering_maintenance_interval_sec",
            "tiering_retry_backoff_sec",
        }
        # Previously these keys were silently ignored. Reject, do not pretend
        # to implement GPU eviction or phase-specific scheduler rewrites.
        unknown = sorted(
            key for key in extra if key.startswith("tiering_") and key not in known
        )
        if unknown:
            raise ValueError(
                f"unsupported tiering option(s): {', '.join(unknown)}; use documented policy options and host offload_prompt_only"
            )
        return cls(
            pressure_aware=boolean(extra, "tiering_hbm_pressure_aware", True),
            hbm_high=finite(extra, "tiering_hbm_high_watermark", 0.8),
            hbm_low=finite(extra, "tiering_hbm_low_watermark", 0.6),
            secondary_pressure_aware=boolean(
                extra, "tiering_secondary_pressure_aware", True
            ),
            max_store_blocks_per_step=positive_int(
                extra, "tiering_max_device_store_blocks_per_step", 128
            ),
            max_inflight_store_jobs=positive_int(
                extra, "tiering_max_inflight_device_store_jobs", 8
            ),
            max_spill_jobs=positive_int(extra, "tiering_max_spill_jobs", 4),
            min_session_requests=positive_int(
                extra, "tiering_min_session_requests_for_offload", 1
            ),
            min_reuse_probability=finite(extra, "tiering_min_reuse_probability", 0.0),
            prefill_tokens_per_second=finite(
                extra, "tiering_prefill_tokens_per_second", 0.0
            ),
            restore_bytes_per_second=finite(
                extra, "tiering_restore_bandwidth_bytes_per_second", 0.0
            ),
            store_bytes_per_second=finite(
                extra, "tiering_store_bandwidth_bytes_per_second", 0.0
            ),
            benefit_safety_factor=finite(extra, "tiering_benefit_safety_factor", 1.2),
            maintenance_interval_sec=finite(
                extra, "tiering_maintenance_interval_sec", 0.05
            ),
            retry_backoff_sec=finite(extra, "tiering_retry_backoff_sec", 1.0),
        )


class PressureGate:
    """A hysteresis latch; missing telemetry is not treated as low pressure."""

    def __init__(self, high: float, low: float):
        if not 0 < low <= high <= 1:
            raise ValueError("pressure watermarks must satisfy 0 < low <= high <= 1")
        self.high, self.low = high, low
        self.active = False

    def update(self, usage: float, *, preempted: bool = False) -> bool:
        if (
            not isinstance(usage, (int, float))
            or not math.isfinite(usage)
            or not 0 <= usage <= 1
        ):
            raise ValueError("cache pressure must be a finite fraction in [0, 1]")
        if preempted or usage >= self.high:
            self.active = True
        elif usage <= self.low:
            self.active = False
        return self.active


def admission_reason(
    config: TieringPolicyConfig,
    *,
    requests: int,
    reuse_probability: float,
    num_blocks: int,
    bytes_per_block: int,
    tokens_per_block: int,
) -> str | None:
    """None permits admission. Rates are user-calibrated estimates, not facts."""
    if requests < config.min_session_requests:
        return "session_history"
    if reuse_probability < config.min_reuse_probability:
        return "reuse_probability"
    if config.prefill_tokens_per_second:
        recompute = num_blocks * tokens_per_block / config.prefill_tokens_per_second
        restore = num_blocks * bytes_per_block / config.restore_bytes_per_second
        write = num_blocks * bytes_per_block / config.store_bytes_per_second
        if (
            reuse_probability * (recompute - restore)
            <= config.benefit_safety_factor * write
        ):
            return "estimated_cost"
    return None
