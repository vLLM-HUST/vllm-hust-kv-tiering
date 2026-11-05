# SPDX-License-Identifier: Apache-2.0
"""Explicit compatibility boundary for the frozen HUST CPU-primary ABI.

The host exposes read/write pinning publicly, but no selective eviction method.
Only this adapter accesses its private allocation/policy bookkeeping. Unsupported
hosts fail on activation instead of silently bypassing pinning. No host patch or
replacement of LRU/ARC is performed.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from vllm.v1.kv_offload.base import OffloadingEvent, OffloadKey


class HBMPressureSource:
    """O(1) conservative cache pressure, including evictable prefix residency.

    get_usage counts allocated/non-evictable blocks, not cached free blocks.
    The max of allocation fraction and physical cached-block fraction is a
    lower bound on their union; never sum them and double-count active prefixes.
    This is scheduler cache pressure, not total device memory utilization.
    """

    def __init__(self, pool: Any):
        if not callable(getattr(pool, "get_usage", None)):
            raise TypeError("incompatible block pool: missing get_usage()")
        if (
            not isinstance(getattr(pool, "num_gpu_blocks", None), int)
            or pool.num_gpu_blocks < 2
        ):
            raise RuntimeError("incompatible block pool: invalid num_gpu_blocks")
        if not isinstance(getattr(pool, "cached_block_hashes_by_block", None), dict):
            raise TypeError(
                "incompatible block pool: missing physical prefix-cache index"
            )
        self.pool = pool

    def __call__(self) -> float:
        allocated = self.pool.get_usage()
        cached = len(self.pool.cached_block_hashes_by_block) / (
            self.pool.num_gpu_blocks - 1
        )
        return max(allocated, cached)


class CPUPrimaryControl:
    def __init__(self, primary: Any):
        required = (
            "_policy",
            "_num_blocks",
            "_num_allocated_blocks",
            "_free_list",
            "_num_evictable_cache_blocks",
            "_free_block",
            "prepare_read",
            "complete_read",
        )
        missing = [name for name in required if not hasattr(primary, name)]
        if missing:
            raise RuntimeError(
                f"incompatible HUST CPU-primary ABI: missing {', '.join(missing)}"
            )
        for name in ("get", "remove"):
            if not callable(getattr(primary._policy, name, None)):
                raise RuntimeError(f"incompatible CPU cache policy: missing {name}")  # noqa: TRY004 - host ABI activation error
        self.primary = primary

    @property
    def capacity(self) -> int:
        return self.primary._num_blocks

    @property
    def free_slots(self) -> int:
        return (
            len(self.primary._free_list)
            + self.capacity
            - self.primary._num_allocated_blocks
        )

    @property
    def occupancy(self) -> float:
        return 1.0 - self.free_slots / self.capacity if self.capacity else 0.0

    def status(self, key: OffloadKey):
        return self.primary._policy.get(key)

    def ready(self, key: OffloadKey) -> bool:
        block = self.status(key)
        return block is not None and block.is_ready

    def removable(self, key: OffloadKey) -> bool:
        block = self.status(key)
        return block is not None and block.is_ready and block.ref_cnt == 0

    def remove(self, keys: Collection[OffloadKey]) -> set[OffloadKey]:
        """Remove only ready, unpinned entries; re-check immediately before reuse."""
        removed = set()
        for key in keys:
            block = self.status(key)
            if block is None or not block.is_ready or block.ref_cnt != 0:
                continue
            self.primary._policy.remove(key)
            self.primary._num_evictable_cache_blocks -= 1
            if self.primary._num_evictable_cache_blocks < 0:
                raise RuntimeError("CPU-primary evictable counter invariant violated")
            self.primary._free_block(block)
            removed.add(key)
        if removed and self.primary.events is not None:
            self.primary.events.append(
                OffloadingEvent(
                    keys=list(removed), medium=self.primary.medium, removed=True
                )
            )
        return removed
