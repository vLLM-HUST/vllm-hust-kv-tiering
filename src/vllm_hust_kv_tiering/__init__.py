"""Out-of-tree KV tiering backend for the vLLM-HUST v1 release line."""

from __future__ import annotations

from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

from vllm_hust_kv_tiering.fs.manager import SegmentFileSystemTier
from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec


def register() -> None:
    """Register the optional short tier name; safe in every vLLM process."""
    if "hust_fs" not in SecondaryTierFactory._registry:
        SecondaryTierFactory.register_tier(
            "hust_fs",
            "vllm_hust_kv_tiering.fs.manager",
            "SegmentFileSystemTier",
        )


__all__ = ["HustTieringOffloadingSpec", "SegmentFileSystemTier", "register"]
