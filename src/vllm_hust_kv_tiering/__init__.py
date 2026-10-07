"""Out-of-tree KV tiering backend for the vLLM-HUST v1 release line."""

from __future__ import annotations

from typing import Any


def register() -> None:
    """Register the optional short tier name; safe in every vLLM process."""
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

    if "hust_fs" not in SecondaryTierFactory._registry:
        SecondaryTierFactory.register_tier(
            "hust_fs",
            "vllm_hust_kv_tiering.fs.manager",
            "SegmentFileSystemTier",
        )


def __getattr__(name: str) -> Any:
    """Keep package metadata and Manager provider discovery host-independent."""
    if name == "HustTieringOffloadingSpec":
        from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec

        return HustTieringOffloadingSpec
    if name == "SegmentFileSystemTier":
        from vllm_hust_kv_tiering.fs.manager import SegmentFileSystemTier

        return SegmentFileSystemTier
    raise AttributeError(name)


__all__ = ["HustTieringOffloadingSpec", "SegmentFileSystemTier", "register"]
