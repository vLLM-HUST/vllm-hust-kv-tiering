"""vLLM-HUST v1 spec entry point.

The v1 offloading factory loads this class from ``spec_module_path``.  The
base spec owns the stable connector protocol; this plugin supplies its own
secondary storage implementation through the host's ``hust_fs`` registry entry.
"""

from dataclasses import replace

from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from vllm_hust_kv_tiering.lifecycle import LifecycleConfig, SessionLifecycleManager
from vllm_hust_kv_tiering.lifecycle_manager import LifecycleAwareManager
from vllm_hust_kv_tiering.policy import TieringPolicyConfig, boolean, positive_int
from vllm_hust_kv_tiering.residency import build_tiering_metric_definitions


class HustTieringOffloadingSpec(TieringOffloadingSpec):
    """Tiering spec pinned to the vLLM-HUST v1 offloading ABI."""

    @staticmethod
    def _register_storage():
        from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

        from vllm_hust_kv_tiering.fs.manager import SegmentFileSystemTier

        if "hust_fs" not in SecondaryTierFactory._registry:
            SecondaryTierFactory.register_tier(
                "hust_fs", "vllm_hust_kv_tiering.fs.manager", "SegmentFileSystemTier"
            )
        if (
            SecondaryTierFactory.get_tier_class({"type": "hust_fs"})
            is not SegmentFileSystemTier
        ):
            raise ValueError("hust_fs is already registered by another implementation")

    def __init__(self, vllm_config, kv_cache_config):
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        host_options = {
            "spec_name",
            "spec_module_path",
            "cpu_bytes_to_use",
            "block_size",
            "eviction_policy",
            "secondary_tiers",
            "ascend_copy_backend",
            "offload_prompt_only",
            "self_describing_kv_events",
            "store_threshold",
        }
        unknown = sorted(
            key
            for key in extra
            if key not in host_options
            and not key.startswith(("tiering_", "lifecycle_", "residency_"))
        )
        if unknown:
            raise ValueError(f"unsupported KV tiering option(s): {', '.join(unknown)}")
        threshold = extra.get("store_threshold", 0)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or threshold not in (0, 1)
        ):
            raise ValueError(
                "store_threshold must be 0 or 1; use tiering session/cost admission for filtering"
            )
        positive_int(extra, "cpu_bytes_to_use", 0)
        boolean(extra, "offload_prompt_only", True)
        boolean(extra, "self_describing_kv_events", False)
        if "block_size" in extra:
            positive_int(extra, "block_size", 0)
        if extra.get("eviction_policy", "lru") not in ("lru", "arc"):
            raise ValueError("eviction_policy must be lru or arc")
        self._register_storage()
        super().__init__(vllm_config, kv_cache_config)
        if self.num_blocks < 1:
            raise ValueError(
                "CPU budget cannot hold one aligned KV block, or KV geometry is empty"
            )
        self._policy_config = TieringPolicyConfig.from_extra_config(self.extra_config)
        lifecycle_config = LifecycleConfig.from_extra_config(self.extra_config)
        # Residency ownership is required internally for safe policy decisions.
        # TTL and age-based demotion remain disabled unless configured; the
        # package itself is enabled only through an explicit transfer config.
        self._lifecycle = SessionLifecycleManager(
            replace(lifecycle_config, residency_tracking_enabled=True)
        )
        backend = self.extra_config.get("ascend_copy_backend")
        if backend not in (None, "torch_async", "torch_sync"):
            raise ValueError(
                "ascend_copy_backend must be torch_async (legacy alias: torch_sync), or omitted for the host worker"
            )
        self._wrapped_manager = None

    def get_manager(self):
        manager = super().get_manager()
        if self._wrapped_manager is None:
            try:
                self._wrapped_manager = LifecycleAwareManager(
                    manager,
                    self._lifecycle,
                    self._policy_config,
                    bytes_per_block=self.kv_bytes_per_offloaded_block,
                    tokens_per_block=min(self.gpu_block_size) * self.block_size_factor,
                )
            except BaseException:
                manager.shutdown()
                self._manager = None
                self._scheduler_mmap = None
                raise
        return self._wrapped_manager

    @classmethod
    def build_metric_definitions(cls, extra_config):
        cls._register_storage()
        TieringPolicyConfig.from_extra_config(extra_config)
        LifecycleConfig.from_extra_config(extra_config)
        metrics = super().build_metric_definitions(extra_config)
        metrics.update(build_tiering_metric_definitions())
        return metrics

    def _ascend_copy_enabled(self):
        return self.extra_config.get("ascend_copy_backend") in (
            "torch_async",
            "torch_sync",
        )

    def get_worker(self, kv_caches):
        if not self._ascend_copy_enabled():
            return super().get_worker(kv_caches)
        import torch
        from vllm.distributed import get_world_group
        from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

        from vllm_hust_kv_tiering.ascend_layout import AscendLayout
        from vllm_hust_kv_tiering.ascend_worker import (
            AscendCopyWorker,
            AscendLayoutWorker,
        )

        if not hasattr(torch, "npu"):
            raise ValueError("torch_async requires the Ascend torch_npu runtime")
        if self._worker is None:
            region = SharedOffloadRegion(
                instance_id=self.vllm_config.instance_id,
                num_blocks=self.num_blocks,
                rank=get_world_group().rank_in_group,
                kv_bytes_per_block=self.kv_bytes_per_offloaded_block,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            try:
                worker_type = (
                    AscendLayoutWorker
                    if isinstance(kv_caches, AscendLayout)
                    else AscendCopyWorker
                )
                self._worker = worker_type(
                    kv_caches,
                    self.block_size_factor,
                    self.num_blocks,
                    region,
                )
            except BaseException:
                region.cleanup()
                raise
        return self._worker
