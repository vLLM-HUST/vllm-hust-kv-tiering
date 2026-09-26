"""vLLM-HUST v1 spec entry point.

The v1 offloading factory loads this class from ``spec_module_path``.  The
base spec owns the stable connector protocol; this plugin supplies its own
secondary storage implementation through the host's ``hust_fs`` registry entry.
"""

from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


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
        self._register_storage()
        super().__init__(vllm_config, kv_cache_config)

    @classmethod
    def build_metric_definitions(cls, extra_config):
        cls._register_storage()
        return super().build_metric_definitions(extra_config)

    def _ascend_copy_enabled(self):
        return self.extra_config.get("ascend_copy_backend") == "torch_sync"

    def get_worker(self, kv_caches):
        if not self._ascend_copy_enabled():
            return super().get_worker(kv_caches)
        import torch
        from vllm.distributed import get_world_group
        from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
        from vllm_hust_kv_tiering.ascend_worker import AscendCopyWorker
        from vllm_hust_kv_tiering.ascend_worker import AscendLayoutWorker
        from vllm_hust_kv_tiering.ascend_layout import AscendLayout

        if not hasattr(torch, "npu"):
            raise ValueError("torch_sync requires the Ascend runtime")
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
