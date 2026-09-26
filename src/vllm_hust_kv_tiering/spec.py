"""vLLM-HUST v1 spec entry point.

The v1 offloading factory loads this class from ``spec_module_path``.  The
base spec owns the stable connector protocol; this plugin supplies its own
secondary storage implementation through either ``type=hust_fs`` (after the
general plugin is loaded) or the portable ``module_path`` configuration.
"""

from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


class HustTieringOffloadingSpec(TieringOffloadingSpec):
    """Tiering spec pinned to the vLLM-HUST v1 offloading ABI."""

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
