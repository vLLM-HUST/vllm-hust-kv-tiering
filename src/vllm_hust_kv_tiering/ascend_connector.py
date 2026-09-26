"""Plugin-owned offloading registration for Ascend logical cache layouts."""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

from vllm_hust_kv_tiering.ascend_layout import build_layout


class HustAscendTieringConnector(OffloadingConnector):
    def register_kv_caches(self, kv_caches):
        worker = self.connector_worker
        if worker is None or not worker.spec._ascend_copy_enabled():
            raise ValueError("Ascend tiering requires the explicit torch_sync profile")
        layout = build_layout(worker.spec.kv_cache_config, kv_caches)
        worker._init_worker(layout)
