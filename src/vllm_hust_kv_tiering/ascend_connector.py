"""Plugin-owned offloading registration for Ascend logical cache layouts."""

import os

from vllm.logger import init_logger

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

from vllm_hust_kv_tiering.ascend_layout import build_layout

logger = init_logger(__name__)


class HustAscendTieringConnector(OffloadingConnector):
    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        if os.environ.get("HUST_TIERING_DIAGNOSTICS") == "1":
            for kind, jobs in (
                ("load", metadata.load_jobs),
                ("store", metadata.store_jobs),
            ):
                for job_id, job in jobs.items():
                    gpu = job.dst_spec if kind == "load" else job.src_spec
                    cpu = job.src_spec if kind == "load" else job.dst_spec
                    logger.info(
                        "TIERING_JOB kind=%s job=%s request=%s groups=%s logical=%s device=%s cpu=%s",
                        kind,
                        job_id,
                        job.req_id,
                        gpu.group_sizes,
                        gpu.block_indices,
                        gpu.block_ids,
                        cpu.block_ids,
                    )
        return metadata

    def register_kv_caches(self, kv_caches):
        worker = self.connector_worker
        if worker is None or not worker.spec._ascend_copy_enabled():
            raise ValueError("Ascend tiering requires the explicit torch_sync profile")
        layout = build_layout(worker.spec.kv_cache_config, kv_caches)
        if os.environ.get("HUST_TIERING_DIAGNOSTICS") == "1":
            for i, group in enumerate(worker.spec.kv_cache_config.kv_cache_groups):
                refs = layout.group_data_refs[i]
                logger.info(
                    "TIERING_GROUP index=%s spec=%s layers=%s first_rows=%s",
                    i,
                    group.kv_cache_spec,
                    len(refs),
                    [tuple(r.shape) for r in layout.layers[refs[0].tensor_idx][1]],
                )
        worker._init_worker(layout)
