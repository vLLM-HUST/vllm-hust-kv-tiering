"""Plugin-owned offloading registration for Ascend logical cache layouts."""

import os

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from vllm_hust_kv_tiering.ascend_layout import build_layout
from vllm_hust_kv_tiering.host_adapter import HBMPressureSource
from vllm_hust_kv_tiering.scheduler import TieringConnectorScheduler

logger = init_logger(__name__)


class HustTieringConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        # Assemble existing host components, selecting only our thin scheduler
        # admission hook. Do not construct/discard a second host scheduler/mmap.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TieringConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def bind_gpu_block_pool(self, gpu_block_pool):
        super().bind_gpu_block_pool(gpu_block_pool)
        if self.connector_scheduler is not None:
            self.connector_scheduler.manager.bind_pressure_provider(
                HBMPressureSource(gpu_block_pool)
            )

    def build_connector_meta(self, scheduler_output):
        if self.connector_scheduler is None:
            raise RuntimeError("connector metadata can only be built by the scheduler")
        self.connector_scheduler.manager.begin_step(
            preempted=bool(scheduler_output.preempted_req_ids)
        )
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
        if worker is None:
            raise RuntimeError("KV cache registration requires the worker role")
        if not worker.spec._ascend_copy_enabled():
            return super().register_kv_caches(kv_caches)
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


class HustAscendTieringConnector(HustTieringConnector):
    """Backward-compatible connector name; select the Ascend copy profile explicitly."""
