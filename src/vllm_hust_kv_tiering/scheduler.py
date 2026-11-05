# SPDX-License-Identifier: Apache-2.0
"""One admission hook; hybrid-group mapping/fencing remains host-owned."""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)


class TieringConnectorScheduler(OffloadingConnectorScheduler):
    def _build_store_jobs(self, scheduler_output):
        if not self.manager.should_build_store_jobs():
            # Crucially do not call advance_stored_idx(): postpone, do not
            # permanently skip prefixes while the pressure gate is closed.
            return {}
        return super()._build_store_jobs(scheduler_output)
