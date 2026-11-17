# SPDX-License-Identifier: Apache-2.0
"""One admission hook; hybrid-group mapping/fencing remains host-owned."""

from copy import copy

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)


class TieringConnectorScheduler(OffloadingConnectorScheduler):
    def _build_store_jobs(self, scheduler_output):
        if not self.manager.should_build_store_jobs():
            # Crucially do not call advance_stored_idx(): postpone, do not
            # permanently skip prefixes while the pressure gate is closed.
            return {}
        scheduled = scheduler_output.num_scheduled_tokens
        eligible = {
            req_id: tokens
            for req_id, tokens in scheduled.items()
            if not self.manager.store_request_skipped(req_id)
        }
        if len(eligible) == len(scheduled):
            return super()._build_store_jobs(scheduler_output)
        # This view is consumed only by the host's store-job builder. The
        # original output still drives request updates, loads, preemption and
        # flushes. Never advance stored_idx to manufacture a successful store.
        if not eligible:
            return {}
        store_output = copy(scheduler_output)
        store_output.num_scheduled_tokens = eligible
        return super()._build_store_jobs(store_output)
