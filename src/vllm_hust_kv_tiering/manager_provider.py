"""Extension Manager integration for the explicit tiering connector profile."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_ext.providers.base import PlanAction, ProviderPlan, RenderArtifact
from vllm_hust_ext.providers.vllm import VllmProvider


class TieringProvider(VllmProvider):
    name = "hust-kv-tiering"

    def plan(self, manifest, configuration, *, enabled):
        size = configuration.get("cpu_bytes_to_use")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("cpu_bytes_to_use must be a positive integer")
        storage = configuration.get("storage_directory")
        if not isinstance(storage, str) or not Path(storage).is_absolute():
            raise ValueError("storage_directory must be an absolute path")
        connector = {
            "kv_connector": "HustAscendTieringConnector",
            "kv_connector_module_path": "vllm_hust_kv_tiering.ascend_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "HustTieringOffloadingSpec",
                "spec_module_path": "vllm_hust_kv_tiering.spec",
                "cpu_bytes_to_use": size,
                "ascend_copy_backend": "torch_sync",
                "secondary_tiers": [
                    {
                        "type": "hust_fs",
                        "root_dir": storage,
                        "storage_layout": "segment",
                        "n_read_threads": 2,
                        "n_write_threads": 2,
                    }
                ],
            },
        }
        return ProviderPlan(
            manifest.bundle_id,
            self.name,
            (
                PlanAction(
                    "render_connector_config",
                    "vllm",
                    "vllm",
                    details={"enabled": enabled},
                ),
            ),
            {"kv_transfer_config": connector},
            warnings=(
                "Experimental synchronous Ascend copies; serving qualification required.",
            ),
        )

    def render(self, plan):
        return (
            RenderArtifact(
                "kv-transfer-config.json",
                "application/json",
                json.dumps(plan.generated_config["kv_transfer_config"], indent=2),
            ),
        )
