from pathlib import Path

import pytest
from vllm_hust_ext.manifest import activation_blocker, load_manifest
from vllm_hust_ext.cli import _merge_provider_plan
from vllm_hust_kv_tiering.manager_provider import TieringProvider


def test_manager_plan_preserves_frontier_arguments():
    import vllm_hust_kv_tiering

    manifest = load_manifest(
        Path(vllm_hust_kv_tiering.__file__).parent
        / "manifests/vllm-hust-extension-v0.2.json"
    )
    assert activation_blocker(manifest) is None
    config = {"cpu_bytes_to_use": 8388608, "storage_directory": "/tmp/tiering-test"}
    plan = TieringProvider().plan(manifest, config, enabled=True)
    command = [
        "vllm",
        "serve",
        "MODEL",
        "--enable-prefix-caching",
        "--async-scheduling",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
    ]
    generated = _merge_provider_plan(command, plan)
    assert generated[: len(command)] == command
    assert generated[len(command)] == "--kv-transfer-config"
    with pytest.raises(ValueError, match="conflicts"):
        _merge_provider_plan(command + ["--kv-transfer-config", "{}"], plan)
    with pytest.raises(ValueError, match="positive integer"):
        TieringProvider().plan(
            manifest, {**config, "cpu_bytes_to_use": True}, enabled=True
        )


def test_storage_profile_resolves_on_actual_host_without_general_plugin(monkeypatch):
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
    from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec
    from vllm_hust_kv_tiering.fs.manager import SegmentFileSystemTier

    registry = dict(SecondaryTierFactory._registry)
    registry.pop("hust_fs", None)
    monkeypatch.setattr(SecondaryTierFactory, "_registry", registry)
    HustTieringOffloadingSpec._register_storage()
    HustTieringOffloadingSpec._register_storage()
    assert (
        SecondaryTierFactory.get_tier_class({"type": "hust_fs"})
        is SegmentFileSystemTier
    )
    registry["hust_fs"] = lambda: object
    with pytest.raises(ValueError, match="another implementation"):
        HustTieringOffloadingSpec._register_storage()
