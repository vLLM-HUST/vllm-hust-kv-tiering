from vllm_hust_kv_tiering import register
from vllm_hust_kv_tiering.fs.manager import SegmentFileSystemTier
from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec


def test_v1_plugin_contract() -> None:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

    register()
    assert issubclass(HustTieringOffloadingSpec, OffloadingSpec)
    assert issubclass(SegmentFileSystemTier, SecondaryTierManager)
    assert "hust_fs" in SecondaryTierFactory._registry
