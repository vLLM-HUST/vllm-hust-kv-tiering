import mmap
from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm_hust_kv_tiering.spec import HustTieringOffloadingSpec


def test_actual_host_factory_segment_store_restore(tmp_path):
    parallel = SimpleNamespace(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        rank=0,
    )
    group = KVCacheGroupSpec(
        ["attention"],
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.int8),
    )
    spec = SimpleNamespace(
        extra_config={"ascend_copy_backend": "torch_sync"},
        block_size_factor=1,
        vllm_config=SimpleNamespace(
            parallel_config=parallel,
            use_v2_model_runner=False,
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=8),
            model_config=SimpleNamespace(model="fixture"),
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
    )
    raw = mmap.mmap(-1, 4 * 4096)
    raw[:] = bytes(range(256)) * 64
    view = memoryview(raw).cast("B", shape=(4, 4096))
    HustTieringOffloadingSpec._register_storage()
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "hust_fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "storage_layout": "segment",
        },
        view,
        spec,
    )
    key = make_offload_key(bytes(range(32)), 0)
    request = SimpleNamespace(req_id="fixture")
    try:
        tier.submit_store(JobMetadata(1, [key], np.array([1]), False, request))
        tier.drain_jobs()
        result = list(tier.get_finished_jobs())
        assert len(result) == 1 and result[0].success
        expected = bytes(raw[4096:8192])
        raw[12288:16384] = bytes(4096)
        tier.submit_load(JobMetadata(2, [key], np.array([3]), True, request))
        tier.drain_jobs()
        result = list(tier.get_finished_jobs())
        assert len(result) == 1 and result[0].success
        assert bytes(raw[12288:16384]) == expected
    finally:
        tier.shutdown()
        view.release()
        raw.close()
