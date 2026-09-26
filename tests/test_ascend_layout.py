import uuid

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

from vllm_hust_kv_tiering.ascend_layout import build_layout
from vllm_hust_kv_tiering.ascend_worker import AscendLayoutWorker


def fixture(device="cpu"):
    raw = torch.arange(96, dtype=torch.int8).to(device)
    config = KVCacheConfig(
        4,
        [KVCacheTensor(96, ["attn", "mamba"])],
        [
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.int8,
                    page_size_padded=24,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=8,
                    shapes=((8,), (16,)),
                    dtypes=(torch.int8, torch.int8),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    caches = {
        "attn": (raw[32:64].view(8, 4), raw[64:96].view(8, 4)),
        "mamba": [raw[:32].view(4, 8), raw[32:].view(4, 16)],
    }
    return raw, config, caches


def test_logical_rows_retain_runner_storage_and_share_cpu_allocation():
    raw, config, caches = fixture()
    layout = build_layout(config, caches)
    assert layout.page_sizes == [24]
    assert [r[0].page_size_bytes for r in layout.group_data_refs] == [16, 24]
    assert [index for index, _ in layout.layers] == [0, 0]
    layout.layers[0][1][0][1].fill_(99)
    assert torch.all(raw[40:48] == 99)
    assert torch.equal(raw[:40], torch.arange(40, dtype=torch.int8))
    config.kv_cache_tensors[0].block_stride = 24
    with pytest.raises(ValueError, match="packed"):
        build_layout(config, caches)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU"
)
def test_real_npu_soa_hybrid_store_restore_and_transaction_validation():
    torch.npu.set_device(0)
    raw, config, caches = fixture("npu:0")
    layout = build_layout(config, caches)
    region = SharedOffloadRegion(uuid.uuid4().hex, 4, 0, 4096, 48)
    worker = AscendLayoutWorker(layout, 2, 4, region)
    try:
        for group in (0, 1):
            expected = raw.cpu()
            sizes = [0, 0]
            sizes[group] = 2
            logical = [0, 0]
            logical[group] = 1
            gpu = GPULoadStoreSpec([1, 2], sizes, logical)
            cpu = CPULoadStoreSpec([0, 1])
            worker.submit_store(group * 2, gpu, cpu)
            assert worker.get_finished()[0].transfer_size == (32 if group == 0 else 48)
            for row in layout.layers[group][1]:
                row[1:3].zero_()
            worker.submit_load(group * 2 + 1, cpu, gpu)
            worker.get_finished()
            assert torch.equal(raw.cpu(), expected)
        before = raw.cpu()
        with pytest.raises(ValueError, match="device block"):
            worker.submit_load(
                10, CPULoadStoreSpec([0]), GPULoadStoreSpec([1, 99], [2, 0], [0, 0])
            )
        assert torch.equal(raw.cpu(), before)
    finally:
        worker.shutdown()
