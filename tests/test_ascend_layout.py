import os
import uuid

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm.v1.kv_cache_interface")
pytestmark = pytest.mark.host

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


def fixture(device="cpu", typed=False):
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
                    shapes=((8,), (8,)),
                    dtypes=(torch.int8, torch.int8),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    caches = {
        "attn": (raw[32:64].view(8, 4), raw[64:96].view(8, 4)),
        # Native Ascend aligns the SSM row width with one attention K row.
        # V occupies Mamba padding, so cross-group aliases share only SAME IDs.
        "mamba": [raw[:32].view(4, 8), raw[32:64].view(4, 8)],
    }
    if typed:
        caches = {
            "attn": tuple(t.view(torch.bfloat16) for t in caches["attn"]),
            "mamba": [
                caches["mamba"][0].view(torch.bfloat16),
                caches["mamba"][1].view(torch.float32),
            ],
        }
        config.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.bfloat16,
            page_size_padded=24,
        )
        config.kv_cache_groups[1].kv_cache_spec = MambaSpec(
            block_size=4,
            shapes=((4,), (2,)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_cache_mode="align",
        )
    return raw, config, caches


def test_logical_rows_retain_runner_storage_and_share_cpu_allocation():
    raw, config, caches = fixture()
    layout = build_layout(config, caches)
    assert layout.page_sizes == [24]
    assert [r[0].page_size_bytes for r in layout.group_data_refs] == [16, 16]
    assert [index for index, _ in layout.layers] == [0, 0]
    layout.layers[0][1][0][1].fill_(99)
    assert torch.all(raw[40:48] == 99)
    assert torch.equal(raw[:40], torch.arange(40, dtype=torch.int8))
    config.kv_cache_tensors[0].block_stride = 24
    with pytest.raises(ValueError, match="packed"):
        build_layout(config, caches)


@pytest.mark.parametrize("typed", [False, True])
def test_rejects_cross_id_soa_alias_even_when_single_group_roundtrip_would_pass(typed):
    raw, config, caches = fixture(typed=typed)
    # Historical tiny fixture: SSM block1 occupies bytes48:64, overlapping
    # attention K block2 at48:56. Per-group byte roundtrips missed this hazard.
    unsafe_ssm = raw[32:].view(4, 16)
    caches["mamba"][1] = unsafe_ssm.view(torch.float32) if typed else unsafe_ssm
    config.kv_cache_groups[1].kv_cache_spec = MambaSpec(
        block_size=4 if typed else 8,
        shapes=((4,), (4,)) if typed else ((8,), (16,)),
        dtypes=(torch.bfloat16, torch.float32) if typed else (torch.int8, torch.int8),
        mamba_cache_mode="align",
    )
    with pytest.raises(ValueError, match="across canonical block IDs"):
        build_layout(config, caches)


def test_rejects_a_nonleading_logical_block_axis():
    raw, config, caches = fixture()
    caches["attn"] = (raw[32:64].view(2, 16), caches["attn"][1])
    with pytest.raises(ValueError, match="leading block axis"):
        build_layout(config, caches)


@pytest.mark.npu
@pytest.mark.skipif(
    os.environ.get("VLLM_KV_TIERING_RUN_NPU_TESTS") != "1"
    or not hasattr(torch, "npu")
    or not torch.npu.is_available(),
    reason="requires VLLM_KV_TIERING_RUN_NPU_TESTS=1 and an available real Ascend NPU",
)
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("device_index", [0, 1])
def test_real_npu_soa_hybrid_store_restore_and_transaction_validation(
    typed, device_index
):
    if device_index >= torch.npu.device_count():
        pytest.skip("requested NPU is not visible")
    torch.npu.set_device(device_index)
    raw, config, caches = fixture(f"npu:{device_index}", typed)
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
            worker.wait({group * 2})
            assert worker.get_finished()[0].transfer_size == 32
            for row in layout.layers[group][1]:
                row[1:3].zero_()
            worker.submit_load(group * 2 + 1, cpu, gpu)
            worker.wait({group * 2 + 1})
            worker.get_finished()
            assert torch.equal(raw.cpu(), expected)
        # A historical Mamba block1 restore must not corrupt a distinct,
        # subsequently modified/live attention block2 in their shared raw SoA.
        gpu = GPULoadStoreSpec([1], [0, 1], [0, 1])
        cpu = CPULoadStoreSpec([0])
        worker.submit_store(4, gpu, cpu)
        worker.wait({4})
        worker.get_finished()
        for row in layout.layers[0][1]:
            row[2].fill_(77)
        expected = raw.cpu()
        for row in layout.layers[1][1]:
            row[1].zero_()
        worker.submit_load(5, cpu, gpu)
        worker.wait({5})
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
