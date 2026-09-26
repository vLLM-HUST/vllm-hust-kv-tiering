import uuid

import pytest
import torch
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm_hust_kv_tiering.ascend_worker import AscendCopyWorker, copy_plan
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def test_group_alignment_and_padding():
    refs = [[CanonicalKVCacheRef(0, 5)], [CanonicalKVCacheRef(1, 3)]]
    gpu = GPULoadStoreSpec([7, 8, 9], [2, 1], [1, 3])
    cpu = CPULoadStoreSpec([4, 5, 6])
    assert copy_plan(gpu, cpu, refs, 2) == [
        (0, 7, 4, 1, 5),
        (0, 8, 5, 0, 5),
        (1, 9, 6, 1, 3),
    ]
    with pytest.raises(ValueError, match="shorter"):
        copy_plan(gpu, CPULoadStoreSpec([4]), refs, 2)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires real Ascend",
)
def test_real_npu_hybrid_partial_page_round_trip(tmp_path):
    torch.npu.set_device(0)
    tensors = [
        torch.arange(64, dtype=torch.int8).view(4, 16).npu(),
        (torch.arange(64, dtype=torch.int8) + 40).view(4, 16).npu(),
    ]
    expected = [tensor.cpu() for tensor in tensors]
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(t, 16) for t in tensors],
        group_data_refs=[[CanonicalKVCacheRef(0, 11)], [CanonicalKVCacheRef(1, 7)]],
    )
    region = SharedOffloadRegion(
        instance_id=uuid.uuid4().hex,
        num_blocks=4,
        rank=0,
        kv_bytes_per_block=4096,
        cpu_page_size=64,
    )
    worker = AscendCopyWorker(caches, 2, 4, region)
    gpu = GPULoadStoreSpec([1, 2, 3], [2, 1], [1, 3])
    cpu = CPULoadStoreSpec([0, 1, 2])
    try:
        assert worker.submit_store(1, gpu, cpu)
        assert worker.get_finished()[0].transfer_size == 29
        tensors[0][1:3].zero_()
        tensors[1][3].zero_()
        assert worker.submit_load(2, cpu, gpu)
        worker.wait({2})
        assert worker.get_finished()[0].transfer_size == 29
        assert torch.equal(tensors[0][1:3, :11].cpu(), expected[0][1:3, :11])
        assert torch.equal(tensors[1][3, :7].cpu(), expected[1][3, :7])
        assert torch.count_nonzero(tensors[0][1:3, 11:]).item() == 0
        assert torch.count_nonzero(tensors[1][3, 7:]).item() == 0
        assert worker.get_finished() == []
    finally:
        worker.shutdown()
