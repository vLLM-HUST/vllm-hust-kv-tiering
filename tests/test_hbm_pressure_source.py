"""Actual frozen BlockPool primary-index regression; CPU/metadata only."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.v1.core.block_pool")

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

from vllm_hust_kv_tiering.host_adapter import HBMPressureSource
from vllm_hust_kv_tiering.policy import PressureGate

pytestmark = pytest.mark.host


class NoIterationIndex:
    def __init__(self, count):
        self.count = count
        self.len_calls = 0

    def __len__(self):
        self.len_calls += 1
        return self.count

    def __iter__(self):
        raise AssertionError("pressure collection must not iterate the prefix index")

    def items(self):
        raise AssertionError("pressure collection must not scan index items")

    def values(self):
        raise AssertionError("pressure collection must not scan physical blocks")


def fake_pool(count=9, usage=0.0):
    return SimpleNamespace(
        num_gpu_blocks=11,
        get_usage=lambda: usage,
        cached_block_hash_to_block=NoIterationIndex(count),
        cached_block_hashes_by_block={},
    )


def key(value):
    return make_block_hash_with_group_id(BlockHash(value.to_bytes(32, "big")), 0)


def test_real_pool_primary_hashes_open_pressure_gate_with_no_aliases():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    source = HBMPressureSource(pool)
    blocks = pool.get_new_blocks(9)
    for index, block in enumerate(blocks):
        pool._insert_block_hash(key(index), block, num_tokens=1)
    pool.free_blocks(blocks)
    assert pool.get_usage() == 0.0
    assert pool.cached_block_hashes_by_block == {}  # supplemental aliases ONLY
    assert len(pool.cached_block_hash_to_block) == 9
    assert source() == pytest.approx(0.9)
    assert PressureGate(0.8, 0.6).update(source())


def test_real_pool_reset_replaces_index_and_pressure_follows_new_object():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    source = HBMPressureSource(pool)
    blocks = pool.get_new_blocks(9)
    for index, block in enumerate(blocks):
        pool._insert_block_hash(key(index), block, num_tokens=1)
    pool.free_blocks(blocks)
    assert source() == pytest.approx(0.9)
    previous_index = pool.cached_block_hash_to_block
    assert pool.reset_prefix_cache()
    assert pool.cached_block_hash_to_block is not previous_index
    assert source() == 0.0


def test_primary_count_is_o1_without_iteration_and_refetched_after_replacement():
    pool = fake_pool()
    source = HBMPressureSource(pool)
    assert source() == 0.9
    previous = pool.cached_block_hash_to_block
    assert previous.len_calls == 1
    pool.cached_block_hash_to_block = NoIterationIndex(1)
    assert source() == 0.1
    assert previous.len_calls == 1
    assert pool.cached_block_hash_to_block.len_calls == 1


@pytest.mark.parametrize("bad_index", [None, object(), 7])
def test_primary_index_abi_required_even_when_alias_dictionary_exists(bad_index):
    pool = fake_pool()
    pool.cached_block_hash_to_block = bad_index
    with pytest.raises(TypeError, match="primary prefix-cache index"):
        HBMPressureSource(pool)


@pytest.mark.parametrize(
    "count,usage,expected",
    [(0, -0.2, 0.0), (0, 1.2, 1.0), (35, 0.0, 1.0), (9, 0.8, 0.9)],
)
def test_pressure_clamps_and_takes_maximum_not_sum(count, usage, expected):
    assert HBMPressureSource(fake_pool(count, usage))() == expected


def test_cached_key_count_can_undercount_duplicate_physical_blocks():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        pool._insert_block_hash(key(1), block, num_tokens=1)
    pool.free_blocks(blocks)
    assert len(pool.cached_block_hash_to_block) == 1
    assert HBMPressureSource(pool)() == 0.1  # not an exact physical count of2


def test_cached_key_count_can_exceed_physical_count_with_aliases():
    pool = BlockPool(num_gpu_blocks=11, enable_caching=True, hash_block_size=1)
    block = pool.get_new_blocks(1)[0]
    pool._insert_block_hash(key(1), block, num_tokens=1)
    pool._insert_block_hash(key(2), block, num_tokens=1)
    pool.free_blocks([block])
    assert len(pool.cached_block_hash_to_block) == 2
    assert HBMPressureSource(pool)() == 0.2  # one physical block, two cache keys
