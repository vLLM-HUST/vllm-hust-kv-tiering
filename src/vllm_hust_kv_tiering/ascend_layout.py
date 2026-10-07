"""Pack logical cache blocks from Ascend's separate K/V and recurrent views.

The CPU slot belongs to an allocation, while its contents belong to the cache
group storing that block. Never reinterpret a shared SoA allocation as AoS:
each layer is copied through the same tensor views used by the model runner.
"""

from dataclasses import dataclass

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.base import CanonicalKVCacheRef


@dataclass
class AscendLayout:
    page_sizes: list[int]
    layers: list[tuple[int, list[torch.Tensor]]]
    group_data_refs: list[list[CanonicalKVCacheRef]]


def build_layout(config, caches):
    """Build zero-copy byte rows; reject layouts that require a copied view."""
    count = config.num_blocks
    if count <= 0:
        raise ValueError("cache must contain blocks")
    pages, allocation_by_layer = [], {}
    for allocation in config.kv_cache_tensors:
        if allocation.block_stride or allocation.size % count:
            raise ValueError("packed or uneven Ascend allocation is unsupported")
        index = len(pages)
        pages.append(allocation.size // count)
        for name in allocation.shared_by:
            if name in allocation_by_layer:
                raise ValueError("layer belongs to multiple allocations")
            allocation_by_layer[name] = index

    layers, groups = [], []
    for group in config.kv_cache_groups:
        refs = []
        used = set()
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if not isinstance(spec, (AttentionSpec, MambaSpec)):
                raise ValueError("unsupported Ascend cache spec")
            index = allocation_by_layer[name]
            if index in used:
                raise ValueError("multiple layers in one group share an allocation")
            used.add(index)
            tensors = caches[name]
            if isinstance(tensors, torch.Tensor):
                tensors = (tensors,)
            if not tensors or not isinstance(tensors, (list, tuple)):
                raise ValueError("missing layer cache tensors")
            rows = []
            for tensor in tensors:
                # view, deliberately not reshape: silently allocating a copy
                # here would disconnect restored data from the model's cache.
                row = tensor.view(torch.int8).view(count, -1)
                if row.stride(1) != 1 or row.stride(0) < row.shape[1]:
                    raise ValueError("overlapping or non-contiguous byte rows")
                rows.append(row)
            size = sum(row.shape[1] for row in rows)
            if size <= 0 or size > pages[index] or size > spec.page_size_bytes:
                raise ValueError("logical cache exceeds its allocation page")
            refs.append(CanonicalKVCacheRef(len(layers), size))
            layers.append((index, rows))
        groups.append(refs)
    return AscendLayout(pages, layers, groups)
