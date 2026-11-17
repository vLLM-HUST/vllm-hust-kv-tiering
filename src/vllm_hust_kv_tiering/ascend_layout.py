"""Pack logical cache blocks from Ascend's separate K/V and recurrent views.

The CPU slot belongs to an allocation, while its contents belong to the cache
group storing that block. Never reinterpret a shared SoA allocation as AoS:
each layer is copied through the same tensor views used by the model runner.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
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
                raise ValueError(  # noqa: TRY004 - pre-existing contract
                    "unsupported Ascend cache spec"
                )
            index = allocation_by_layer[name]
            if index in used:
                raise ValueError("multiple layers in one group share an allocation")
            used.add(index)
            tensors = caches[name]
            # A combined [2, num_blocks, ...] K/V tensor is ambiguous when
            # num_blocks == 2: divisibility alone would silently pack whole
            # planes rather than canonical blocks. The known non-MLA Ascend
            # FullAttention runner supplies explicit separate K/V views.
            if (
                isinstance(spec, FullAttentionSpec)
                and not isinstance(spec, MLAAttentionSpec)
                and (not isinstance(tensors, (list, tuple)) or len(tensors) != 2)
            ):
                raise ValueError(
                    "Ascend FullAttention requires separate K/V tensor views"
                )
            if isinstance(tensors, torch.Tensor):
                tensors = (tensors,)
            if not isinstance(tensors, (list, tuple)) or not tensors:
                raise ValueError("missing layer cache tensors")
            if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise ValueError("layer cache views must be torch tensors")
            if isinstance(spec, MambaSpec):
                if len(tensors) != len(spec.shapes) or len(spec.dtypes) != len(
                    spec.shapes
                ):
                    raise ValueError(
                        "Mamba cache requires exactly the declared state views"
                    )
                for tensor, shape, dtype in zip(
                    tensors, spec.shapes, spec.dtypes, strict=True
                ):
                    if tuple(tensor.shape) != (count, *shape):
                        raise ValueError(
                            "Mamba state view shape does not match its spec"
                        )
                    if tensor.dtype != dtype:
                        raise ValueError(
                            "Mamba state view dtype does not match its spec"
                        )
            rows = []
            for tensor in tensors:
                if tensor.ndim < 1 or tensor.shape[0] % count:
                    raise ValueError("Ascend cache requires a leading block axis")
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
    layout = AscendLayout(pages, layers, groups)
    _check_canonical_noninterference(layout, count)
    _trace_layout(config, caches, layout)
    return layout


def _check_canonical_noninterference(layout, count):
    """Host reuse fences protect IDs, not arbitrary overlapping byte ranges.

    Shared SoA layers may intentionally alias the SAME canonical block ID.
    Reject any cross-ID overlap before constructing a transfer worker. Only
    tensor metadata is inspected; no device synchronization or byte read occurs.
    Disjoint planes and identical shared rows take constant-time fast paths.
    """
    if count <= 1:
        return
    by_device = {}
    for layer_index, (_, rows) in enumerate(layout.layers):
        for row_index, row in enumerate(rows):
            start, stride, width = row.data_ptr(), row.stride(0), row.shape[1]
            end = start + (count - 1) * stride + width
            for previous in by_device.setdefault(str(row.device), []):
                (
                    other_start,
                    other_stride,
                    other_width,
                    other_end,
                    other_layer,
                    other_row,
                ) = previous
                if end <= other_start or other_end <= start:
                    continue
                if start == other_start and stride == other_stride:
                    continue
                left = right = 0
                left_start, right_start = start, other_start
                while left < count and right < count:
                    left_end = left_start + width
                    right_end = right_start + other_width
                    if (
                        left != right
                        and left_start < right_end
                        and right_start < left_end
                    ):
                        raise ValueError(
                            "GPU cache views overlap across canonical block IDs "
                            f"(layer {layer_index} row {row_index} block {left}; "
                            f"layer {other_layer} row {other_row} block {right})"
                        )
                    if left_end <= right_end:
                        left += 1
                        left_start += stride
                    if right_end <= left_end:
                        right += 1
                        right_start += other_stride
            by_device[str(row.device)].append(
                (start, stride, width, end, layer_index, row_index)
            )


def _layout_trace_payload(config, caches, layout):
    """Read tensor metadata only; never copy/read contents or launch kernels."""
    storages, storage_ids = [], {}

    def tensor_metadata(tensor):
        storage = tensor.untyped_storage()
        base, nbytes = int(storage.data_ptr()), int(storage.nbytes())
        key = (str(tensor.device), base, nbytes)
        if key not in storage_ids:
            storage_ids[key] = f"storage-{len(storage_ids)}"
            storages.append(
                {
                    "storage_id": storage_ids[key],
                    "device": str(tensor.device),
                    "nbytes": nbytes,
                }
            )
        return {
            "shape": [int(n) for n in tensor.shape],
            "strides_elements": [int(n) for n in tensor.stride()],
            "dtype": str(tensor.dtype),
            "element_size_bytes": int(tensor.element_size()),
            "storage_offset_elements": int(tensor.storage_offset()),
            "storage_id": storage_ids[key],
            "byte_offset": int(tensor.data_ptr()) - base,
        }

    rank, rank_source = None, "unavailable"
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_initialized():
        rank, rank_source = int(distributed.get_rank()), "torch.distributed.get_rank"
    elif os.environ.get("RANK") is not None:
        rank, rank_source = int(os.environ["RANK"]), "RANK environment"
    payload = {
        "schema": "ascend-layout-trace-v1",
        "pid": os.getpid(),
        "rank": rank,
        "rank_source": rank_source,
        "num_blocks": int(config.num_blocks),
        "pointer_policy": "normalized storage IDs and relative byte offsets; no raw pointers or tensor contents",
        "allocations": [
            {
                "allocation_index": i,
                "size_bytes": int(allocation.size),
                "page_bytes": int(layout.page_sizes[i]),
                "block_stride": int(allocation.block_stride or 0),
                "shared_by": list(allocation.shared_by),
            }
            for i, allocation in enumerate(config.kv_cache_tensors)
        ],
        "groups": [],
        "layers": [],
        "storages": storages,
    }
    index = 0
    for group_id, group in enumerate(config.kv_cache_groups):
        indices = []
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            tensors = caches[name]
            if isinstance(tensors, torch.Tensor):
                tensors = (tensors,)
            allocation, rows = layout.layers[index]
            payload["layers"].append(
                {
                    "layer_index": index,
                    "layer_name": name,
                    "group_id": group_id,
                    "allocation_index": int(allocation),
                    "spec_type": type(spec).__name__,
                    "block_size": int(spec.block_size),
                    "spec_page_bytes": int(spec.page_size_bytes),
                    "original_tensors": [tensor_metadata(tensor) for tensor in tensors],
                    "byte_rows": [tensor_metadata(row) for row in rows],
                }
            )
            indices.append(index)
            index += 1
        payload["groups"].append({"group_id": group_id, "layer_indices": indices})
    return payload


def _trace_layout(config, caches, layout):
    directory = os.environ.get("HUST_TIERING_LAYOUT_TRACE")
    if not directory:
        return
    try:
        target = Path(directory).expanduser()
        if not target.is_absolute():
            raise ValueError(
                "HUST_TIERING_LAYOUT_TRACE must name an absolute owned directory"
            )
        payload = _layout_trace_payload(config, caches, layout)
        target.mkdir(parents=True, exist_ok=True)
        rank = payload["rank"] if payload["rank"] is not None else "unknown"
        path = target / f"layout-rank{rank}-pid{os.getpid()}-{uuid.uuid4().hex}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except Exception:
        logging.getLogger(__name__).warning(
            "Cannot write Ascend layout trace", exc_info=True
        )
