"""Portable checks of metadata-only trace helpers, without torch imports."""

import ast
import json
import logging
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


class MetadataTensor:
    # Deliberately has no cpu(), clone(), numpy(), or other content operation.
    def __init__(self, offset, shape, strides, storage):
        self.offset, self.shape, self.strides, self.storage = (
            offset,
            shape,
            strides,
            storage,
        )
        self.device, self.dtype = "cpu", "torch.int8"

    def untyped_storage(self):
        return self.storage

    def data_ptr(self):
        return self.storage.data_ptr() + self.offset

    def stride(self, dimension=None):
        return self.strides if dimension is None else self.strides[dimension]

    def element_size(self):
        return 1

    def storage_offset(self):
        return self.offset


@pytest.fixture
def helpers():
    # Compile the exact stdlib-only helper bodies; importing ascend_layout itself
    # would unnecessarily require the accelerator/host packages in portable CI.
    path = Path(__file__).parents[1] / "src/vllm_hust_kv_tiering/ascend_layout.py"
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_trace_layout",
            "_layout_trace_payload",
            "_check_canonical_noninterference",
        }
    ]
    namespace = {
        "json": json,
        "logging": logging,
        "os": os,
        "uuid": uuid,
        "Path": Path,
        "torch": SimpleNamespace(Tensor=MetadataTensor, distributed=None),
        "UniformTypeKVCacheSpecs": type("UniformTypeKVCacheSpecs", (), {}),
    }
    exec(  # noqa: S102 - trusted own-source helper bodies only
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def geometry():
    storage = SimpleNamespace(data_ptr=lambda: 0x123456789ABC, nbytes=lambda: 96)
    first = MetadataTensor(32, (4, 8), (8, 1), storage)
    second = MetadataTensor(64, (4, 8), (8, 1), storage)
    allocation = SimpleNamespace(size=96, shared_by=["attention"], block_stride=0)
    spec = SimpleNamespace(block_size=8, page_size_bytes=24)
    config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[allocation],
        kv_cache_groups=[
            SimpleNamespace(layer_names=["attention"], kv_cache_spec=spec)
        ],
    )
    layout = SimpleNamespace(page_sizes=[24], layers=[(0, [first, second])])
    return config, {"attention": (first, second)}, layout


def test_trace_disabled_does_not_inspect_metadata_or_write_files(helpers, monkeypatch):
    monkeypatch.delenv("HUST_TIERING_LAYOUT_TRACE", raising=False)
    helpers["_trace_layout"](object(), object(), object())


def test_trace_normalizes_storage_and_records_geometry_only(
    helpers, monkeypatch, tmp_path
):
    monkeypatch.setenv("HUST_TIERING_LAYOUT_TRACE", str(tmp_path))
    monkeypatch.setenv("RANK", "7")
    config, caches, layout = geometry()
    helpers["_trace_layout"](config, caches, layout)
    files = list(tmp_path.glob("layout-rank7-pid*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["rank"] == 7 and payload["num_blocks"] == 4
    assert payload["allocations"][0]["shared_by"] == ["attention"]
    assert len(payload["storages"]) == 1
    rows = payload["layers"][0]["byte_rows"]
    assert [row["byte_offset"] for row in rows] == [32, 64]
    assert rows[0]["strides_elements"] == [8, 1]
    assert rows[0]["storage_id"] == rows[1]["storage_id"]
    assert str(0x123456789ABC) not in files[0].read_text()
    helpers["_trace_layout"](config, caches, layout)
    assert len(list(tmp_path.glob("layout-*.json"))) == 2  # no overwrite


def test_trace_invalid_destination_does_not_change_layout(helpers, monkeypatch, caplog):
    monkeypatch.setenv("HUST_TIERING_LAYOUT_TRACE", "relative-not-owned")
    config, caches, layout = geometry()
    before = layout.layers
    with caplog.at_level(logging.WARNING):
        helpers["_trace_layout"](config, caches, layout)
    assert layout.layers is before
    assert "Cannot write Ascend layout trace" in caplog.text


def test_metadata_guard_allows_same_id_aliases_but_rejects_cross_id(helpers):
    storage = SimpleNamespace(data_ptr=lambda: 1000, nbytes=lambda: 96)

    def row(offset, width):
        return MetadataTensor(offset, (4, width), (width, 1), storage)

    layout = SimpleNamespace(
        layers=[(0, [row(32, 8), row(64, 8)]), (0, [row(0, 8), row(32, 8)])]
    )
    guard = helpers["_check_canonical_noninterference"]
    guard(layout, 4)
    layout.layers[1][1][1] = row(32, 16)
    with pytest.raises(ValueError, match="across canonical block IDs"):
        guard(layout, 4)


def test_common_metadata_guard_fast_paths_do_not_walk_block_rows(helpers):
    count = 10**12  # Metadata only: an accidental block scan would never finish.
    storage = SimpleNamespace(data_ptr=lambda: 1000, nbytes=lambda: 16 * count)
    first = MetadataTensor(0, (count, 8), (8, 1), storage)
    disjoint = MetadataTensor(8 * count, (count, 8), (8, 1), storage)
    shared = MetadataTensor(0, (count, 8), (8, 1), storage)
    layout = SimpleNamespace(layers=[(0, [first, disjoint]), (0, [shared])])
    helpers["_check_canonical_noninterference"](layout, count)
