# vllm-hust-kv-tiering

Out-of-tree KV-cache tiering, filesystem persistence, and lifecycle-management
work for **vLLM-HUST v1** (`vllm 0.28.1rc0`).  It does not patch the vLLM
source tree: v1's documented `spec_module_path` and secondary-tier loading
contracts are the integration boundary.

## Features

- pressure-aware device → CPU → filesystem placement;
- segment-file + SQLite index layout, with batched `preadv` reads;
- asynchronous read/write queues with read priority;
- lifecycle/residency components for session retention, TTL expiry and CPU
  demotion policy;
- a v1-compatible `HustTieringOffloadingSpec` entry point.

## Installation

Install this package into the same environment as the vLLM-HUST v1 release:

```bash
pip install -e .
```

Select the out-of-tree spec in the offloading connector configuration:

```json
{
  "spec_name": "HustTieringOffloadingSpec",
  "spec_module_path": "vllm_hust_kv_tiering.spec",
  "cpu_bytes_to_use": 1073741824,
  "secondary_tiers": [{
    "type": "SegmentFileSystemTier",
    "module_path": "vllm_hust_kv_tiering.fs.manager",
    "root_dir": "/var/lib/vllm-kv",
    "storage_layout": "segment"
  }]
}
```

`type: hust_fs` may be used after the package's general plugin is loaded. The
explicit `module_path` form above is preferred because it is independent of
plugin loading order.

## Compatibility

This package is intentionally pinned to the HUST v1 ABI. It must be tested
again before upgrading vLLM, because offloading interfaces are experimental.
