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

## Ownership and provenance

This is the owner-maintained extraction of the tiered KV-cache residency and
lifecycle work from vLLM-HUST. See [MAINTAINERS.md](MAINTAINERS.md) and
[PROVENANCE.md](PROVENANCE.md) for ownership, attribution, and migration
history.

## Experimental Ascend qualification branch

This branch adds a default-off `ascend_copy_backend: torch_sync` adapter for the
frozen Qwen3.5 Frontier capsule (`vllm` distribution
`0.25.1+frontier.bidkv.empty`, core `d0f22d2bda562156e4dbf433ce645e1769b4f804`).
It uses PyTorch NPU/CPU copies with explicit synchronization and the host's
canonical KV-group/sub-block metadata. CUDA/XPU paths continue using the host
worker. This is a correctness-first synchronous adapter, not a claimed fast
DMA implementation. Its serving qualification and performance are pending.

Provision the matched host separately and install this wheel with `--no-deps`;
the package no longer asks pip to replace the accelerator runtime. Use the
Extension Manager with an experiment-specific `VLLM_HUST_EXT_CONFIG`:

```bash
vllm-hust-ext extension validate org.vllm-hust.kv-tiering
vllm-hust-ext extension configure org.vllm-hust.kv-tiering --file tiering.json
vllm-hust-ext extension check org.vllm-hust.kv-tiering
vllm-hust-ext extension enable org.vllm-hust.kv-tiering
vllm-hust-ext run --dry-run -- vllm serve MODEL [unchanged Frontier arguments]
vllm-hust-ext run -- vllm serve MODEL [unchanged Frontier arguments]
```

Configuration supplies `cpu_bytes_to_use` (positive integer) and
`storage_directory` (absolute path). The provider renders the explicit
`OffloadingConnector` spec and segment-file secondary tier; it does not change
APC, MTP2, asynchronous scheduling, graph mode, context capacity or parallelism.
An existing conflicting `--kv-transfer-config` is rejected. The installed host
requires sufficient `/dev/shm` for the primary buffer. Use a fresh storage
directory for qualification. Manager `enabled` means launch intent, not
`runtime_effective` or a measured improvement.

Validation so far includes a real 910B2 partial-page two-group NPU→CPU→NPU
round trip and preservation of untouched padding. Full-model correctness,
actual cache reuse, performance, and recovery must pass before production use
or Frontier publication. Disable through `vllm-hust-ext extension disable`
and restart the owned server to return to the Native arm; uninstalling is a
separate package operation.

The experimental Ascend profile uses a plugin-owned connector to pack logical
blocks from the runner's separate K/V and Mamba state views. CPU slots remain
shared according to the host allocation map; padding is not read as live state.
Packed allocations and layouts that require copying to form a view are rejected.
This addresses the Tensor-only registration failure found in the first full-model
Frontier attempt. Full-model qualification is still required before publishing
performance or treating this adapter as a supported release.
