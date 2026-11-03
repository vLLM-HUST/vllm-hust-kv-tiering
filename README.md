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
Its ECPA 0.3 manifest exclusively claims the process-local KV transfer
configuration and `hust_fs` secondary-tier registration. Installing the wheel
only makes the extension discoverable; enablement remains explicit, and
`runtime_effective` requires process-owned observer evidence.

## Ownership and provenance

This is the owner-maintained extraction of the tiered KV-cache residency and
lifecycle work from vLLM-HUST. See [MAINTAINERS.md](MAINTAINERS.md) and
[PROVENANCE.md](PROVENANCE.md) for ownership, attribution, and migration
history.

## Experimental Ascend qualification branch

This branch adds a **default-off** `ascend_copy_backend: torch_sync` adapter for
the frozen Qwen3.5 Frontier capsule (`vllm` distribution
`0.25.1+frontier.bidkv.empty`, core `d0f22d2bda562156e4dbf433ce645e1769b4f804`).
CUDA/XPU paths continue using the host worker.

The adapter now issues `copy_(..., non_blocking=True)` on a **dedicated NPU
stream** and records a per-job `torch.npu.Event`: `submit_*` returns
immediately, `get_finished()` polls the events, and `wait()` blocks on them.
Store copies are ordered after the compute stream (`wait_stream`) so they read
stable KV rather than racing the decode kernels; load completion is observed
through the event. Errors surfaced at query/synchronize time are reported as
`success=False` results instead of being raised out of the worker, and a job is
reported at most once (`get_finished` removes it before reporting).

Buffer ownership is documented: the worker holds long-lived *views* (not
copies) into the host mmap region and the runner's KV-cache tensors, and the
offloading manager pins the source/destination blocks for the duration of a
job, so those views remain valid until the job's event is observed complete.
The earlier revision (copy bracketed by `torch.npu.synchronize()`) was
synchronous and did not overlap device transfer with compute; that has been
replaced. **This remains an experimental adapter, not a production-default
optimization.**

## Evaluation status

Two claims are deliberately kept separate:

- **Save/store path is functional.** A real 910B2 partial-page two-group
  NPU→CPU→NPU round trip, segment-file creation, and preservation of untouched
  padding have been exercised. The connector, spec, `hust_fs` secondary tier,
  and `/dev/shm` offload region all initialize and run.
- **Recovery/reuse benefit is established (minimal recovery scenario).** Under
  the delivery configuration (Qwen3.5-35B-A3B, 262144 context, 26 GB/card KV)
  the natural `reuse_conversation` load reuses prefixes before HBM eviction, so
  it remains write-only with no read-back — the same observation as the
  Frontier review. A minimal recovery scenario that forces genuine cache
  pressure (a reduced KV budget plus repeated prefixes) exercises the load
  path: reusing an evicted prefix reads back from the FS tier instead of
  re-prefilling it. Attention KV and GDN state are both covered;
  resumed-progress correctness still needs model-level verification. Measured
  results and reproduction commands are in
  [`docs/serving-evaluation.md`](docs/serving-evaluation.md).

### Async rework measurement (2026-11, 32K smoke)

After the dedicated-stream/event rework, a 200-request `reuse_conversation`
run (Qwen3.5-35B-A3B, TP2, 32768 context, concurrency 8, tiering pressure
policy enabled) wrote segment files into the FS tier. At this configuration the
store path overlaps decode and adds no measurable cost. This is a 32K-context
smoke result, not the 262144-context delivery line, and it still observed
store-only traffic (no load), so the reuse benefit remains unestablished.

### 256K delivery-line + round-trip (2026-11)

On the 262144-context delivery config (explicit 26 GB KV budget per card),
pressure-aware offload triggered as expected: an 800-request run wrote segment
files into the FS tier, still with no read-back under natural load. An explicit
minimal-recovery round trip (`reset_prefix_cache` then re-request) confirmed
that reusing the FS-tier copy is faster than recomputing the prefix. Dataset,
config, and reproduction commands are in
[`docs/serving-evaluation.md`](docs/serving-evaluation.md); the measured results
are recorded in the canonical
[Frontier KV tiering evidence](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/FRONTIER-KV-TIERING-20260926.md).

### Known boundaries (verified, not assumed)

- **Load/reuse needs real HBM pressure.** With `tiering_reclaim_device_cache_after_store`
  the HBM copy is only *eligible* for eviction; it is not evicted until HBM is
  actually under pressure. A repeated-prefix request at low pressure hits the
  HBM prefix cache and does **not** read back from the FS tier. At 32K context
  the hybrid model's Mamba cache also imposes a floor: `gpu_memory_utilization`
  below ~0.62 is rejected (`max_num_seqs > available Mamba cache blocks`), so the
  KV budget cannot be shrunk far enough to force eviction. Establishing reuse
  therefore requires the 262144-context delivery line (or equivalent
  pressure), not a 32K smoke.
- **`/dev/shm` is a bounded tmpfs.** The CPU offload region is mmap'd under
  `/dev/shm` (`vllm_offload_*.mmap`). An 8 GiB `cpu_bytes_to_use` plus stale
  files from prior `kill -9` runs can fill the default 16 GiB mount and make
  the next `SharedOffloadRegion` fail with `madvise: Bad address`. Clean
  `/dev/shm/vllm_offload_*.mmap` after an abnormal shutdown, or provision a
  larger `/dev/shm` for bigger CPU tiers.

Provision the matched host separately and install this wheel normally; the
package has no runtime dependencies and does not replace the accelerator
runtime. Enable the connector directly through the vLLM KV-transfer config
(`cpu_bytes_to_use` is a positive integer and `root_dir` an absolute path):

```json
{
  "kv_connector": "HustAscendTieringConnector",
  "kv_connector_module_path": "vllm_hust_kv_tiering.ascend_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "HustTieringOffloadingSpec",
    "spec_module_path": "vllm_hust_kv_tiering.spec",
    "cpu_bytes_to_use": 1073741824,
    "ascend_copy_backend": "torch_sync",
    "secondary_tiers": [{
      "type": "hust_fs",
      "root_dir": "/var/lib/vllm-kv",
      "storage_layout": "segment",
      "n_read_threads": 2,
      "n_write_threads": 2
    }]
  }
}
```

Pass it to the server, for example via `--kv-transfer-config tiering.json`.
The connector does not change APC, MTP2, asynchronous scheduling, graph mode,
context capacity or parallelism. An existing conflicting `--kv-transfer-config`
is rejected. The installed host requires sufficient `/dev/shm` for the primary
buffer. Use a fresh storage directory for qualification.

Validation so far includes a real 910B2 partial-page two-group NPU→CPU→NPU
round trip and preservation of untouched padding. Full-model correctness,
actual cache reuse, performance, and recovery must pass before production use
or Frontier publication. Disable by dropping `--kv-transfer-config` and
restarting the owned server to return to the Native arm; uninstalling is a
separate package operation.

[`examples/ecpa-tiering.json`](examples/ecpa-tiering.json) is a minimal
non-production configuration for contract and clean-wheel validation.

The experimental Ascend profile uses a plugin-owned connector to pack logical
blocks from the runner's separate K/V and Mamba state views. CPU slots remain
shared according to the host allocation map; padding is not read as live state.
Packed allocations and layouts that require copying to form a view are rejected.
This addresses the Tensor-only registration failure found in the first full-model
Frontier attempt. Full-model qualification is still required before publishing
performance or treating this adapter as a supported release.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
