# Configuration and deployment boundaries

## Matched host and activation

This package targets frozen vLLM-HUST v1 core
`d0f22d2bda562156e4dbf433ce645e1769b4f804`, distribution line
`0.25.1+frontier.bidkv` in the CPU integration environment. The historical
`.empty` capsule distribution is not separately hardware-qualified here.
It is not a general upstream-vLLM compatibility layer. The packaged discovery
manifest currently names exactly `0.25.1+frontier.bidkv.empty`; source-ABI support
and manifest version matching are distinct checks. Re-test before upgrading the
host, runtime, cache layout or plugin.

For the paired Ascend commit `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`,
its existing version-routing code needs `VLLM_VERSION=0.25.1` to select the
available FLA implementation with this local-version core. This is a compatibility
routing override, not proof of the installed distribution; record the actual
`importlib.metadata` version separately. Preserve the inherited CANN
`PYTHONPATH` and library paths when prepending this plugin's source path:
overwriting `PYTHONPATH` hides the host-provided `acl` bindings and breaks model
startup. Installed-wheel deployments should not need a source-path override.

**Known frozen-host model gate:** real Native (no connector) testing of
Qwen3.5 MTP2 + Mamba `align` reached readiness but failed on its first inference:
the core calls the fused Mamba postprocess kernel with a newer ABI than the
Ascend replacement (`block_size` bound twice). No host patch or kernel
monkeypatch is supplied by this plugin. A no-MTP diagnostic does not qualify
that MTP2 deployment profile; resolve the host pairing before production use.

Runtime dependencies are deliberately empty. The host supplies torch, accelerator
support and vLLM. The `dev` extra only installs test/build/lint tooling. No install
step enables the plugin or starts a server.

Use the complete [Ascend CPU+filesystem profile](<../examples/pressure-tiering.json>)
or [Ascend CPU-only profile](<../examples/cpu-only.json>) as a starting point:

```bash
vllm serve YOUR_MODEL --kv-transfer-config "$(cat examples/pressure-tiering.json)"
```

The frozen host's CLI expects an **inline JSON string**, not the JSON filename;
the shell substitution above supplies the file contents. Directly passing
`--kv-transfer-config examples/pressure-tiering.json` is rejected by this host's
actual parser. This can be checked without starting a model.

Both use `HustTieringConnector` from
`vllm_hust_kv_tiering.ascend_connector`, role `kv_both`, and the plugin spec
`HustTieringOffloadingSpec` from `vllm_hust_kv_tiering.spec`. This generic connector
binds the pressure source and selects the appropriate registration path. The
historical `HustAscendTieringConnector` name remains an alias. Do not install a
second competing connector configuration.

The example's `ascend_copy_backend: torch_async` explicitly selects the NPU
stream/event worker. `torch_sync` is a legacy alias for the **same** worker.
For a separately tested host CUDA/XPU deployment, omit `ascend_copy_backend` to
retain the host worker. Omitting it on Ascend does not auto-detect/select this
NPU adapter. “CPU-only” means NPU↔CPU offload without a filesystem secondary,
not CPU model execution.

The Ascend logical builder validates known runner forms rather than inferring
all semantics from shape divisibility. Non-MLA `FullAttentionSpec` caches require
an explicit two-view K/V tuple/list; a combined planes-first tensor is rejected,
even when two blocks make its first dimension appear valid. Mamba views must
match the exact declared state count, `(num_blocks, *state_shape)` and dtype;
a single tensor is accepted only for an exact one-state spec. MLA has a separate
contract and is not newly qualified by these checks. The builder never silently
transposes/packs data to fit. The generic host canonical/CUDA worker is unchanged.

## Budget the resources first

The examples specify 1 GiB (`1073741824`) of host CPU offload memory. Provision
sufficient `/dev/shm` for the matched host's complete mmap geometry, ranks and
other processes; the JSON budget is not a host-wide resource reservation. A
failed allocation is not permission to delete an unrelated process's mmap.

The filesystem example uses the absolute root
`/var/lib/vllm-kv/qualification-910b-v1`. Change it to an owned, writable namespace.
Use immutable model/tokenizer/adapter revisions and a **fresh cold-cache root**
when changing model contents, cache geometry, quantization, runtime or
qualification arm. A model path that is overwritten in place is not a reliable
content identity. Existing metadata/config checks do not promise detection of
every such change. Do not share an active writer namespace between unrelated
servers. Compare warmed and cold runs explicitly; do not attribute an old cache
hit to the new configuration.

The 64 GiB (`68719476736`) `max_storage_bytes` in the example is a segment payload
reservation quota for each store namespace/rank, **not** a global disk limit. It
includes failed/in-flight segment reservations, but excludes SQLite/WAL/config
files and legacy per-block files. Reserve disk headroom and monitor aggregate
rank usage. Fully dead segments can be reclaimed; partially live segments are
not compacted. Exhaustion fails cache writes safely; it does not evict live
persistent entries to manufacture free capacity. Reopening a namespace with a
different persisted quota/durability setting is rejected.

## Policy options

Options below live in `kv_connector_extra_config`. Booleans must be JSON
`true`/`false`, not strings or integers. Numeric policies require finite JSON
numbers; count limits require positive integers (booleans/fractions are not
counts). Unsupported outer options, including misspelled `tiering_`,
`lifecycle_` and `residency_` names, fail rather than silently accepting an
unimplemented policy. Legacy `pressure_aware` and inert host `max_tracker_size`
are not accepted aliases; `store_threshold` only accepts 0 or 1 (no filtering). Watermarks satisfy
`0 < low <= high <= 1`; supply both when lowering the high watermark below a
larger default low watermark.

### Device→CPU admission

| Option | Default | Meaning |
| --- | --- | --- |
| `cpu_bytes_to_use` | Required by host | Positive byte budget for the CPU offload pool; use an integer. |
| `offload_prompt_only` | `true` | Host option: offload newly computed prompt blocks; `false` also permits decode blocks. It is not a phase-specific scheduler rewrite. |
| `tiering_hbm_pressure_aware` | `true` | Gate construction of new device-store jobs on cache pressure. |
| `tiering_hbm_high_watermark` | `0.8` | Open the pressure gate at/above this fraction. |
| `tiering_hbm_low_watermark` | `0.6` | Close the gate at/below this fraction. |
| `tiering_max_device_store_blocks_per_step` | `128` | Soft block budget: the first whole request can exceed it to preserve hybrid-group progress. |
| `tiering_max_inflight_device_store_jobs` | `8` | Hard concurrent device-store admission limit. |
| `tiering_min_session_requests_for_offload` | `1` | Minimum observed request count for a tracked session. |
| `tiering_min_reuse_probability` | `0.0` | Minimum historical session-return fraction, in `[0, 1]`. |

The connector samples the matched block pool in O(1), taking the maximum of
allocation pressure (`BlockPool.get_usage()`) and the **cached-key pressure proxy**
`len(pool.cached_block_hash_to_block) / (pool.num_gpu_blocks - 1)`, clamped to
`[0, 1]`. This is the primary lookup index, fetched anew on each sample after any
cache reset; the frozen host implements its key count in O(1). The similarly named
`cached_block_hashes_by_block` contains only supplemental aliases and must not
be used to count ordinary cached prefixes.

This detects ordinary evictable cached keys at low active concurrency. It is
**not exact physical block occupancy**: alias keys may overcount physical blocks,
while duplicate blocks sharing a key may undercount them. Consequently it is not
an exact resident union, a guaranteed union lower bound, physical accelerator
memory utilization, or general HBM telemetry. Preemption remains an additional
pressure signal. Missing pressure binding/primary-index ABI support is a
compatibility error, not a fabricated low-pressure reading.

The host still controls GPU block allocation and eviction. Store admission
postpones work without skipping the host's prefix progress indices. It does not
promise to save every prefix or make an arbitrarily large active attention
working set fit. Ownership/capacity rejection degrades to native cache miss or
recomputation, not incorrect KV.

### CPU→filesystem placement and maintenance

| Option | Default | Meaning |
| --- | --- | --- |
| `tiering_secondary_pressure_aware` | `true` | Defer ordinary CPU→FS cascades until CPU pressure; explicit maintenance spills remain possible. |
| `lifecycle_cpu_high_watermark` | `1.0` | CPU occupancy high watermark; example uses `0.9`. |
| `lifecycle_cpu_low_watermark` | `1.0` | CPU occupancy low watermark; example uses `0.7`. |
| `lifecycle_cpu_demote_after_sec` | `0.0` | Idle-age threshold for CPU spill/reclamation; zero disables this trigger; example uses `60`. |
| `tiering_max_spill_jobs` | `4` | Hard bound on plugin maintenance spill jobs. |
| `lifecycle_reclaim_batch_size` | `64` | Bounded maintenance candidate/delete batch size. |
| `tiering_maintenance_interval_sec` | `0.05` | Positive scheduler-activity maintenance interval, not a dedicated wall-clock timer. |
| `tiering_retry_backoff_sec` | `1.0` | Positive retry backoff after failed cache maintenance. |

CPU occupancy includes both reserved and resident slots. Cold CPU copies remain
pinned until their secondary write completes. Safe reclamation rechecks host
readiness/refcounts, not only metadata. With `secondary_tiers: []` there is no FS
spill/recovery path; keep that arm separate from CPU+FS when measuring benefits.

### Optional calibrated cost gate

| Option | Default | Meaning |
| --- | --- | --- |
| `tiering_prefill_tokens_per_second` | `0.0` | User-calibrated prefill throughput. |
| `tiering_restore_bandwidth_bytes_per_second` | `0.0` | User-calibrated effective restore bandwidth. |
| `tiering_store_bandwidth_bytes_per_second` | `0.0` | User-calibrated effective store bandwidth. |
| `tiering_benefit_safety_factor` | `1.2` | Cost margin, finite and at least `1`. |

All three rates must be zero (disabled) or all positive. These are supplied
calibrations, not online measurements. Admission compares historical reuse
fraction × (recompute cost − restore cost) against write cost × safety factor.
The reuse estimate `(request_count - 1) / request_count` is a historical session
return fraction, not a learned prediction. No fixed unmeasured throughput numbers
are filled in automatically.

## Session ownership and TTL

| Option | Default | Meaning |
| --- | --- | --- |
| `lifecycle_idle_ttl_sec` | `0.0` | Retain idle session hints for this many seconds; zero disables TTL; example uses `300`. |
| `lifecycle_delete_expired_secondary` | `false` | Opt in to asynchronous deletion of expired, unreferenced secondary keys. |
| `lifecycle_max_sessions` | `4096` | Hard session metadata admission bound. |
| `residency_max_entries` | `64000` | Hard block ownership metadata budget, also conservatively constraining deletion backlog. |
| `residency_tracking_enabled` | `false` | Standalone helper default; the spec enables internal ownership tracking for policy safety regardless. |

Stable session identifiers are read from request `kv_transfer_params` as
`session_id`, `conversation_id`, or `kv_session_id`; otherwise the request ID is
ephemeral. Concurrent requests in one session and shared prefix keys are tracked
without releasing another active owner's protection. Request finish does not
mean transfer finish. Deferred promotions, device copies and secondary jobs
must be accounted for before finalization. Ephemeral state is removed on final
completion; metadata pressure may forget finalized idle stable-session hints,
but it does **not** implicitly delete their files or evict active ownership.

TTL is checked during scheduler activity and new-request/lookup processing.
Idle engines do not spin until the deadline, so TTL is not a secure-erasure SLA.
With the default `lifecycle_delete_expired_secondary: false`, expiry can release
CPU retention/ownership but preserves FS copies. When deletion is enabled,
pending keys are tombstoned against lookup, touch and new publication; errors
and partial results retain unacknowledged keys for bounded retry. New cache
admission can be refused while deletion backlog consumes the metadata budget.

## Filesystem secondary options

These options belong to each object in `secondary_tiers`, not the outer extra
config. Set `type: hust_fs`; the plugin spec registers this factory alias before
constructing tiers, independently of general-plugin discovery order. Do not add
`module_path` to the secondary-tier object: the frozen factory forwards unknown
fields to the tier constructor instead of using them as loader directives.

| Option | Default | Meaning |
| --- | --- | --- |
| `root_dir` | Required | Absolute owned cache namespace; use a fresh namespace for incompatible runs. |
| `storage_layout` | `segment` | Indexed segments; legacy `file` is available but does not support quota/fsync options. |
| `segment_size_bytes` | `1073741824` | Maximum segment-object size. |
| `segment_io_batch_blocks` | `0` | Zero selects approximately 1 MiB per I/O task. |
| `segment_shards` | `0` | Zero selects up to 16 append streams based on thread count. |
| `max_storage_bytes` | `null` | Unbounded if omitted; a finite positive payload quota is strongly recommended. |
| `durability` | `none` | Disposable-cache default; `fsync` selects stronger publication ordering. |
| `integrity` | `crc32` for segment; `none` for legacy file | Per-block accidental-corruption checks. `none` explicitly disables content checks; legacy file layout cannot enable CRC. |
| `n_read_threads` | `2` | Read-priority threads, integer `1..64`. |
| `n_write_threads` | `2` | Write-priority threads, integer `1..64`. |
| `max_pending_jobs` | `64` | Admitted FS jobs across queued, running and completed-but-unconsumed states. |

Both thread groups can drain work, with read priority for restore traffic. Job
queue saturation enqueues no new I/O and reports a failed secondary result so
the host can release its reservation. This is separate from fatal device-fence
failure and must never be interpreted as a successful persistent copy.

`durability: none` is a best-effort disposable cache, with no stable-media
promise. `fsync` orders file/directory flushing before SQLite FULL publication;
it does not create a replicated backup or guarantee behavior of faulty storage.
Quota, durability and integrity settings must match when reopening a namespace.
CRC32 checks source/destination block memoryviews without a block-sized copy;
a checksum mismatch fails the promotion and invalidates only the exact bad
index location. It detects accidental corruption, not malicious modification or
all possible faults. `integrity: none` cannot detect same-sized data corruption.
Existing nonempty, unchecksummed indexes are refused in CRC mode: use a fresh
namespace rather than trusting old bytes. CRC mode never falls back to unverified
legacy per-block files. Checksum overhead is part of the measured store/load
cost, not assumed to be free. Qualify crash recovery and performance separately
for the selected filesystem and hardware.

## Validation and rollback

The [README](<../README.md>) lists portable and opt-in NPU test commands. CI uses
Python 3.10/3.12/3.14 without torch/vLLM, builds a wheel, and imports that wheel in
a fresh no-dependency virtual environment. Frozen-host CPU tests validate the
actual private adapter and host bookkeeping; they do not qualify real serving.
Hardware tests run only with `VLLM_KV_TIERING_RUN_NPU_TESTS=1` plus the matched
runtime and available NPU.

The implemented pressure/lifecycle changes remain unqualified on production
hardware. Do not infer true asynchronous mmap overlap from `non_blocking=True`,
or FS recovery from device→CPU/CPU→device counters alone. Compare Native,
CPU-only and CPU+FS under identical immutable model and request contracts.

On a device/event failure, fences raise; an undrained worker retains views/mmap
and refuses normal work. Retry safe shutdown rather than forcibly unmapping
active DMA. To disable the plugin, remove the KV-transfer configuration and
restart the owned server. Uninstalling the wheel is separate from stopping and
draining the active process.
