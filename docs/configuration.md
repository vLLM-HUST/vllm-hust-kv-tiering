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
| `tiering_min_session_requests_for_offload` | `1` | Minimum observed request count from session returns or matching content. |
| `tiering_min_reuse_probability` | `0.0` | Minimum historical session/content return fraction, in `[0, 1]`. |

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
The reuse estimate `(request_count - 1) / request_count` uses the larger of the
session request count and matching-content request count. Content observations
are keyed by the host's full offload key, survive anonymous request finalization,
and count a given request only once per key, including misses and rejected stores.
A batch uses its least-observed key so hot content cannot supply the history of
an unrelated cold key. The LRU history is bounded by `residency_max_entries`;
forgotten content starts cold. This is a historical hint, not a learned prediction.
No fixed unmeasured throughput numbers are filled in automatically.

### Optional candidate working-set admission

This opt-in policy limits large cold fills when recent candidate prefixes do
not fit the CPU pool. It borrows the working-set motivation of EfficientAgent,
but estimates a recent set of **actual host offload keys**, not task count times
a pure-attention bytes/token formula. It does not require LMCache or a different
model/runtime. It is not an exact predictor of future reuse or a complete
reimplementation of the paper.

| Option | Default | Meaning |
| --- | --- | --- |
| `tiering_working_set_aware` | `false` | Enable candidate-set tracking and request-level cold-fill rejection. |
| `tiering_working_set_window_sec` | `60.0` | Positive recency window for keys seen in lookup or store candidates. |
| `tiering_working_set_max_entries` | `64000` | Hard metadata bound for recent distinct keys. |
| `tiering_working_set_max_new_bytes` | `67108864` | New-write byte budget per request when the candidate set exceeds CPU capacity and insertion needs to displace existing/reserved slots. |

The byte estimate uses the same `kv_bytes_per_offloaded_block` geometry as the
host CPU allocator, including its mixed cache payload. Interpret it using that
host's rank/shared-memory geometry, not an assumed per-rank pure-attention beta.
The default 64 MiB is a configurable policy budget, **not** a calibrated optimum
for Qwen3.5 or a measured bandwidth. Check the actual block size and choose a
budget that can admit the desired whole checkpoint bundles.

Repeated callbacks and shared keys do not enlarge the distinct set. Full key
identities, including cache-group and salt isolation, remain unchanged. If the
metadata bound is exceeded, the snapshot marks the window `censored`; the
retained set is only a lower bound, and large fills under capacity displacement
are conservatively rejected until the lost observations age out. A small
metadata bound can therefore reject useful writes; size and report it explicitly.

When free slots cover the candidate, normal admission remains available. When
the recent set fits capacity, a full cache alone does not trigger rejection.
Under over-capacity pressure, small appends remain eligible; cumulative bytes
**submitted** by a request count toward its budget, including failed writes, so
chunked prefill cannot bypass it. Such appends can still cause churn; the policy
does not guarantee that every admitted write will be useful.

A working-set rejection is sticky for that request until asynchronous
finalization, including cache reset/preemption continuation. It does not mean
"retry after one second". New requests are evaluated again. The manager never
advances the host's stored prefix index and never filters/splits an offered
hybrid bundle. After the first lifetime rejection, the plugin's thin store
scheduler hook omits that request from subsequent store candidate construction.
The original scheduler output still drives loads, completion polling, request
updates, preemption and transfer flushes. No stored prefix index is advanced to
pretend a rejected write succeeded. Transient budget/pressure refusals remain
retryable. General host request bookkeeping and hashing still occur.

Existing/in-flight CPU copies are passed back to the host unchanged, even if
new-write budgets or request rejection would block allocation. They are not
charged as fresh D2H writes; the host owns readiness, deduplication and progress.
This manager-level rule does not require rebuilding store candidates for an
already lifetime-rejected request. Other requests can still reuse its ready
copies, and already submitted jobs complete through the original host path.
History/cost gates evaluate missing copies, so an already-present cold key does
not suppress a different repeatedly observed missing key. Mixed *missing* hot
and cold keys still use conservative least-observed content history.

Use [working-set-tiering.json](<../examples/working-set-tiering.json>) for an
explicit CPU-only candidate arm; compare it with the same JSON with
`tiering_working_set_aware: false`. Keep Native's APC, model, fixed host version,
actual HBM configuration and other shared optimizations unchanged. Do not
invent throughput calibrations or disable protocol-required salt isolation.
This example defines an experimental policy profile, not a performance result.
Record matched-host, real-device and model recovery qualification separately
for the exact source revision and runtime; portable tests alone do not qualify it.

### Distinguish lookup, transfers and effective recovery

`get_policy_snapshot()` now includes `working_set`, `store_skipped_requests`,
`pipeline_blocks` and `pipeline_bytes`. Stage counters distinguish:

- `lookup_attempt_blocks`: per-key queries, including repeated callbacks;
  `lookup_hit_blocks`, `lookup_miss_blocks`, `lookup_pending_blocks` and
  `lookup_rejected_blocks` are their outcomes, not unique prefixes.
- `store_submitted_blocks`, `store_completed_blocks`, `store_failed_blocks`:
  device-store admission versus scheduler-observed successful/failed completion.
- `load_submitted_blocks`, `load_completed_blocks`: device-load submission
  versus the host's completion callback. Tentative FS-to-CPU promotions are not
  counted as completed device restores.
- `primary_evicted_blocks`: keys evicted by delegate device-store preparation,
  excluding TTL deletion and explicit maintenance reclamation.

The metrics are `vllm:kv_tiering_pipeline_blocks{stage=...}` and
`vllm:kv_tiering_candidate_working_set_{blocks,bytes,censored}`. Counters are
exported as deltas without counting repeated stats collection twice. Byte
snapshots use block count times host payload size; lookup outcomes have no byte
volume, and partial failed transfers are not measured as successful traffic.
These are nominal payload sizes at each stage, not independently measured DMA
traffic or bandwidth. The snapshot also records CPU capacity/free slots and
`host_payload_bytes_per_block` to make its geometry auditable.

Neither a HIT nor a successful load callback establishes a usable full mixed
prefix, accepted external tokens, or saved prefill. Those require matched-host
scheduler/model evidence. Do not compute "reuse hit rate" from D2H/H2D byte
ratios. Automatic bandwidth calibration, a profitability-based load rejection
hook and adaptive low-yield shutdown are **not implemented** by this change;
the existing explicit calibrated store-cost gate remains optional.

Maintenance on request arrival now respects the configured interval rather
than forcing an O(cache size) scan for every new request. Transfer completion
polling at `begin_step` is still unconditional. TTL maintenance remains
scheduler-driven and best effort; no wall-clock erasure guarantee is introduced.

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
must be accounted for before finalization. Finalization always releases request
bookkeeping. With positive TTL, anonymous idle ownership remains until expiration
just like stable-session ownership. With TTL disabled, anonymous ownership is
released immediately; metadata pressure may forget finalized idle stable-session
hints without deleting their files. Pending TTL ownership is never discarded to
make metadata room: at the session cap new cache tracking is refused until space
is available. Active ownership is never silently evicted.

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
