# vllm-hust-kv-tiering

Development version: `0.2.0.dev0` (pressure/lifecycle safety rework; not a qualified release).

Out-of-tree, explicitly enabled KV prefix/state-cache tiering for the frozen
**vLLM-HUST v1** host at core
`d0f22d2bda562156e4dbf433ce645e1769b4f804` (`d0f22d`). The matched distribution
line used for CPU integration validation is `0.25.1+frontier.bidkv`.
The historical `.empty` capsule string remains in the ECPA discovery manifest;
that distribution is not separately hardware-qualified here. This is **not** a
compatibility claim for upstream vLLM or `0.28.x`.

The package adds **zero runtime dependencies**. Provision the matched host,
PyTorch and accelerator runtime separately. Installing this package does not
install or replace them, start a server, or enable offloading automatically.

## Implemented behavior

- Device → CPU cache admission driven by scheduler block-pool pressure, with
  high/low hysteresis, preemption signals, bounded jobs and whole-request budgets.
- CPU → filesystem spill and CPU reclamation based on occupancy or idle age.
  A CPU copy is not reclaimed for a spill until a secondary write succeeds.
- Session/shared-prefix ownership, request-versus-transfer finalization, optional
  idle TTL, bounded metadata and asynchronous optional secondary deletion.
- Segment files plus a SQLite index, bounded I/O admission, configurable read/write
  threads, a payload quota, per-block CRC32/short-I/O checks and explicit durability modes.
- An opt-in Ascend stream/event worker with fail-fast transfer fences and
  conservative drain/teardown. CUDA uses the matched host's worker.

These policies are wired into the runtime; they are not merely accepted options.
They do **not** enlarge the active model's HBM working set or explicitly evict
host GPU prefix-cache blocks. A rejected cache admission normally falls back to
native inference/recomputation. An unproven device fence is fatal, not a safe
cache miss.

See [architecture](<docs/architecture.md>) for ownership and policy invariants,
and [configuration](<docs/configuration.md>) for options and operating boundaries.

## Install and enable

Install into the environment that already contains the matched host:

```bash
python -m pip install --no-deps .
```

Use the generic plugin-owned `HustTieringConnector`, which binds the scheduler's
pressure source while preserving host transfer scheduling. The module name
retains its historical Ascend spelling, but the connector also supports the
host CUDA path. Supplying only a spec to a different connector is not a substitute
for this pressure binding.

The complete [Ascend CPU+filesystem example](<examples/pressure-tiering.json>)
selects `ascend_copy_backend: torch_async` for the target 910B runtime and uses:

- Composite HBM cache-pressure high/low: `0.8` / `0.6`;
- CPU occupancy high/low: `0.9` / `0.7`;
- idle TTL: `300` seconds; secondary deletion: `false`;
- CPU demotion age: `60` seconds;
- 1 GiB CPU budget and a 64 GiB segment-payload quota;
- two read-priority and two write-priority threads, with 64 admitted FS jobs.

Review the absolute storage path, memory/filesystem budgets and permissions before
starting an owned server:

```bash
vllm serve YOUR_MODEL --kv-transfer-config "$(cat examples/pressure-tiering.json)"
```

Use the [CPU-only example](<examples/cpu-only.json>) for the device→CPU comparison
arm; it has `secondary_tiers: []`. Here “CPU-only” describes the offload tier,
not CPU model execution. The Native arm omits the KV-transfer configuration.
Do not combine this configuration with a conflicting connector configuration.

### Ascend opt-in

Both examples explicitly select the matched Ascend runtime with this entry
inside `kv_connector_extra_config`:

```json
"ascend_copy_backend": "torch_async"
```

`torch_sync` is a supported legacy alias for the same stream/event adapter; it
does **not** select a separate synchronous implementation.
`HustAscendTieringConnector` remains a connector-name alias. For a separately
qualified host CUDA/XPU deployment, **remove** `ascend_copy_backend` to use the
host's worker; omitting it on an NPU host does not select the Ascend adapter.
Unsupported cache layouts fail explicitly rather than silently copying a
guessed layout.

Requesting `copy_(..., non_blocking=True)` for mmap-backed CPU memory does not
prove asynchronous overlap. No host-pinning heuristic is applied. Copy, event
record/query/wait errors stop normal worker operation; failed fences raise and
never return a fabricated successful completion. Shutdown retains views and the
mmap if it cannot prove that the transfer stream has stopped, and can be retried.

## Compatibility and safety boundary

The integration is deliberately version-bound. The host supplies factories,
connector protocols, request scheduling, CPU allocation and transfer bookkeeping.
Selective CPU reclamation additionally uses a small **private-host adapter**;
not every accessed interface is public. Re-run actual-host ABI and transfer tests
before changing the host, PyTorch, accelerator runtime, cache layout or policy.

The packaged ECPA manifest currently pins the exact
`0.25.1+frontier.bidkv.empty` distribution string. Its discovery constraint is
narrower than the matched source-ABI line described above. Wheel discovery is
not runtime qualification or evidence of activation.

CPU mmap capacity and filesystem capacity are separate resources. Provision
sufficient `/dev/shm` for the host offload regions and other consumers. After an
abnormal exit, identify the owning process before reclaiming stale resources;
never delete a live process's mmap or cache directory. Use a separate storage
namespace for incompatible models/configurations and qualification runs.

## Tests and qualification

Portable tests require neither torch nor vLLM and do not initialize hardware:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/test_policy.py tests/test_lifecycle_state.py \
  tests/test_async_worker_safety.py tests/test_segment_store_safety.py
```

The full suite skips host-dependent modules when their dependencies are absent.
The original real-NPU tests are preserved and additionally require explicit
operator consent on a suitable, isolated device:

```bash
VLLM_KV_TIERING_RUN_NPU_TESTS=1 python -m pytest -m npu \
  tests/test_ascend_worker.py tests/test_ascend_layout.py
```

Portable safety tests, wheel imports and frozen-host CPU integration tests do not
establish real-device correctness, serving stability, crash recovery, overlap or
performance. The current pressure/lifecycle policy changes are **not hardware-
qualified**. This README makes no measured throughput, latency or reuse-benefit
claim. Historical evaluation notes are not qualification of the current code.
Compare Native, CPU-only and CPU+filesystem under the same frozen workload;
filesystem completed-read evidence is required to claim filesystem recovery.

## Rollback and ownership

Remove `--kv-transfer-config` and restart the owned server to return to Native;
uninstallation is a separate operation. Do not abandon a process holding an
unproven transfer fence and reuse its storage in another live process.

See [maintainers](<MAINTAINERS.md>), [provenance](<PROVENANCE.md>) and
[MOD metadata](<MOD_METADATA.json>) for ownership, attribution and historical
qualification labels. The [ECPA example](<examples/ecpa-tiering.json>) is a
minimal contract fixture, not the bounded deployment example above.
