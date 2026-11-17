# vllm-hust-kv-tiering

An installable, explicitly enabled KV prefix/state-cache tiering plugin for the
version-bound vLLM-HUST v1 host. Development version: `0.2.0.dev0`.

The plugin supports **NPU HBM ↔ CPU memory ↔ filesystem storage**. Filesystem
storage is optional: `secondary_tiers: []` selects the HBM/CPU path. It caches
reusable prefixes and hybrid recurrent checkpoints; active inference working-set
allocation and scheduling remain owned by the host.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/vllm_hust_kv_tiering/` | Connector, admission policy, lifecycle, transfer worker and filesystem implementation |
| `tests/` | Portable, matched-host and explicitly enabled NPU test source |
| `examples/` | Plugin activation/configuration examples |
| `scripts/` | Parameterized replay input preparation and client source |
| `docs/` | Design, compatibility, options and reproduction instructions |
| `runtime-patches/` | Explicit source patches for the exact reference SDK bases |
| `pyproject.toml`, `MANIFEST.in` | Independent Python packaging |

Generated inputs, logs, result files and evidence archives belong outside this
repository. Publish experimental records through
[vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark).

## Implemented behavior

- Device-to-CPU admission uses HBM cache-pressure hysteresis and bounded jobs.
- Optional bounded working-set admission limits cold fills that would displace
  existing CPU cache slots. Request rejection persists until finalization.
- Complete Attention/GDN bundles retain host alignment and prefix progress.
- CPU-to-filesystem spill and logical CPU reclamation account for occupancy,
  idle age and successfully completed secondary writes.
- Shared-prefix/session ownership, transfer finalization, optional idle TTL,
  bounded residency metadata and optional secondary deletion are tracked.
- Segment storage and a SQLite index provide bounded I/O, quotas, CRC32 and
  configurable durability. Completed secondary copies can be restored through CPU.
- The opt-in Ascend stream/event worker retains transfer ownership until fences
  complete and reports fence errors explicitly.

A rejected cache admission permits recomputation. These mechanisms do not promise
that every workload benefits or that an arbitrarily large active attention
working set fits in HBM. See [architecture](docs/architecture.md) and
[configuration](docs/configuration.md).

## Install and enable

Provision the matched vLLM-HUST, PyTorch and accelerator runtime first. The plugin
has no mandatory runtime dependencies and does not install/replace the host:

```bash
python -m pip install --no-deps .
```

The reference Qwen3.5 hybrid path requires core
`3bf4ba0b0ee547577b4e6bb772ccd45aa5793670` and Ascend
`b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9`, or a separately validated equivalent.
See [SDK prerequisites](runtime-patches/README.md) and
[hybrid host safety](docs/ascend-hybrid-host-safety.md). Source patches are never
applied by installing or importing this plugin. Upstream vLLM and arbitrary
newer host versions are not implicitly compatible.

For the three-tier Ascend path, edit the filesystem namespace and budgets in
[pressure-tiering.json](examples/pressure-tiering.json), then start:

```bash
vllm serve YOUR_MODEL \
  --kv-transfer-config "$(cat examples/pressure-tiering.json)"
```

The example selects `HustTieringConnector`, the plugin spec and the explicit
`ascend_copy_backend: torch_async` worker. `torch_sync` is a legacy alias for the
same adapter. A spec attached to another connector does not supply the required
pressure binding. CUDA uses the matched host worker when the Ascend option is
omitted; it requires separate runtime qualification.

For two tiers, use [cpu-only.json](examples/cpu-only.json). Native omits the
transfer configuration. The [SWE replay profile](examples/swe-prefix-replay.json)
is a separate two-tier setup with a larger CPU budget and different lifecycle
settings; it is not the default or a three-tier performance result.

## Reproduce the experiment configuration

The [Chinese reproduction guide](docs/reproduction.zh-CN.md) specifies hardware,
model/runtime versions, pinned public input references, deterministic task
selection, server commands, client commands, all policy/lifecycle values and
metric definitions. It separates the two-tier SWE replay from three-tier
functional checks. Input preparation and replay source write outputs only to a
new external directory. Their tool dependencies do not become plugin runtime
dependencies. Actual speedup requires measurement on the chosen workload.

See the [evaluation and applicability index](docs/evaluation-index.zh-CN.md) for
central evidence custody, version scope and the five-dataset status.

## Tests

Portable tests do not need vLLM or accelerator hardware:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests -m 'not host and not npu'
```

The suite is grouped by component rather than by the change that introduced a
regression:

- Portable: policy/working-set admission, lifecycle/residency, manager admission
  contracts, asynchronous worker safety, segment storage, and host/layout guards.
- Matched host: tiering manager lifecycle and cache-policy invariants,
  filesystem factory/I/O, connector/scheduler pressure, and Ascend layout/copies.

Shared setup lives with its component's tests. Host-only modules skip before
importing optional dependencies; device round trips retain the separate NPU
marker and explicit activation guard.

Matched-host tests require the pinned host and its real dependencies. NPU tests
additionally require an isolated device and explicit activation:

```bash
VLLM_KV_TIERING_RUN_NPU_TESTS=1 python -m pytest -q -m npu \
  tests/test_ascend_layout.py
```

Passing portable or CPU/FS contract tests does not establish full-model hybrid
correctness, asynchronous overlap or performance. See
[validation boundaries](docs/validation-status.md) and
[serving protocol](docs/serving-evaluation.md).

## Compatibility and rollback

The host owns cache geometry, scheduling and transfer bookkeeping. Selective CPU
reclamation additionally uses a small private-host adapter. Revalidate its ABI
when upgrading the host. Provision CPU mmap capacity and IPC headroom separately
from filesystem quota, and use a new storage namespace for incompatible models.

The ECPA discovery manifest retains its historical exact
`0.25.1+frontier.bidkv.empty` constraint. Discovery is not runtime activation or
qualification; direct connector activation uses the explicit runtime above.

To return to Native, remove `--kv-transfer-config` and restart the owned service.
Drain transfers before reusing resources; do not remove another live process's
shared memory or filesystem namespace.

See [maintainers](MAINTAINERS.md), [provenance](PROVENANCE.md) and
[MOD metadata](MOD_METADATA.json). The plugin remains experimental and defaults
off. Repository documentation records mechanisms and configuration, not bundled
benchmark outputs.
