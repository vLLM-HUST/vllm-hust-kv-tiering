# NPU qualification status: pre-PR gates are not green

`0.2.0.dev0` is a development candidate, not a production-qualified release.
Software tests and byte-copy tests do not establish full-model cache correctness.

## Runtime actually exercised

- vLLM-HUST source: `d0f22d2bda562156e4dbf433ce645e1769b4f804`.
- vLLM-Ascend-HUST source: `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`.
- Installed vLLM distribution: `0.25.1+frontier.bidkv`; torch `2.10.0+cpu`,
  torch-npu `2.10.0.post4`, CANN 9.1.0.
- Two real Ascend 910B2 devices, BF16 TP2, Qwen3.5-35B-A3B.
- The existing Ascend version-routing override `VLLM_VERSION=0.25.1` was used;
  it does not change the installed distribution metadata. Inherited CANN Python
  and library paths must be preserved, not replaced.

The packaged ECPA discovery manifest still has the historical exact `.empty`
distribution constraint. Direct connector activation, discovery, package import,
and a successful ECPA install/enable lifecycle are different claims; do not
interpret these tests as qualification of that lifecycle or distribution.

## Blocking full-model observations

### Native MTP2 baseline fails independently of this plugin

With the intended 256K context, 26,038,239,232 KV bytes per card, APC, asynchronous
scheduling, MTP2 and FULL_AND_PIECEWISE graphs, the Native server reached readiness
but its first inference failed in fused Mamba postprocessing. The frozen core and
Ascend replacement have incompatible call signatures, producing:

```text
TypeError: dynamic_func() got multiple values for argument 'block_size'
```

This occurred without the tiering connector. Dropping an argument, pretending to
be different hardware, or disabling MTP2 is not qualification of that baseline.
No host source or kernel was patched to manufacture a passing result.

### Connector-present hot-prefix correctness remains unresolved without MTP

In a separate no-MTP diagnostic, C1 retrieval and cold inference can pass while
later C8 hot-prefix reuse produces instruction echoes, repetitive fragments,
whitespace or gibberish. CPU-only and filesystem-backed runs showed the symptom.
A refused-admission control with zero transfer history also reproduced it.

Crucially, a separate empty connector implementing only the host interfaces,
without importing the tiering implementation or creating cache views, mmap,
streams, managers or transfers, also reproduced the severe hot-pass symptom.
This demonstrates that tiering copies and its internal scheduler/manager are not
necessary to trigger that failure. It narrows the implicated integration path;
it does **not** identify a particular host kernel or prove every plugin behavior
correct.

Switching to NHD or disabling asynchronous scheduling did not remove the observed
failure. An eager/NONE-graph empty-connector run removed the observed severe
second-wave loops in that run, but retained an instruction echo and failed strict
warm/reference comparisons. Execution timing and batching also changed. This is
not an accepted fix, a supported workaround, or proof of a graph-specific cause.

CRC32 validates stored bytes, not the meaning of a prefix key, layout, GDN state,
or subsequent HBM cache reuse. Zero checksum errors cannot waive this gate.

## What lower-level tests do and do not show

Real-device tests exercise both visible NPU devices, typed hybrid SoA views,
partial pages, stream/event ownership and cross-group noninterference. Separate
byte probes exercise D2H/H2D and query-only D2H completion before any explicit
fence. Native versus nonzero-offset aligned Mamba-copy microcases also check full
bytes and guards. These are valuable component checks, not substitutes for the
blocked full-model path or evidence of compute/copy overlap.

During validation, the pressure adapter was corrected to read the primary prefix
lookup index rather than its supplemental-alias table. Its O(1) cached-key signal
is a pressure proxy, not exact physical residency: aliases can overcount blocks,
and duplicate physical blocks under one key can undercount them. Layout checks
also reject cross-canonical-ID overlaps and unsupported ambiguous cache views.
Those fixes must not be described as fixing the independent hot-prefix failure.

## Operational and release boundaries

- Do not deploy this frozen pair for production hybrid-prefix offloading on the
  strength of passing unit tests, C1 examples or transfer/integrity counters.
- Require a compatible base runtime and full-model correctness requalification
  before accepting performance or production-readiness claims.
- Small synthetic retrieval fixtures and capped/truncated Pujiang diagnostics are
  not official benchmark or general task-quality certification. No official
  large-scale quality/performance acceptance follows from them.
- Provision CPU mmap capacity **plus IPC headroom** in `/dev/shm`. Use an owned
  graceful API shutdown (for example `--shutdown-timeout 60`) before a forced
  process-group backstop. Abnormal termination may leave named mmap resources;
  verify exact ownership and absence of open mappings before reclaiming them.
- TTL deletion concerns plugin-managed historical CPU/FS metadata. It is not
  active attention oversubscription, HBM zeroization or physical secure erasure.

**Current decision: pre-PR production/correctness qualification is not complete;
these blocking observations must remain visible, not converted into a green
result by changing references or silently weakening the serving configuration.**
