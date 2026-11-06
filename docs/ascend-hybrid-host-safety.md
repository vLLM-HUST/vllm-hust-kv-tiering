# Ascend hybrid host safety prerequisites

For the explicit `ascend_copy_backend: torch_async` profile (including its
`torch_sync` legacy alias), caches containing `MambaSpec` require the separately
repaired frozen core ABI. This also includes Mamba specifications nested inside
`UniformTypeKVCacheSpecs`.

The plugin checks this **before storage registration and before its host spec
constructor can allocate shared CPU regions or initialize copy workers**. A known
legacy host fails immediately with an actionable error. There is no override to
allow the known-unsafe pairing.

## Why this pairing is unsafe

The legacy core scheduler selects a special prefix-hit path merely because a KV
connector is present. The Ascend hybrid coordinator on that path skips local
Mamba checkpoints, but the scheduler still advances computed tokens by the full
attention hit. A connector returning zero external matches cannot restore a
checkpoint the scheduler discarded. Even a resource-free no-op connector can
therefore trigger cached-prefix corruption without any DMA or filesystem I/O.

The separate core repair makes this optimization an explicit, request-scoped
`SupportsNonuniformHybridLocalHits` capability with callable
`can_use_nonuniform_hybrid_local_hits(request)`. Its default is false; ordinary
offloading connectors use unified local HMA cache hits, preserving actual
recurrent checkpoint IDs. `SupportsHMA` alone is not this guarantee.

The plugin guard recognizes the presence of this repair API on
`vllm.distributed.kv_transfer.kv_connector.v1.base`. It neither opts the plugin
into nonuniform hits nor invokes the capability method. **API recognition is not
proof that an arbitrary host correctly implements the scheduler repair**; use the
separately reviewed repaired core with a matching Ascend runtime and validate
correctness. No-MTP use does not inherently require an Ascend source change.

Pure-attention cache configurations and the generic host/CUDA profile without
an explicit Ascend-copy backend are unchanged by this guard. No accelerator
runtime is imported by the guard, and no hardware detection or fake platform is
used.

## Independent Ascend V1 MTP ABI prerequisite

When MTP/speculative decoding is enabled, the frozen core/Ascend pair must also
include its separately reviewed V1 Mamba postprocessing ABI compatibility fix.
The original pair has an independent positional-argument mismatch on that path.
The hybrid-prefix guard does **not** inspect or certify the MTP fix. Matching the
new HMA capability API is therefore necessary for this explicit hybrid profile,
not sufficient qualification for all MTP/model configurations.

Current MTP evidence: the candidate V1 ABI compatibility patch passes96 NPU
micro cases, and the whole SDK test suite passes287 tests in14.39 seconds on the
qualified test host. These are **not** model qualification: Native MTP2 Needle
still returns wrong codes on2/8 requests (including75002 and757577), reproducibly.
The planned boundary run did not execute because the first model gate failed.
MTP ABI/semantics remain unqualified; do not infer that all bugs are fixed.

Separately, changing only the Null connector's hybrid local-hit scheduler branch
clears the severe hot-prefix symptom: all16 warm responses show task-directed
reasoning instead of instruction echo/blank/gibberish. Exact output parity is
still partial, so this is not universal bit-exact quality certification. No-MTP
performance, if measured, must be qualified and reported separately from MTP.

This change does not alter model/client/golden inputs, performance settings,
package version, CANN paths, or production processes. Rebuild and qualify the
plugin against the repaired pair before deployment; do not interpret a
successful guard as model correctness or performance certification.
