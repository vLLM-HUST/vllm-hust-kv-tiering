# Ascend hybrid host safety prerequisites

The explicit `ascend_copy_backend: torch_async` profile (including its
`torch_sync` legacy alias) is supported only with a reviewed, matched core and
Ascend runtime. An original version string is not a compatibility guarantee.
This includes `MambaSpec` nested in `UniformTypeKVCacheSpecs`.

## What the plugin guard checks

Before storage registration, shared CPU allocation, or worker initialization,
the plugin checks for the core's request-scoped
`SupportsNonuniformHybridLocalHits` API and callable
`can_use_nonuniform_hybrid_local_hits(request)` method. The safe default is
false. Ordinary offloading uses unified local HMA cache hits, preserving the
actual recurrent checkpoint IDs. `SupportsHMA` alone does not establish this
contract.

The legacy scheduler selected its special hybrid hit path merely because a
connector existed. The Ascend coordinator then omitted local Mamba checkpoints
while the scheduler advanced by attention hits. Even a no-op connector could
therefore corrupt a cached prefix without moving any bytes. The guard rejects
that known-unsafe pairing and has no bypass switch.

**API recognition is necessary, not proof of arbitrary host correctness.** The
guard neither invokes the opt-in method nor imports an accelerator runtime. It
does not inspect every Ascend implementation detail listed below. Pure-attention
configurations and the generic host/CUDA profile are unchanged by this guard.

## Additional matched-runtime contracts

The reviewed SDK repairs are explicit source patches, not import-time host
monkeypatches, fake platform detection, or package-version overrides. See the
[patch installation notes](<../runtime-patches/README.md>) and
[exact-base manifest](<../runtime-patches/manifest.json>).

1. **Prefill is not a decode graph.** In aligned hybrid mode, a remaining prompt
   of one token (or three with MTP) must not be classified as a uniform decode
   simply because its query length matches. Graph eligibility uses the same CPU
   `computed < prompt` predicate as attention metadata. Normal decode FULL and
   explicit dummy/capture overrides remain supported; eligible short prefills
   can use PIECEWISE. Large-prefill effective modes remain backend-dependent.
   Consequently, NoMTP does not remove the need for this Ascend repair.
2. **MTP V1 postprocess ABI.** The frozen core and Ascend implementation disagree
   on the positional ABI and state-copy metadata. The V1 repair must match the
   full conv/temporal/group/address/acceptance contract. Unsupported V2 flags and
   unsupported convolution layouts fail closed; passing a small copy micro-test
   alone is not whole-model qualification.
3. **Feedback has a producer identity.** Aligned postprocess writes a private
   accepted-token feedback buffer. The consumer must synchronize its event and
   map those rows by the exact producing request identities. A wholly new batch
   must start at acceptance 1, not inherit the previous request's count. The
   reviewed repair keeps a dedicated producer snapshot for both scheduling
   modes and preserves non-align handling. Resetting acceptance after preparing
   a cached-state copy is too late to repair wrong source offsets.
4. **A recurrent checkpoint is not a volatile draft-attention tail.** The
   core offloader's generic MTP/EAGLE fallback must not demand two consecutive
   Mamba checkpoints or omit the sole current checkpoint from storage. Mamba
   keeps its single-checkpoint lookup window. Explicit attention draft-group
   annotations are retained; an explicitly unsupported recurrent draft-tail
   annotation is rejected instead of silently overwritten.

## Qualification boundaries

Use the precise SDK bases and patch hashes in the manifest. Correct HTTP status,
byte CRC, a capability marker, or a passing unit suite is not semantic proof.
Validate cold and cached-prefix model responses, actual GPU transfer completion,
and negative paths before deployment. FS reads can be tentative CPU promotions
that never become an accepted external match; require device-load evidence and
unchanged golden responses when claiming recovery.

The reference workload is Qwen3.5-35B-A3B BF16, TP2, aligned hybrid caching on the
reviewed 910B2 host. Report NoMTP and MTP2 separately. A configured context limit
is not evidence that all context lengths were exercised. Private-data throughput
without answer gold is not universal model-quality certification, and a strict
exact-output subset must not be presented as whole-dataset equal-quality speedup.
The repair does not silently change client inputs, golden answers, output caps,
CANN paths, device masks, or unrelated processes.
