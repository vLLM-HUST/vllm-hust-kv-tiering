# Explicit matched-runtime repairs

These Python-only SDK patches are a reviewed dependency of the reference
Ascend hybrid profile. They are **not applied by importing or installing the
plugin**. Never apply them to an unrelated running checkout.

| SDK | Exact base | Repaired reference |
|---|---|---|
| vLLM-HUST | `d0f22d2bda562156e4dbf433ce645e1769b4f804` | `3bf4ba0b0ee547577b4e6bb772ccd45aa5793670` |
| vLLM-Ascend-HUST | `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` | `b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9` |

- [Core patch](<vllm-core-mtp-offload-safe.patch>): request-scoped safe hybrid
  local hits and recurrent-checkpoint-safe MTP offload eligibility.
- [Ascend patch](<vllm-ascend-mamba-v1-safe.patch>): V1 postprocess ABI, feedback
  buffer and producer-identity ownership, and prefill/decode graph eligibility.
- [Manifest](<manifest.json>): exact SHA256, bases, repaired commits and patch sizes.
  Verify application against your isolated source checkout before deployment.

## Apply explicitly in isolated SDK worktrees

1. Preserve the original SDK and its uncommitted changes. Create isolated source
   checkouts from the exact bases above. A plain git clone may omit ignored
   runtime binaries and generated version files; it is not automatically a
   runnable Ascend installation.
2. Follow the SDK's normal build/deployment procedure with the matching CANN,
   torch, torch-npu, compiled extensions, and model runtime. In the reference
   experiment, existing ignored runtime artifacts were copied from the frozen
   host and checked byte-for-byte; no replacement C++ binary is shipped here.
3. In the isolated **core** checkout, verify its base commit, then run:

   ```bash
   git apply --check /path/to/vllm-core-mtp-offload-safe.patch
   git apply /path/to/vllm-core-mtp-offload-safe.patch
   git diff --check
   ```

   Repeat in the isolated **Ascend** checkout with its matching patch. Do not
   force a failed hunk, change package metadata to mimic an ABI, or use a wider
   device mask to make a failed deployment appear successful.
4. Install/build the plugin separately and verify the actual import origins and
   payload hashes. Configure the repaired source roots explicitly. The reference
   host's existing `VLLM_VERSION=0.25.1` implementation routing is not permission
   to spoof distribution metadata on another host.
5. Run the plugin suite and independent model gates before performance. Require
   cold and cached-prefix exact golden responses, correct actual token usage,
   both ranks, and completed H2D plus FS-read evidence for a recovery claim.
   HTTP200 or a passing copy micro-test alone is insufficient.

For the rationale and limits, see the [host safety contract](<../docs/ascend-hybrid-host-safety.md>).
The patches intentionally avoid model/client/golden changes and preserve normal
FULL decode. They are not certification of arbitrary models, SDK commits,
convolution layouts, speculative variants, or every configured context length.
