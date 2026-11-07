# Scoped NPU validation and remaining boundaries

`0.2.0.dev0` remains a development version, not a universal production certificate.
The original frozen SDK failures below were not waived; explicit, version-bound
repairs were tested in isolated source checkouts. No source or metadata was
silently replaced by installing/importing the plugin.

## Reference runtime

- Original vLLM-HUST base: `d0f22d2bda562156e4dbf433ce645e1769b4f804`;
  repaired reference: `3bf4ba0b0ee547577b4e6bb772ccd45aa5793670`.
- Original Ascend base: `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`;
  repaired reference: `b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9`.
- Installed vLLM distribution `0.25.1+frontier.bidkv`; torch `2.10.0+cpu`,
  torch-npu `2.10.0.post4`, CANN 9.1, two real 910B2 devices, BF16 TP2,
  Qwen3.5-35B-A3B.
- Reference model gates configure context 262144, KV 26,038,239,232 bytes per
  card, APC, async scheduling, MTP2, and `FULL_AND_PIECEWISE`. Normal decode FULL
  remains enabled; graph eligibility is corrected for actual prefill, not
  globally disabled with an eager flag.
- The host's existing `VLLM_VERSION=0.25.1` implementation routing was preserved;
  installed distribution metadata and CANN paths were not spoofed.

The [exact-base manifest](<../runtime-patches/manifest.json>) and
[explicit installation notes](<../runtime-patches/README.md>) pin the repairs.
The ECPA discovery capsule retains its historical `.empty` constraint; direct
connector activation and a complete ECPA install/enable lifecycle are different
claims. That distribution/lifecycle is not separately hardware-qualified here.

## What passed on the repaired pair

- **295 installed-payload SDK tests**, zero failures/errors/skips, with import
  origins and all 18 Python/JSON payload files checked against the source.
- **Native, CPU-only tier, and CPU+FS each pass 80 strict model responses**.
  These include fresh/cached C1 and C8 replay, independent prompt salts, prompt
  lengths8190/8191/8192/8193 around the8192 boundary, and an earlier6217-token
  retrieval fixture. Golden strings, prompt bytes, output caps and actual usage
  checks were not changed to accept failures.
- The FS diagnostic also witnesses **8,325,808,128 reported FS-read bytes** and
  **472,940,544 reported H2D bytes**, with the strict80-response gate passing.
  Positive-device-copy cases are the C8 reset/fresh passes; a tentative FS→CPU
  promotion alone is not called a GPU recovery. Observed FS load/checksum errors
  are zero. This does not measure physical SSD cache misses or copy/compute overlap.
- Exact-source CPU contract tests cover feedback producer identity across both
  scheduling modes and batch replacement/reordering. Separate real-NPU probes
  cover transfer fences, typed SoA views, cross-group guards, V1 copies at
  representative40KiB conv/1MiB temporal sizes, and short-prefill convolution.
  These are component checks, not independent general answer-quality cases.

The CPU/FS model gate uses a diagnostic2GiB CPU tier, forced admission and a fresh
FS namespace. It is not a same-settings performance run. A configured256K context
is not proof of semantic correctness at256K input length; these named retrieval
fixtures stop at8193 input tokens and use short deterministic answers.

## Historical failures and causal fixes

1. **Connector-present hybrid hits:** the legacy scheduler/Ascend coordinator
   could retain attention hits while omitting Mamba checkpoint IDs, even for a
   no-op connector. The core now requires explicit request-scoped opt-in for
   nonuniform hybrid local hits. This plugin stays on safe unified local hits;
   old hosts are rejected before allocating transfer storage.
2. **Native MTP postprocess ABI:** the original pair failed before tiering use
   with `dynamic_func() got multiple values for argument 'block_size'`. The V1
   ABI/state metadata was ported explicitly, not repaired by dropping an argument.
3. **Feedback buffer and ownership:** adapting to the core's private accepted
   feedback buffer initially fixed continued decode but exposed a whole-new-batch
   ownership error. A new cached request could inherit acceptance3, shifting its
   conv copy by two rows and choosing an uninitialized SSM block. Actual byte
   traces showed unchanged checkpoints but wrong restore arguments. The final
   repair binds feedback to the exact producer's request identities and defaults
   new requests to1 before preparing copies.
4. **One-token prefill graph eligibility:** actual state traces showed prefill
   writes following prior decode addresses. The runtime now distinguishes
   `computed < prompt` from true decode, preserving FULL decode and eligible
   PIECEWISE prefill rather than globally disabling graphs.
5. **MTP external-match starvation:** the generic EAGLE/MTP fallback treated
   recurrent groups as volatile draft-attention tails. That demanded two
   consecutive checkpoints and omitted a sole current checkpoint from storage.
   FS reads could complete without any accepted device restore. The repaired
   core excludes recurrent groups from that fallback while preserving attention
   annotations and rejecting unsupported explicit recurrent draft annotations.

These failures remain part of the record. Passing lower-level tests, a capability
marker, CRC or HTTP status alone did not establish these fixes; unchanged model
and device-transfer gates were required.

## Operational and interpretation limits

- Performance must pin the actual SDK/source pair and serving parameters. Report
  NoMTP separately from MTP2. Do not use earlier NoMTP numbers as the final MTP2
  result or present a changed runtime as the original frozen pair.
- The private5000-row dataset has no semantic answer gold. Output caps are kept
  per row; first64/256 subsets are not a full5000 evaluation. Exact-parity common
  subsets must retain their full measured-pass wall costs and disclose selection
  bias, unverified outputs, tail latency and actual I/O before claiming benefit.
- CRC verifies bytes, not prefix meaning. TTL deletion concerns historical
  metadata/files, not active-HBM eviction, zeroization or secure erasure. CPU
  reclamation releases logical slots; it does not promise mmap RSS unmapping.
- Lifecycle CRC-fallback, activity-driven idle reclaim and TTL deletion were
  separately exercised on the earlier NoMTP maintenance profile. They are not
  automatically reclassified as MTP2 or long-soak production qualification.
- Provision CPU mmap capacity plus IPC headroom in `/dev/shm`. Stop owned API
  leaders gracefully before an owned process-group backstop. Never reclaim
  shared-memory files without verifying exact ownership and open mappings.
- Arbitrary models/SDK commits, unexercised context lengths, nonstandard state
  layouts or short/variable speculative query variants require separate
  qualification. Source-level latent constraints are not waived by these tests.

The supported conclusion is scoped functional/reference-path validation, not a
blanket claim that all model behavior or production failure modes are certified.
