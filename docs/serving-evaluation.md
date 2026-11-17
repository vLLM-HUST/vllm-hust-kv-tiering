# Serving validation status and acceptance protocol

Qualification requires the exact current plugin source and matched host. Run
software, device and full-model checks separately; keep generated results and
logs outside this repository. [The reproduction guide](reproduction.zh-CN.md)
provides a concrete two-tier SWE replay configuration and three-tier checks.

## Three separate questions

1. **Correctness:** are Attention KV and hybrid GDN/Mamba state restored with the
   required alignment and unchanged model behavior, including abort/preemption,
   shared prefixes, TTL and transfer failures?
2. **Delivery-line non-regression:** under the frozen Qwen3.5-35B-A3B BF16,
   2×910B2/TP2, 262144 context and 26038239232 KV bytes/card contract, does enabling
   this plugin avoid a meaningful throughput/SLO regression?
3. **Recovery benefit:** under an explicitly approved pressure/repeated-prefix
   contract, is completed FS recovery faster than recomputation, after separating
   DDR hits from filesystem hits and accounting for write cost?

A positive answer to one does not establish the others. A CPU-only cache can
improve TTFT without a single FS read. A completed NPU→CPU→NPU byte roundtrip is
not full-model hybrid-state correctness. A reduced-budget scenario is
supplementary evidence, not the unchanged delivery configuration.

## Freeze before running

Use the active `vllm-hust-benchmark` acceptance contract and the authorized
Pujiang datasets. Preserve their original input/response protocol, identities,
order and hashes; do not silently flatten multi-turn messages, invent prompts,
truncate contexts or replace specified data with a convenient synthetic trace.
Synthetic pressure inputs are supplementary diagnostics only.

Record immutable identities for:

- plugin/core/Ascend commits; wheel hashes; container/driver/CANN/torch/torch_npu;
- model weights/revision, tokenizer and chat-template hashes;
- complete server argv, effective configuration and request/scoring contract;
- source/prepared dataset hashes, subset, request IDs, session IDs and ordering;
- pressure thresholds, CPU budget, FS quota/durability, storage device/mount,
  thread limits, concurrency, cache warmup/reset and measurement windows.

A `VLLM_VERSION` environment variable is not version provenance. Use a separate
cache namespace for immutable model/configuration generations. Do not hot-update
weights while continuing to serve persisted KV from the old generation.

## Paired arms

Run Native (no connector), CPU-only (`examples/cpu-only.json`) and CPU+FS
(`examples/pressure-tiering.json`) with the same host/model, HBM budget and client
contract. The CPU-only arm describes the offload tier, not CPU model execution.
Do not start concurrent full-model arms on a two-card host unless the resource
contract explicitly permits it.

Use independent storage namespaces and declared cold/warm-cache treatment. Freeze
warmup and window boundaries before measurement; retain every submitted request,
failure and timeout. If the host multi-turn driver uses early stop, declare its
sampling rule or disable it (`--no-early-stop`); do not call an incomplete sample
an entire-dataset run. Repeat/interleave arms as the benchmark contract requires,
and report uncertainty rather than selecting the best run.

## Evidence and metrics

Collect Prometheus snapshots and original per-request streaming times. Keep the
following paths separate:

- `vllm:kv_offload_load_bytes`: CPU→device adapter traffic, **not FS traffic**;
- `vllm:kv_tiering_fs_load_bytes`: actual filesystem read bytes;
- `vllm:kv_tiering_fs_load_hits`: blocks read successfully from FS;
- `vllm:kv_tiering_fs_store_bytes`: actual segment write bytes;
- `vllm:kv_tiering_fs_*errors` (including `checksum_errors`), `queue_rejections`, `capacity_rejections`:
  failed/dropped cache opportunities;
- policy pressure, residency and admission-decision observations.

Prometheus exporters may append `_total` to counter names. Match labels and use
before/after deltas for the owned process and measurement window. FS counter
increases establish filesystem I/O, not a physical SSD cache miss or performance
gain by themselves; document O_DIRECT/page-cache/storage semantics. In addition
to FS counters, require a successful completed promotion, request-level reuse
attribution, actual prefix eviction, and output/state correctness evidence.

Report throughput, successful/SLO-compliant output tokens, TTFT and TPOT mean and
p95/p99, failed/cancelled/timed-out requests, recomputed/prefilled tokens,
CPU/HBM occupancy, FS bytes/errors and write amplification. Do not hide a TPOT
regression behind improved aggregate request throughput.

## Software and device validation gates

See `README.md` for portable and explicitly enabled NPU tests. On the matched
host, CPU integration tests cover the actual factory/spec/connector/primary/FS
contracts without running a model. These tests are useful prerequisites, not a
replacement for the serving protocol above.

Before claiming true async overlap, measure submit latency and collect stream
traces for the matched torch_npu/mmap path. `non_blocking=True` is a request, not
proof that the runtime avoids synchronous staging. Test device coherence,
preemption/reuse fences, cancellation and conservative shutdown separately.

Keep actual experimental evidence in `vllm-hust-benchmark` or the separately
approved evidence destination. This code repository contains mechanisms,
configuration and validation entry points, not fabricated or copied performance
results. Historical Frontier evidence remains historical; link a new immutable
record only after the current code has been measured and independently checked.
