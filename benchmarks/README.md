# Recovery diagnostics, not qualified performance results

The generated multi-turn configuration in this directory is a supplementary
pressure diagnostic. It does not replace the active `vllm-hust-benchmark`
acceptance plan or authorized Pujiang datasets. No historical performance claim
is attached to the current implementation. See
[the validation protocol](<../docs/serving-evaluation.md>).

## Prepare immutable inputs

Use the matched host's `benchmarks/multi_turn/benchmark_serving_multi_turn.py`
and its dataset utilities. `generate_multi_turn_mid.json` names a text file;
obtain that input under an appropriate license and record its exact hash. Record
the driver/core revision, tokenizer/template, generated request IDs, session IDs
and source/prepared input hashes. Do not quietly substitute another text source.

The generated scenario may need a deliberately reduced HBM budget to exercise
recovery. Such a budget is a separate, approved diagnostic configuration and
must not be relabeled as the unchanged delivery line. Ensure the hybrid model's
Mamba/Attention cache geometry can support the chosen context/concurrency.

## Start one owned arm at a time

Use a dedicated resource window; these are command templates, not commands that
this repository runs automatically. Provision matching immutable model/runtime
artifacts first. Do not overwrite running services or another experiment's cache.

```bash
# Delivery-line template; replace MODEL with the immutable local model path.
vllm serve MODEL --served-model-name qwen3.5-35b-a3b \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 262144 --enable-prefix-caching \
  --kv-cache-memory-bytes 26038239232 --max-num-seqs 16
```

This Native arm omits the connector. For a CPU-only arm, append:

```bash
--kv-transfer-config "$(cat examples/cpu-only.json)"
```

For CPU+FS, append the following instead, after setting an owned absolute storage
namespace and verifying `/dev/shm` and filesystem/quota headroom:

```bash
--kv-transfer-config "$(cat examples/pressure-tiering.json)"
```

Keep all other server arguments equal. Freeze APC, MTP, graph, async scheduling,
batch tokens and other release settings in the actual benchmark manifest rather
than assuming the short template is a complete official capsule.

For a separately approved reduced-budget recovery scenario, freeze the altered
context, HBM budget and concurrency identically across all three arms. A full
CPU-only versus CPU+FS comparison is needed to attribute benefit specifically to
FS rather than DDR. A prefix-cache reset alone does not prove a disk hit.

## Run the same diagnostic driver

From the matched host's multi-turn benchmark directory, after making the exact
configuration and its text input available:

```bash
python benchmark_serving_multi_turn.py \
  -m MODEL --served-model-name qwen3.5-35b-a3b \
  --url http://127.0.0.1:8000 -i generate_multi_turn_mid.json \
  --num-clients 16 --max-active-conversations 16 \
  --warmup-step --seed 0 --no-early-stop
```

Retain all errors/timeouts and actual request counts. Validate responses using
the frozen correctness protocol, not merely HTTP success. Record TTFT, TPOT,
tail latency, successful output-token throughput and the cost of stores as well
as restores. Collect per-arm before/after metric snapshots and request-level
recovery attribution.

**Do not use `vllm:kv_offload_load_bytes` as proof of FS recovery.** It counts
CPU→device traffic. Use completed filesystem-read and promotion evidence,
including `vllm:kv_tiering_fs_load_bytes`/`load_hits`, and distinguish FS I/O from
physical SSD misses/page-cache effects. Store experimental artifacts only in the
approved benchmark/evidence destination, not in this code repository.
