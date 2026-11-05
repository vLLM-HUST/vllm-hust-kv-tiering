# Multi-turn minimal recovery benchmark

This directory reproduces the exploratory "minimal recovery scenario" used to
exercise the FS read-back path (reusing an evicted prefix reads back from the
`hust_fs` tier instead of re-prefilling). It is **not** a qualified delivery
benchmark: it uses a reduced KV budget to force eviction, which the delivery
config (26 GB/card) does not for this workload. Results are directional, not a
supported performance claim.

## Prerequisites

- Model: `Qwen3.5-35B-A3B` (BF16, 2× Ascend 910B2, TP2).
- A text source `pg1184.txt` (e.g. Gutenberg's "The Count of Monte Cristo"):
  `curl -LsSf https://www.gutenberg.org/ebooks/1184.txt.utf-8 -o pg1184.txt`.
- The host multi-turn benchmark under
  `vllm-hust/benchmarks/multi_turn/` (`benchmark_serving_multi_turn.py`,
  `bench_dataset.py`, `bench_utils.py`).

## Serve (HBM-only vs Tiering)

Both arms share the host config; the only difference is the `--kv-transfer-config`.

```bash
# HBM-only (no tiering)
VLLM_VERSION=0.25.1 vllm serve /models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --max-model-len 16384 --enable-prefix-caching \
  --kv-cache-memory-bytes 1610612736 --max-num-seqs 16

# Tiering (active offload to CPU + hust_fs; no pressure-aware switch)
VLLM_VERSION=0.25.1 vllm serve /models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --max-model-len 16384 --enable-prefix-caching \
  --kv-cache-memory-bytes 1610612736 --max-num-seqs 16 \
  --kv-transfer-config "$(cat tiering.json)"
```

`tiering.json` (no un-read pressure keys):

```json
{
  "kv_connector": "HustAscendTieringConnector",
  "kv_connector_module_path": "vllm_hust_kv_tiering.ascend_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "HustTieringOffloadingSpec",
    "spec_module_path": "vllm_hust_kv_tiering.spec",
    "cpu_bytes_to_use": 8589934592,
    "ascend_copy_backend": "torch_sync",
    "secondary_tiers": [
      {
        "type": "hust_fs",
        "root_dir": "/root/CGCL/paiouyun-fs",
        "storage_layout": "segment",
        "n_read_threads": 4,
        "n_write_threads": 4
      }
    ]
  }
}
```

## Benchmark (both arms)

```bash
python benchmark_serving_multi_turn.py \
  -m /models/Qwen3.5-35B-A3B --served-model-name qwen3.5-35b-a3b \
  --url http://127.0.0.1:8000 \
  -i generate_multi_turn_mid.json \
  --num-clients 16 --max-active-conversations 16 --warmup-step --seed 0
```

The Tiering arm's read-back is confirmed by the Prometheus metric
`vllm:kv_offload_load_bytes_total` being non-zero, while the HBM-only arm stays
at zero (it re-prefills evicted prefixes instead).
