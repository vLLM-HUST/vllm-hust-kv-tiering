# Serving evaluation (2026-11)

KV tiering 插件的 serving 性能评测记录：数据集来源、测试配置、复现步骤与定性结论。

## 1. 测试数据集

**来源**：派欧云（paiouyun）服务负载数据集的「复用对话」子集
`reuse_conversation.jsonl`。

- 原始归档：`vLLM-HUST/kv-tiering-evaluation-backup` 仓库的
  `dataset-archive/paiouyun-testset.tar.gz.part-*`（6 个分片，`cat` 合并后
  `tar -xzf` 解压）。
- 解压后路径：`派欧云/final_workloaddata_v2/final_workloaddata_v2/reuse_conversation.jsonl`。
- 规模：**5000 条**。每条是一次多轮对话，含一个**组内共享的上下文前缀**
  （`group_id` 相同则前缀相同）加不同的用户查询，是构造「重复前缀复用」的
  最小负载。

### 数据集准备

原始 JSONL 是多轮 `messages` 结构，需转成 vLLM 的 CustomDataset 格式
（`{"prompt": ..., "output_tokens": ...}`）：

```python
def message_text(messages):
    parts = []
    for m in messages or []:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        parts.append(f"[{m.get('role', 'user')}]\n{content}")
    return "\n\n".join(parts)

# 每条：prompt = flatten(messages), output_tokens = metadata.output_tokens_target
```

转换后的 `reuse_conversation.jsonl` 每条约 5900 字符（~3000 tokens），
`output_tokens` 默认 384。

## 2. 测试配置

### 强制基线（email #2 / V5.5 对齐）

| 项 | 值 |
| --- | --- |
| 模型 | Qwen3.5-35B-A3B |
| 精度 | BF16 |
| 硬件 | 2 × Ascend 910B2 / TP2 |
| 上下文 | 32768（32K）或 262144（256K） |
| prefix caching | 开启 |
| 并发 | 8（`--max-concurrency 8`） |

### Tiering 配置（`kv-transfer-config`）

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
    "tiering_hbm_pressure_aware": true,
    "tiering_hbm_high_watermark": 0.70,
    "tiering_hbm_low_watermark": 0.50,
    "tiering_reclaim_device_cache_after_store": true,
    "secondary_tiers": [{
      "type": "hust_fs",
      "root_dir": "/root/CGCL/paiouyun-fs",
      "storage_layout": "segment",
      "n_read_threads": 4,
      "n_write_threads": 4
    }]
  }
}
```

两种 offload 策略：

- **压力感知（生产）**：`tiering_hbm_pressure_aware: true`，只在 HBM 用量
  超过 `high_watermark` 时 offload。
- **主动 offload（测试用）**：`tiering_hbm_pressure_aware: false`，无条件
  offload。用于在短上下文下强行制造 offload/reuse 场景。

### 256K 显存配置

256K 上下文需显式 KV 预算（`--gpu-memory-utilization` 会算不足）：

```bash
--max-model-len 262144 \
--kv-cache-memory-bytes 26038239232   # email #2 的「每卡 KV 26GB」
```

## 3. 定性结论

实测数据（吞吐、TTFT、TPOT、加速比、store/load 字节数）遵循仓库的
benchmark-only 证据政策，不落在本代码仓库；统一见 vLLM-HUST 官网证据页
[FRONTIER-KV-TIERING-20260926.md](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/FRONTIER-KV-TIERING-20260926.md)。
本节只保留定性结论与复现口径。

### 3.1 32K Native vs Tiering（主动 offload）

32K 短上下文下，主动 offload 的 store 开销几乎为 0（异步 stream/event 与
decode 重叠），但也没有复用收益（KV 未超 HBM）。

### 3.2 256K 压力感知 store

256K + 压力感知下，HBM 到高水位线自动触发 offload，三级瀑布
（HBM → DDR → SSD）正常工作；自然负载下无回读，复现了「只有写没有读」的
现象。

### 3.3 round-trip（最小恢复工况）

流程：发长请求（冷 prefill）→ offload 到 FS → `POST /reset_prefix_cache`
强制淘汰 HBM → 重发相同请求（从 FS load）。

复用已淘汰的 KV 时，分层「从 SSD 读回」比「重新 prefill」更快，且真实长
前缀下 prefill 耗时远高于 load，加速优势随前缀长度增大。

### 3.4 多轮对话最小恢复工况（真实缓存压力 + 重复前缀）

负责人 issue #2 要求的「真实缓存压力 + 重复前缀的最小恢复工况」用合成多轮
对话负载复现：

- 24 个对话，每个含一段共享前缀 + 一段独有前缀 + 15~25 轮（后续轮次复用
  前面的 KV，构成重复前缀）；
- 将 KV 预算压小（每卡 1.5 GB）以制造真实的 HBM 淘汰压力；
- 对比 HBM-only（无分层，淘汰后重新 prefill）与 Tiering（分层，淘汰后从
  FS 读回）。

定性结论：该工况下 Tiering 的吞吐与 TTFT 均显著优于 HBM-only，且
`kv_offload_load_bytes` 确认发生了从 FS 的读回（而非 HBM 前缀缓存命中）——
即「读回 > 重算」的恢复收益成立。具体数值、完整命令与原始数据见 PR 描述与
benchmark 仓库证据。

## 4. 复现命令

```bash
# 数据集准备（见 §1）
# 启动 serve（Tiering，压力感知，256K）
VLLM_VERSION=0.25.1 vllm serve /models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --max-model-len 262144 --enable-prefix-caching \
  --kv-cache-memory-bytes 26038239232 \
  --kv-transfer-config "$(cat tiering-config.json)"

# 压测
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model qwen3.5-35b-a3b \
  --tokenizer /models/Qwen3.5-35B-A3B \
  --dataset-name custom --dataset-path reuse_conversation.jsonl \
  --num-prompts 800 --disable-shuffle --request-rate inf \
  --max-concurrency 8 --temperature 0 --ignore-eos
```

## 5. 已知边界

1. **复用（load）不自然触发**：`reuse_conversation` 的同组复用发生在 HBM
   淘汰之前，故连续负载下无 FS 回读。要触发 load 需显式的最小恢复工况
   （`reset_prefix_cache` + 重发），或负载的 KV 真正超过 HBM。
2. **`/dev/shm` 是 16GB tmpfs**：8GB CPU offload 层 mmap 在 `/dev/shm`，异常
   退出（`kill -9`）会残留 `vllm_offload_*.mmap`，占满后导致
   `madvise: Bad address`。需手动清理或增大 `/dev/shm`。
3. **32K 下压小 KV 预算受限**：混合模型的 Mamba cache 要求
   `max_num_seqs ≤ Mamba blocks`，`gpu_memory_utilization` 过低会被拒。
