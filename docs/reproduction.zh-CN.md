# Qwen3.5 KV 分层复现配置

本文提供功能检测与专项实验的运行方法。原始数据、生成输入、服务日志、结果和
证据归档保存在插件仓库之外。

SWE 历史回放使用 [`examples/swe-prefix-replay.json`](../examples/swe-prefix-replay.json)，
只启用 HBM 与 CPU，用于检查缓存压力下的前缀恢复。
[`examples/pressure-tiering.json`](../examples/pressure-tiering.json)开启文件层，
三层功能检查另见下文。两层实验不能作为文件层加速的证明。

## 固定运行环境

| 项目 | 配置 |
| --- | --- |
| 加速器 | 两张 Ascend 910B2，每张 64 GiB，TP=2 |
| 模型 | Qwen3.5-35B-A3B，混合 Attention/GDN，BF16 |
| vLLM-HUST 源码 | `3bf4ba0b0ee547577b4e6bb772ccd45aa5793670` |
| Ascend 源码 | `b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9` |
| Python | 3.12.13 |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| CANN | 9.1.0 |
| Transformers / Tokenizers | `5.14.1` / `0.22.2` |
| 执行方式 | eager、APC、异步调度，不启用 MTP |
| HBM 分配 | 双方均 `gpu-memory-utilization=0.92`，自动分配 |
| 服务容量 | context=262144，max-num-seqs=128，max-num-batched-tokens=4096 |

必须使用实际导入的上述源码及匹配的编译扩展。原专项环境保留的 distribution
metadata 为 `vllm=0.23.0+empty`、`vllm-ascend=0.23.0`，且已有
`VLLM_VERSION=0.25.1` 路由设置；这些字符串不是宿主源码身份，不能通过修改
metadata 代替部署指定源码。新环境应按 SDK 正常构建/安装并记录真实导入路径。
安装插件不会自动应用宿主修复，参见 [SDK 前置条件](../runtime-patches/README.md)。

记录待测插件提交及未提交差异/源码哈希，使用当前完整源码。记录模型 revision、
权重、分词器和模板校验值；原专项环境没有完整哈希权重分片，因此不能声称不同
机器的权重已经过逐字节一致性核验。

## 数据来源与确定性选样

使用项目所列 SWE-bench Pro 的公开 `claude-sonnet-4-paper` 历史轨迹。专项配置
与官方 fresh-salt/零复用档不同，不是实时 Agent 解决率或官方评分实验。

以下 benchmark 固定提交中的协议和工具定义与原专项实验使用的文件逐字节一致：

```bash
# 在插件根目录运行；生成物全部写到仓库外的新目录。
export KV_REPLAY_ROOT=/tmp/kv-tiering-swe-replay
export KV_REPLAY_MODEL=/models/Qwen3.5-35B-A3B
mkdir -p "$KV_REPLAY_ROOT"
curl -fL 'https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/132c37f739d443daa9db5f2d98fc23374ab4d42a/docs/dataset-standards/five-datasets-v1/public-agent-trace-fixed32-v1.json' \
  -o "$KV_REPLAY_ROOT/protocol.json"
curl -fL 'https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/132c37f739d443daa9db5f2d98fc23374ab4d42a/docs/dataset-standards/five-datasets-v1/swe-agent-v110-tools.json' \
  -o "$KV_REPLAY_ROOT/tools.json"
```

核对下载文件 SHA256：

- protocol.json：`eacf3628cda8d7accc2e39b20fce687b1deb1cb0b0f470e8b11c8dd4517f71f8`
- tools.json：`95c0546bbc92c4a6a9e6780ae825d78e05e2c436211603790502303b7f74fcf8`

沿用协议中的 32 个 SWE 任务及顺序；从协议其他已发布任务中，按
`SHA256(UTF8("kv-tiering-pressure-pool-v1" + NUL + task_id))` 十六进制值升序选前
96 个，追加构成 128 任务池。源任务版本为 `ScaleAI/SWE-bench_Pro` v1 revision
`2d52cb3df914a3fcf80c7f66738b3a88ae37fc50`。不按结果选样，下载失败时终止，
不换任务。额外任务字节锁在准备阶段生成，双方共用同一准备输入；逐字节核对
原实验还需维护者仓库外保存的额外任务下载锁。

```bash
python scripts/prepare_swe_replay.py \
  --protocol "$KV_REPLAY_ROOT/protocol.json" \
  --tools "$KV_REPLAY_ROOT/tools.json" \
  --model "$KV_REPLAY_MODEL" \
  --output "$KV_REPLAY_ROOT/prepared"
```

可用 `--source-cache /path/to/downloaded-trajectories` 读取已有 `.traj` 文件。
准备过程保留完整历史，不截断、填充、对齐或加请求盐；剔除原本不完整或工具调用
结构无效的步骤。任一步输入加输出超过 261120 token 时排除整个任务，不补选。
Qwen 模板使用等价 JSON 对象工具参数、`enable_thinking=False`、
`add_generation_prompt=True`，并校验协议中的长度检查点。

每轮提交各任务的下一个有效步骤，等本轮全部完成再开始下一轮。后续输入继续使用
原录制历史，而非本轮生成内容；不执行工具、不重放工具等待。请求使用
`/v1/completions` 和预先模板化的 token ID；固定输出 32 token、`temperature=0`、
`ignore_eos=true`，客户端并发上限 128，单请求超时 1800 秒。不得删除慢请求、失败请求或只取前半程。

## Native 与两层分层启动

按宿主安装流程配置 CANN 环境，确认两张卡可由实验独占，并为 14 GiB CPU 共享
缓存预留足够 `/dev/shm` 与 IPC 余量。准备脚本需要上述 tokenizer 依赖；回放
脚本另需 `aiohttp`，这些实验工具依赖不添加到插件运行时依赖。

```bash
python -m pip install --no-deps .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 18236 \
  --model "$KV_REPLAY_MODEL" --served-model-name qwen3.5-35b-a3b \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --async-scheduling --enforce-eager --shutdown-timeout 60
```

服务就绪后，运行 Native 客户端：

```bash
python scripts/replay_swe.py \
  --plan "$KV_REPLAY_ROOT/prepared/plan.json" \
  --output "$KV_REPLAY_ROOT/native"
```

Native 完成并正常停止后，使用相同公共启动命令，加上以下参数启动分层臂：

```bash
--kv-transfer-config "$(cat examples/swe-prefix-replay.json)"
```

分层服务就绪后，同一客户端、同一 `plan.json` 的输出目录改为
`"$KV_REPLAY_ROOT/tiering"`。两臂分别启动新引擎，保留首次冷请求与完整收尾；
禁止同时启动两套模型，也不清理活进程共享内存。双方都不设置
`--kv-cache-memory-bytes`、`--num-gpu-blocks-override` 或 `--speculative-config`。

两层配置的完整参数在 JSON 中，关键值如下：

| 参数 | 值 |
| --- | --- |
| CPU 总预算 | 15032385536 bytes（14 GiB，宿主按 TP 分配） |
| 文件层 | `secondary_tiers=[]` |
| HBM 压力高/低水位 | 0.8 / 0.6，压力感知开启 |
| CPU 高/低水位 | 0.9 / 0.7 |
| 工作集准入 | 开启，窗口 600 秒，最多 64000 条目 |
| 单请求累计新增写入预算 | 67108864 bytes |
| 每步存储 block 预算 | 128（按完整混合组准入的软上限） |
| 在途设备存储 job / spill job 上限 | 8 / 4 |
| idle TTL / 按年龄降级 | 均为 0，关闭 |
| 最大会话 / 驻留条目 | 4096 / 64000 |

## 三层功能检测

在匹配宿主上检查真实 CPU/文件层管理器往返：

```bash
python -m pytest -q tests/test_connector_policy_contract.py \
  -k complete_spec_manager_fs_promotion_roundtrip
```

该测试检查宿主管理器的文件写入、CPU 副本移除、文件回读与字节一致性，不等同
整模型三层恢复。真实设备组件检查需要独占 NPU：

```bash
VLLM_KV_TIERING_RUN_NPU_TESTS=1 python -m pytest -q -m npu \
  tests/test_ascend_worker.py tests/test_ascend_layout.py
```

整模型三层检查使用 `examples/pressure-tiering.json`，把 `root_dir` 改为本次运行
独立绝对目录，记录介质、挂载、页缓存处理和 namespace。示例 CPU 预算为 1 GiB、
文件配额 64 GiB、segment 为 1 GiB、读写线程各 2、pending job 上限 64、CRC32
开启、durability=none、按年龄降级 60 秒、TTL 为 300 秒且不删除文件副本。
这些值属于功能施压示例，与两层性能配置不同。

使用带确定答案的固定长前缀先填充，再使前缀离开 HBM 和 CPU，最后重访。必须
同时核对正确答案、完成的文件回读、CPU→NPU 恢复及请求级复用；仅写入、lookup
命中或 CPU 缓存命中不能证明三层恢复。比较三层净收益时，增加相同 CPU 预算、
生命周期和公共配置、仅清空 `secondary_tiers` 的 CPU 对照臂，输入也必须相同。

## 计时、指标与边界

输出吞吐用全部有效输出 token 除以完整轮次墙钟之和；另报包含客户端启动和
轮次间指标采样等待的全过程墙钟。模型启动不计入客户端计时，首次冷填充计入。
TTFT 从请求发出到首个非空输出块，包含排队；TPOT 用首块至流结束时间除以实际
输出 token 数减一，含客户端/网络尾部开销，采用原专项实验客户端口径。

核对失败、超时、抢占和逐请求实际输入/输出数；失败保留，未完成的臂不得记作
完整测试。用 `prompt_tokens_by_source` 区分本地计算、本地命中和外部恢复。
`kv_offload_load_bytes` 不能代替 `kv_tiering_fs_load_bytes` 的文件层实际读量。
各项性能、错误、压力与传输计数均保存在仓库外。

整理后的准备/客户端源码保留原输入及轮次协议，但尚未在昇腾完成新一轮整模型
测量。配置与过程可复跑，不保证相同百分比收益；独立重复应交替运行顺序，并
保留负结果和不同输入压力下的结果。
