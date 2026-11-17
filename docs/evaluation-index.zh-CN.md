# 适用范围与评测索引

本仓维护源码、配置和说明；原始记录、分析和归档清单集中在benchmark。

## 维护与版本

- MOD维护与交付负责人：@JieYang2001，职责定义沿用MAINTAINERS.md与MOD_METADATA.json。
- 原始实现及历史协作的归属见PROVENANCE.md，不将后续维护追认为全部原始贡献。
- 当前工作基于PR #11的工作集准入与生命周期实现；清理结果文件不改变插件运行代码。
- 对应宿主版本及显式修复边界见runtime-patches/README.md；安装不会自动修复宿主。

## 集中评测入口

- [集中补证 PR（Draft）](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/697)。

- [完整SWE回放报告、分析与原件清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/fbf4a916fb17e3923ed51a31a9318bcaecdfa6db/reports/tech-outcomes-20261117/kv-tiering)。
- [分项补证要求](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/issues/7#issuecomment-6846017531)。
- [组织补证总表](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)。

该报告是历史完整实验的补证草稿，公开9个完整实验臂、9885条请求的数值记录与离线复算入口。
完整输入、轨迹和SSE由保管人受控保存，位置、哈希与获取方式见报告中的controlled-location.json；
尚未独立验证审核者取件，不是已完成验收。无需将公开Release作为唯一归档方式。
归档状态以报告中的RAW_PUBLICATION_STATUS.md与raw-archive-manifest.json为准。
原始输入、流式响应及测量时插件payload已做离线一致性核对，没有新增设备测试。
复现配置见[运行说明](reproduction.zh-CN.md)。三层机制与两层回放应分别评估。

旧合成实验的陈述不追认为新版本证据；采用增补说明，不覆写原历史。
工作集准入是可选启发式机制，不保证任意负载受益。

## 五数据集适用性

以下状态指正式任务质量验收，不将历史回放等同于真实Agent任务执行。

|数据集|状态|机制适用与缺项|
|---|---|---|
|MMLU-Pro|not-exercised|可接推理链路；固定协议下是否存在前缀复用需确认，尚无正式准确率配对|
|HLE-Verified|not-exercised|尚无固定evaluator与任务质量配对，不凭数据集名称推断复用收益|
|SWE-bench-Pro|not-exercised（正式Agent验收）|专项历史回放触发过前缀恢复，但没有fresh-sandbox任务解决率配对|
|FrontierScience|not-exercised|尚缺Olympiad/Research各自固定输入及任务评分配对|
|Terminal-Bench 2.1|not-exercised|尚缺Harbor环境、工具执行与任务解决率配对|

默认官方输入、工具策略与任务质量约束不应为了触发缓存而改写。
当指定协议未触发恢复时保留零结果；另设专项场景须明确其独立身份。
