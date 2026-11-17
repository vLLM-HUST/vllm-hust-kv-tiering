# Provenance

The plugin originates from
[intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831).

Primary implementation history:

- [PR #120: KV tiering lifecycle development](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/120)
- [PR #124: residency and lifecycle implementation](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/124)
- [PR #252: filesystem demand-load isolation guard](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/252)

Related implementation:
[intellistream/vllm-agent-state-tiering](https://github.com/intellistream/vllm-agent-state-tiering).

The version-bound SDK prerequisites are described in
[runtime-patches](runtime-patches/README.md). They are explicit dependency source
patches, never applied by plugin installation or import.

Experiment configuration and input-source references belong in
[the reproduction guide](docs/reproduction.zh-CN.md). Generated datasets,
measurements, logs and evidence archives belong outside this plugin repository;
new public experimental records should be maintained in
[vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark).
