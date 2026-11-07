# Provenance

Source archive:
[intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831)

Primary history:

- [PR #120: KV tiering lifecycle development](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/120)
- [PR #124: merged residency and lifecycle implementation](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/124)
- [PR #252: filesystem demand-load isolation guard](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/252)

The migration must reconcile rather than duplicate
[intellistream/vllm-agent-state-tiering](https://github.com/intellistream/vllm-agent-state-tiering).

## Isolated engineering reference

The reference plugin payload was frozen at
`5d4e1fd91d0a31137ea05e31439dcc5f46d3e05a`; subsequent prerequisite and packaging
commits leave its 18 Python/JSON payload files unchanged. The reference repaired
SDK pair is core `3bf4ba0b0ee547577b4e6bb772ccd45aa5793670` and Ascend
`b8ff9a5f0ae8f60980d895651f37dcd3808ce8d9`.

The [SDK patch manifest](<runtime-patches/manifest.json>) records exact original
bases, patch hashes, and forward/reverse application checks. The explicit patches
are not automatically applied by plugin import or installation. Original runtime
workspaces, distribution metadata, CANN paths and unrelated processes were
preserved during validation; no push or PR is implied by local engineering
commits. See the [scoped validation record](<docs/validation-status.md>) rather
than inferring production certification from version strings or component tests.

