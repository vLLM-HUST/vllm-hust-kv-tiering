# Full-HBM SWE replay evidence

Read ../../../../docs/performance-validation-20261117.zh-CN.md for results and limitations. The128-task full trial and user-shortened1142-request confirmation support a scoped benefit; the32-task repeats did not. These are specialized reusable-prefix replays, not official zero-reuse scores or live-agent quality.

Analysis JSON retains all arms, per-round metrics, input signature digests, counters, timing definitions and source accounting. Three archive receipts identify privately retained full request/SSE/server evidence; the raw archives are not published by this PR. Never infer complete second-run validation from the selected prefix.

Reproduction scripts are the actual experiment sources, including frozen paths. They are an audit trail, not a turnkey command for another host: replace ROOT/PARENT/runtime paths, recreate the fixed runtime and source snapshots, preserve serving arguments, and use a fresh result directory. Controllers refuse existing launch markers; never remove original markers to rerun. The128GiB probe did not run because /dev/shm was16GiB.

The plugin snapshot manifest identifies the measured64 files before adding this evidence. Additional report/script files do not alter runtime payload. Native and candidate both use full automatic HBM utilization0.92 and14GiB additional CPU only for candidate. Use matched source files, not just the ancestor commit. See the copied supervisor/client and runtime-patches manifest for host dependencies.
