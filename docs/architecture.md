# KV tiering runtime design

## Scope and invariants

This is an out-of-tree **prefix/state cache**, not a mechanism that makes an
arbitrarily large active attention working set fit in HBM. The host remains the
owner of scheduling, hybrid-cache alignment and GPU-block reuse fences. Dropping
a cache opportunity may cause recomputation; it must never change model results.

The plugin owns admission policy, lifecycle metadata, conservative CPU reclamation,
Ascend copy lifetime, and the filesystem backend. It does not patch the host or
replace host transfer algorithms. A small, explicitly version-bound host adapter
is needed for selective CPU cache reclamation: the frozen host does not expose a
public remove-by-key operation. That coupling must be tested, not concealed by
claiming every imported interface is public.

Safety invariants:

* A CPU slot is never reclaimed while a device/secondary transfer holds it.
* A live cache copy is demoted for capacity only after at least one secondary
  store has **completed successfully**, not merely been submitted.
* A failed store is never published as an existing secondary copy.
* Failed device fences are fatal to the operation; they do not authorize reuse.
* TTL deletion never races an already-submitted transfer or a new publication.
* Shared keys remain retained until all tracked owning sessions release them.
* Cache metadata, transfer admission and filesystem capacity have explicit bounds.
* A full/unavailable cache degrades to a missed optimization, not fabricated KV.

## Placement decisions

HBM admission uses the public `bind_gpu_block_pool` connector hook. Its O(1)
pressure signal is the maximum of `BlockPool.get_usage()` (non-evictable allocation
pressure) and the physical prefix-cache block fraction from
`cached_block_hashes_by_block`. This includes evictable cached prefixes: using
allocation pressure alone misses prefix-cache turnover at low active concurrency.
Taking the maximum avoids double-counting active cached blocks and is a
conservative lower bound on the resident union, **not total device-memory usage**.
A high/low hysteresis gate opens at the high watermark and closes at the low
watermark. A preemption is an additional pressure signal. Missing pressure
binding/index support is an explicit compatibility error when pressure mode is enabled.

The plugin scheduler subclasses the host scheduler only at its store-admission
hook. When the gate is closed it postpones host store-job construction, preserving
host prefix progress indices rather than silently skipping blocks forever.
No partial arbitrary subset of hybrid-group keys is passed through the host's
store path: the host advances all group cursors after an admitted request.
The per-step block budget is consequently soft for the first whole request;
concurrent-job and CPU-slot limits remain hard. This avoids starvation and
prefix holes. Admission remains best effort, not a promise to save every prefix.

Optional session-return and calibrated cost gates are conservative heuristics.
The return fraction is historical session reuse, not a learned probability or a
claim of future hits. A cost gate is active only with supplied, positive rates;
it compares expected recomputation avoided with write plus restore costs. No
unmeasured bandwidth/prefill constants are presented as online measurements.

CPU occupancy counts **all resident/reserved slots**, not only pinned slots.
High/low hysteresis and idle age drive coldest-first spill/reclamation. Normal
CPU-to-FS cascades can be deferred until CPU pressure; explicit spill jobs pin
CPU slots and release them only after completion. The primary pool is given
free slots before accepting new device stores, avoiding accidental eviction of
unpersisted victims by the host policy. Under storage failure or exhausted quota,
admission backs off and inference can recompute rather than wait indefinitely.

## Session and lifecycle state

A request supplies `session_id`, `conversation_id` or `kv_session_id` through
`kv_transfer_params`; otherwise its request ID is ephemeral. Concurrent requests
in a session share ownership. Lookup/touch/load/store all contribute ownership,
including cache hits, not only new stores.

Request finish is not transfer finish. Finalization occurs only after outstanding
primary and secondary work is accounted for. Stable idle sessions retain metadata
until TTL or the configured metadata budget; ephemeral sessions release request
metadata after finalization. Capacity admission rejects new cache tracking rather
than growing without bound or discarding active ownership silently.

TTL is checked on scheduler activity and before new lookups. It does not keep the
engine spinning until a future deadline. It is a cache-retention policy, not a
wall-clock secure-erasure SLA. Optional secondary deletion is asynchronous;
pending keys are tombstoned against lookup/new publication until deletion
completes. CPU removal checks actual host readiness/refcounts again. Deletion
backlogs are bounded too; storage/deletion failures back off, and new cache
metadata may be declined rather than growing without bound. A full disk with
secondary deletion disabled can stop further FS admissions; live entries are
not silently evicted to create space. Provision an appropriate retention/quota
profile or reclaim an owned cache namespace in a maintenance window.

Session IDs are cache hints, not authentication. Applications must namespace and
authorize them per tenant and preserve the host's cache-salt isolation contract.
The plugin does not provide a security boundary or secure physical erasure.

## Observability and validation

Keep device-to-CPU, CPU-to-device and filesystem read/write counters separate.
Only FS-side completed reads establish FS recovery; device load bytes alone do
not. Cache hit benefit may come from DDR rather than SSD. Measure Native,
CPU-only, and CPU+FS independently under a frozen request contract.

CPU tests validate policy hysteresis, bounds, ownership, lifecycle and injected
errors. Matched-host integration tests validate the actual ABI and filesystem
roundtrip. Real NPU tests validate byte/state coherence and fences. Submit latency
and stream traces, followed by repeated paired serving measurements, are needed
to establish overlap and performance. Passing lint/import/build is not runtime
qualification. No throughput gain is promised by this design alone.
