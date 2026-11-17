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

The opt-in candidate working-set gate adds a CPU-capacity admission decision
before new D2H allocation or explicit spill. It uses a bounded sliding window
of full host offload keys and actual CPU slots, rejecting large cold fills when
the observed set exceeds capacity (or history is censored) and the insertion
cannot use free slots. A request's cumulative submitted bytes bound incremental
fills; rejection lasts until request finalization. Existing copies still reach
the host's ordinary no-op/readiness path. The offered key list is never split
or rewritten, preserving the host's mixed Attention/GDN checkpoint contract.
The candidate set is a heuristic, not the entire Agent working set or a proof
of future reuse. Configuration, telemetry stages and remaining qualification
requirements are described in [configuration](<configuration.md>).

HBM admission uses the public `bind_gpu_block_pool` connector hook. Its O(1)
pressure proxy is the maximum of `BlockPool.get_usage()` (non-evictable allocation
pressure) and `len(pool.cached_block_hash_to_block) / (pool.num_gpu_blocks - 1)`,
clamped to `[0, 1]`. The frozen primary lookup index has an O(1) key count and is
fetched again on every sample because cache reset replaces the index object.
`cached_block_hashes_by_block` contains only supplemental aliases: ordinary cached
prefixes need not appear there, so using it would incorrectly report idle cache
pressure as zero.

The primary index includes ordinary evictable prefixes, but its length counts
**cached keys**, not physical blocks. Alias keys can overcount physical occupancy;
duplicate physical blocks sharing a key can undercount it. This is not an exact
resident union, a guaranteed union lower bound, or total device-memory usage.
Taking the maximum avoids adding two overlapping proxies. A high/low hysteresis
gate opens at the high watermark and closes at the low watermark. Preemption is
an additional pressure signal. Missing binding/primary-index ABI support is an
explicit compatibility error, not a fabricated low-pressure sample.

The plugin scheduler subclasses the host scheduler only at its store-admission
hook. When the gate is closed it postpones host store-job construction, preserving
host prefix progress indices rather than silently skipping blocks forever.
No partial arbitrary subset of hybrid-group keys is passed through the host's
store path: the host advances all group cursors after an admitted request.
The per-step block budget is consequently soft for the first whole request;
concurrent-job and CPU-slot limits remain hard. This avoids starvation and
prefix holes. Admission remains best effort, not a promise to save every prefix.

Optional session/content-return and calibrated cost gates are conservative
heuristics. Content history counts distinct requests per full offload key,
including misses and rejected stores, and survives anonymous finalization. It
uses an LRU bounded by the block metadata budget; callbacks from the same request
do not inflate counts. A batch uses its least-observed key, then combines that
count with the session return count. The resulting fraction is a historical hint,
not a learned probability or a claim of future hits. A cost gate is active only
with supplied, positive rates; it compares expected recomputation avoided with
write plus restore costs. No unmeasured bandwidth/prefill constants are presented
as online measurements.

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
primary and secondary work is accounted for. Finalization releases request
metadata. With positive TTL, both stable and anonymous idle ownership survive
until expiration; a metadata cap cannot silently discard an expiration obligation.
With TTL disabled, anonymous ownership is released at finalization and finalized
idle stable-session hints may be forgotten under metadata pressure without
implying file deletion. Capacity admission rejects new cache tracking rather than
growing without bound or discarding protected ownership silently.

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
