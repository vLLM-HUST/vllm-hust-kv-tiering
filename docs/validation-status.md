# Validation boundaries

The plugin remains experimental. Qualification is specific to the exact source,
SDK, model, layout and configuration. The source repository contains test source
and configuration instructions; generated test reports and serving measurements
are maintained externally.

## Checks to run

- Portable policy, lifecycle, transfer-ownership and filesystem safety tests.
- Full spec/connector/CPU/filesystem contracts with the matching real host.
- Explicitly enabled NPU copy/layout tests on isolated hardware.
- Independent full-model cold/cached prefix and hybrid checkpoint correctness.
- Paired serving comparisons with complete inputs, actual recovery and all costs.

A lower-level check does not establish the next check. In particular, a CPU/FS
roundtrip does not prove NPU restoration; an H2D counter alone does not prove a
filesystem read; and a cache hit does not establish a net serving benefit.

## Host prerequisites

Use the exact SDK bases/repairs in [runtime-patches](../runtime-patches/README.md).
The hybrid path requires safe unified local hits, producer-bound recurrent
feedback, valid state-copy ABI and correct prefill/decode graph eligibility.
The plugin rejects known unsafe host contracts rather than fabricating support.
The API marker itself is not full-model qualification.

The source-bound reference and experiment settings are documented in
[the reproduction guide](reproduction.zh-CN.md). The two-tier SWE profile uses
eager execution without speculative decoding. It must not be described as a
three-tier, graph or MTP serving result. The ECPA discovery version constraint
is separate from direct connector activation.

## Outstanding scope

A new measurement is required after changing source, host ABI, device runtime,
model/template, cache geometry or policy. Component checks do not establish
arbitrary-context answer quality, durable crash recovery, true copy/compute
overlap, long-running stability or workload-independent acceleration.

Working-set admission is an optional heuristic. Its bounded observations do not
measure the entire future workload or automatically calibrate recomputation and
I/O cost. Compare workloads with different reuse and pressure, preserve negative
results externally, and report the complete input denominator.

Filesystem CRC verifies bytes rather than semantic prefix identity. TTL and
logical CPU reclamation do not promise active-HBM eviction, secure erasure or
mmap RSS unmapping. Durability settings and the operating system's page cache
must be recorded separately when evaluating the filesystem tier.
