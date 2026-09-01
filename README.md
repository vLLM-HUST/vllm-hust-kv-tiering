# vLLM-HUST KV Tiering

Owner-maintained extraction of the tiered KV-cache residency and lifecycle
work preserved in the archived vLLM-HUST repository.

**Status: migration scaffold. This repository does not yet contain an
installable or runnable release.**

The target is a KV residency provider for device, CPU, and filesystem tiers.
Extension Manager integration will provide discovery, compatibility checks,
configuration rendering, and health checks. It will not delete retained KV
data or silently manage shared storage.

See [PROVENANCE.md](PROVENANCE.md) and [MAINTAINERS.md](MAINTAINERS.md).

