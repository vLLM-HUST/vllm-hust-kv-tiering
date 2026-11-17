# SPDX-License-Identifier: Apache-2.0
"""Plugin-specific data types for the filesystem secondary tier.

The shared tiering interfaces (``JobMetadata``, ``JobResult``,
``SecondaryTierManager``) and the async lookup / thread-pool / block-I/O helpers
are provided by the host under ``vllm.v1.kv_offload.tiering.*`` and imported
directly by the manager; this module only holds the type the plugin adds on top
of the host.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey


@dataclass
class TierDeleteResult:
    """Result of explicit durable-copy deletion from a secondary tier."""

    removed_keys: set[OffloadKey]
    deleted_count: int = 0
