# SPDX-License-Identifier: Apache-2.0
"""Lifecycle-aware wrapper around the host tiering offloading manager.

The host ``TieringOffloadingManager`` already drives the store/load/promotion
flow. This module layers :class:`SessionLifecycleManager` on top of it so the
session-retention / TTL-expiry logic is actually driven by the running
offloading flow instead of remaining un-referenced.

The wrapper is a thin adapter: every method not overridden here is delegated to
the wrapped manager through ``__getattr__``, so it does not fork or copy the
host tiering manager.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)

from vllm_hust_kv_tiering.lifecycle import SessionLifecycleManager


class LifecycleAwareManager:
    """Delegate to a host manager while driving lifecycle hooks.

    The hooks are attached at the points where the offloading flow already has
    the request context and block keys:

    - ``on_new_request`` / ``on_request_finished`` advance session state.
    - ``prepare_store`` records block ownership for the session.
    - ``on_schedule_end`` expires idle sessions and deletes their unreferenced
      blocks from secondary tiers.
    """

    def __init__(
        self,
        delegate: Any,
        lifecycle: SessionLifecycleManager,
    ) -> None:
        self._delegate = delegate
        self._lifecycle = lifecycle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._lifecycle.on_new_request(req_context)
        return self._delegate.on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lifecycle.on_request_finished(req_context)
        self._delegate.on_request_finished(req_context)

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext):
        self._lifecycle.record_request_keys(req_context, keys)
        return self._delegate.prepare_store(keys, req_context)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        if self._lifecycle.enabled:
            result = self._lifecycle.expire_idle_sessions()
            if result.unreferenced_keys:
                self._delete_keys(result.unreferenced_keys)
        self._delegate.on_schedule_end(context)

    def _delete_keys(self, keys: Collection[OffloadKey]) -> None:
        for tier in getattr(self._delegate, "secondary_tiers", []):
            delete = getattr(tier, "delete", None)
            if delete is not None:
                delete(keys)
