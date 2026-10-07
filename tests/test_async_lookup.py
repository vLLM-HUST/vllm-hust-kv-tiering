import queue
import threading
from types import SimpleNamespace

import pytest

from vllm_hust_kv_tiering.async_lookup import AsyncLookupManager


@pytest.mark.parametrize("hit", [False, True])
def test_result_arriving_after_first_retry_is_consumed(hit):
    release = threading.Event()
    entered = threading.Event()
    delivered = threading.Event()

    class Results(queue.SimpleQueue):
        def put(self, item):
            super().put(item)
            delivered.set()

    class DelayedLookup(AsyncLookupManager):
        def batch_lookup(self, keys, req_context):
            entered.set()
            assert release.wait(5)
            return [hit] * len(keys)

    manager = DelayedLookup("test")
    manager._pending_results = Results()
    request = SimpleNamespace(req_id="delayed")
    try:
        assert manager.lookup(b"key", request) is None
        manager.flush()
        assert entered.wait(5)
        assert manager.lookup(b"key", request) is None
        manager.flush()  # No new keys; the outstanding lookup must still progress.
        release.set()
        assert delivered.wait(5)
        assert manager.lookup(b"key", request) is hit
        manager.cleanup(request.req_id)
        assert not manager._lookup_state
    finally:
        release.set()
        manager.shutdown()
