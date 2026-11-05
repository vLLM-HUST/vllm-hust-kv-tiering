"""CPU-only safety tests of the actual worker, without torch/vllm/pytest.

Run with ``python -m unittest discover -s tests -p test_async_worker_safety.py``
(or pytest). Import stubs exist only during isolated module execution; importing
this test module never installs fake torch/vllm modules into sys.modules.
"""

import importlib.util
import logging
import sys
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


@dataclass
class _Result:
    job_id: int
    success: bool
    transfer_size: int | None = None
    transfer_time: float | None = None


@contextmanager
def loaded_worker(runtime):
    """Module-loading fixture context; restore imports before running tests."""
    torch = ModuleType("torch")
    torch.int8 = object()
    torch.npu = runtime
    logger = ModuleType("vllm.logger")
    logger.init_logger = lambda name: logging.getLogger(name)
    base = ModuleType("vllm.v1.kv_offload.base")
    base.OffloadingWorker = object
    base.TransferResult = _Result
    names = ("vllm", "vllm.v1", "vllm.v1.kv_offload")
    stubs = {name: ModuleType(name) for name in names}
    stubs.update({"torch": torch, "vllm.logger": logger, base.__name__: base})
    path = (
        Path(__file__).resolve().parents[1]
        / "src/vllm_hust_kv_tiering/ascend_worker.py"
    )
    name = "_isolated_async_worker_safety"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    stubs[name] = module
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    # The worker's globals retain its runtime, not the process import table.
    yield module


class FakeView:
    def __init__(self, tensor, indices):
        self.tensor = tensor
        self.indices = indices

    def numel(self):
        return len(self.indices)

    def copy_(self, source, non_blocking):
        runtime = self.tensor.runtime
        runtime.assert_owned()
        runtime.non_blocking.append(non_blocking)
        runtime.copy_calls += 1
        runtime.stream_obj.operations.append((self, source))
        # Failure after enqueue is deliberately stricter than ordinary errors.
        if runtime.copy_calls == runtime.fail_copy_at:
            raise RuntimeError("copy enqueue failure")
        return self

    def values(self):
        return [self.tensor.data[index] for index in self.indices]


class FakeTensor:
    def __init__(self, runtime, rows, width, device, values=None):
        self.runtime = runtime
        self.shape = (rows, width)
        self.device = device
        self.data = list(values) if values is not None else [0] * (rows * width)

    def view(self, *shape):
        return self

    def __getitem__(self, key):
        row, columns = key if isinstance(key, tuple) else (key, slice(None))
        start, stop, step = columns.indices(self.shape[1])
        indices = [row * self.shape[1] + i for i in range(start, stop, step)]
        return FakeView(self, indices)


class FakeStream:
    def __init__(self, runtime):
        self.runtime = runtime
        self.operations = []
        self.executed = 0
        self.sync_calls = 0
        self.wait_calls = 0

    def wait_stream(self, other):
        self.runtime.assert_owned()
        self.wait_calls += 1
        if self.runtime.fail_wait_stream:
            raise RuntimeError("wait_stream failure")

    def advance(self, count=None):
        end = len(self.operations) if count is None else self.executed + count
        while self.executed < end:
            destination, source = self.operations[self.executed]
            for index, value in zip(destination.indices, source.values(), strict=True):
                destination.tensor.data[index] = value
            self.executed += 1

    def synchronize(self):
        self.sync_calls += 1
        if self.runtime.fail_stream_sync:
            raise RuntimeError("stream synchronize failure")
        self.advance()


class FakeEvent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.stream = None
        self.target = None
        self.sync_calls = 0

    def record(self, stream):
        self.runtime.assert_owned()
        if self.runtime.fail_record:
            raise RuntimeError("event record failure")
        self.stream = stream
        self.target = len(stream.operations)

    def query(self):
        if self.runtime.fail_query:
            raise RuntimeError("event query failure")
        return self.stream.executed >= self.target

    def synchronize(self):
        self.sync_calls += 1
        if self.runtime.fail_event_sync:
            raise RuntimeError("event synchronize failure")
        self.stream.advance(self.target - self.stream.executed)


class FakeRuntime:
    def __init__(self):
        self.device = SimpleNamespace(type="npu")
        self.stream_obj = FakeStream(self)
        self.events = []
        self.worker = None
        self.non_blocking = []
        self.copy_calls = 0
        self.fail_copy_at = None
        self.fail_event_create = False
        self.fail_record = False
        self.fail_query = False
        self.fail_event_sync = False
        self.fail_stream_sync = False
        self.fail_wait_stream = False
        self.fail_context_exit = False

    def assert_owned(self):
        assert self.worker.pending
        assert self.worker._inflight
        assert all(entry.spans for entry in self.worker._inflight.values())

    def Stream(self, device):
        return self.stream_obj

    def Event(self):
        self.assert_owned()
        if self.fail_event_create:
            raise RuntimeError("event creation failure")
        event = FakeEvent(self)
        self.events.append(event)
        return event

    def current_stream(self):
        return object()

    @contextmanager
    def stream(self, stream):
        yield
        if self.fail_context_exit:
            raise RuntimeError("stream context exit failure")


class FakeRegion:
    def __init__(self, runtime):
        self.runtime = runtime
        self.views = []
        self.cleanup_calls = 0
        self.fail_cleanup = False

    def create_next_view(self, width):
        tensor = FakeTensor(self.runtime, 4, width, SimpleNamespace(type="cpu"))
        self.views.append(tensor)
        return tensor

    def cleanup(self):
        stream = self.runtime.stream_obj
        assert stream.executed == len(stream.operations), "unmapping active DMA"
        self.cleanup_calls += 1
        if self.fail_cleanup:
            raise RuntimeError("cleanup failure")
        self.views.clear()


class AsyncWorkerSafetyTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.loading = loaded_worker(self.runtime)
        self.module = self.loading.__enter__()
        self.region = FakeRegion(self.runtime)
        self.tensor = FakeTensor(self.runtime, 4, 4, self.runtime.device, range(16))
        refs = [[SimpleNamespace(tensor_idx=0, page_size_bytes=4)]]
        caches = SimpleNamespace(
            group_data_refs=refs,
            tensors=[SimpleNamespace(tensor=self.tensor, page_size_bytes=4)],
        )
        self.worker = self.module.AscendCopyWorker(caches, 1, 4, self.region)
        self.runtime.worker = self.worker
        self.gpu = SimpleNamespace(block_ids=[0, 1], group_sizes=[2], block_indices=[0])
        self.cpu = SimpleNamespace(block_ids=[1, 2])

    def tearDown(self):
        self.runtime.fail_stream_sync = False
        self.region.fail_cleanup = False
        self.worker.shutdown()
        self.loading.__exit__(None, None, None)

    def submit(self, job=1):
        return self.worker.submit_store(job, self.gpu, self.cpu)

    def assert_fatal(self):
        self.assertIsNotNone(self.worker._failure)
        for operation in (
            lambda: self.submit(99),
            self.worker.get_finished,
            lambda: self.worker.wait({1}),
        ):
            with self.assertRaisesRegex(RuntimeError, "fatal"):
                operation()

    def test_delayed_event_and_exactly_once_poll(self):
        self.assertTrue(self.submit())
        self.assertEqual(self.worker.get_finished(), [])
        self.assertEqual(self.region.views[0].data[4:12], [0] * 8)
        self.runtime.stream_obj.advance(1)
        self.assertEqual(self.worker.get_finished(), [])
        self.runtime.stream_obj.advance()
        results = self.worker.get_finished()
        self.assertEqual(
            [(r.job_id, r.success, r.transfer_size) for r in results], [(1, True, 8)]
        )
        self.assertGreaterEqual(results[0].transfer_time, 0)
        self.assertEqual(self.region.views[0].data[4:12], list(range(8)))
        self.assertEqual(self.worker.get_finished(), [])
        self.worker.wait({1, 1234})
        self.assertEqual(self.worker.get_finished(), [])
        self.assertEqual(self.runtime.non_blocking, [True, True])
        self.assertEqual(self.runtime.stream_obj.wait_calls, 1)

    def test_load_and_repeated_wait_report_once(self):
        self.region.views[0].data[4:12] = list(range(30, 38))
        self.assertTrue(self.worker.submit_load(1, self.cpu, self.gpu))
        self.assertEqual(self.tensor.data[:8], list(range(8)))
        self.worker.wait({1})
        self.worker.wait({1})
        self.assertEqual(self.runtime.events[0].sync_calls, 1)
        self.assertEqual(self.tensor.data[:8], list(range(30, 38)))
        self.assertEqual([r.job_id for r in self.worker.get_finished()], [1])
        self.assertEqual(self.worker.get_finished(), [])
        self.assertEqual(self.runtime.stream_obj.wait_calls, 0)

    def test_duplicate_inflight_and_completed_unconsumed(self):
        self.submit()
        for fenced in (False, True):
            if fenced:
                self.worker.wait({1})
            with self.assertRaisesRegex(ValueError, "pending or unconsumed"):
                self.submit()
        self.assertEqual(self.runtime.copy_calls, 2)
        self.assertEqual([r.job_id for r in self.worker.get_finished()], [1])
        self.assertTrue(self.submit())  # Reuse only after result consumption.
        self.worker.wait({1})
        self.assertEqual([r.job_id for r in self.worker.get_finished()], [1])

    def test_validation_failure_does_not_enqueue_or_poison_worker(self):
        self.cpu.block_ids = [1, 99]
        with self.assertRaisesRegex(ValueError, "outside allocated"):
            self.submit()
        self.assertEqual(self.runtime.copy_calls, 0)
        self.assertEqual(self.worker.pending, set())
        self.assertEqual(self.worker._inflight, {})
        self.assertIsNone(self.worker._failure)
        self.cpu.block_ids = [1, 2]
        self.assertTrue(self.submit())

    def test_partial_enqueue_failure_drains_but_remains_terminal(self):
        self.runtime.fail_copy_at = 2
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.submit()
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assertEqual(self.runtime.stream_obj.sync_calls, 1)
        self.assertEqual(self.worker.pending, {1})
        self.assertEqual(len(self.worker._inflight[1].spans), 2)
        self.assertEqual(self.worker.completed, [])
        self.assert_fatal()

    def test_partial_enqueue_and_drain_failure_retain_all_ownership(self):
        self.runtime.fail_copy_at = 2
        self.runtime.fail_stream_sync = True
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.submit()
        views = self.worker.cpu_tensors[:]
        entry = self.worker._inflight[1]
        self.assertEqual(self.runtime.stream_obj.executed, 0)
        self.assert_fatal()
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            self.worker.shutdown()
        self.assertEqual(self.worker.cpu_tensors, views)
        self.assertIs(self.worker._inflight[1], entry)
        self.assertEqual(self.worker.pending, {1})
        self.assertEqual(self.region.cleanup_calls, 0)
        self.runtime.fail_stream_sync = False
        self.worker.shutdown()
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assertEqual(self.region.cleanup_calls, 1)

    def assert_pre_copy_failure_tracked(self, attribute):
        setattr(self.runtime, attribute, True)
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.submit()
        self.assertIn(1, self.worker._inflight)
        self.assertEqual(self.worker.completed, [])
        self.assertEqual(self.runtime.copy_calls, 0)
        self.assert_fatal()

    def test_event_creation_error_is_tracked(self):
        self.assert_pre_copy_failure_tracked("fail_event_create")

    def test_ordering_error_is_tracked(self):
        self.assert_pre_copy_failure_tracked("fail_wait_stream")

    def test_event_record_failure_never_fabricates_completion(self):
        self.runtime.fail_record = True
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.submit()
        self.assertEqual(self.runtime.copy_calls, 2)
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assertIsNone(self.runtime.events[0].target)
        self.assertEqual(self.worker.completed, [])
        self.assert_fatal()

    def test_event_record_and_drain_failure_keep_unrecorded_job_owned(self):
        self.runtime.fail_record = True
        self.runtime.fail_stream_sync = True
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.submit()
        self.assertIsNone(self.worker._inflight[1].event.target)
        self.assertEqual(len(self.worker._inflight[1].spans), 2)
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            self.worker.shutdown()
        self.assertEqual(self.region.cleanup_calls, 0)
        self.assertEqual(self.runtime.stream_obj.executed, 0)
        self.runtime.fail_stream_sync = False
        self.worker.shutdown()
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assertEqual(self.region.cleanup_calls, 1)

    def test_context_exit_failure_drains_recorded_event(self):
        self.runtime.fail_context_exit = True
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.submit()
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assert_fatal()

    def test_query_failure_is_fatal_even_if_stream_drain_succeeds(self):
        self.submit()
        self.runtime.fail_query = True
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.worker.get_finished()
        self.assertEqual(self.worker.completed, [])
        self.assertIn(1, self.worker._inflight)
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.runtime.fail_query = False
        self.assert_fatal()

    def test_query_and_drain_failure_preserve_event_and_views(self):
        self.submit()
        event = self.worker._inflight[1].event
        self.runtime.fail_query = True
        self.runtime.fail_stream_sync = True
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.worker.get_finished()
        self.assertIs(self.worker._inflight[1].event, event)
        self.assertEqual(self.worker.pending, {1})
        self.assertTrue(self.worker.cpu_tensors)
        self.assert_fatal()
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            self.worker.shutdown()
        self.assertEqual(self.region.cleanup_calls, 0)

    def test_wait_failure_raises_fence_and_does_not_pop_job(self):
        self.submit()
        event = self.worker._inflight[1].event
        self.runtime.fail_event_sync = True
        self.runtime.fail_stream_sync = True
        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            self.worker.wait({1})
        self.assertIs(self.worker._inflight[1].event, event)
        self.assertEqual(self.worker.pending, {1})
        self.assertEqual(self.worker.completed, [])
        self.assert_fatal()

    def test_wait_failure_with_successful_drain_still_raises(self):
        self.submit()
        self.runtime.fail_event_sync = True
        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            self.worker.wait({1})
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assertEqual(self.worker.completed, [])
        self.assert_fatal()

    def test_error_after_other_completion_never_loses_reserved_id(self):
        self.submit(1)
        self.submit(2)
        self.runtime.stream_obj.advance(2)
        with (
            patch.object(
                self.runtime.events[1],
                "query",
                side_effect=RuntimeError("query failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "query failed"),
        ):
            self.worker.get_finished()
        self.assertEqual([r.job_id for r in self.worker.completed], [1])
        self.assertEqual(self.worker.pending, {1, 2})
        self.assertEqual(set(self.worker._inflight), {2})
        self.assert_fatal()

    def test_shutdown_drains_without_trusting_events_and_is_idempotent(self):
        self.submit()
        self.runtime.fail_event_sync = True
        self.worker.shutdown()
        self.assertEqual(self.runtime.stream_obj.executed, 2)
        self.assertEqual(self.runtime.events[0].sync_calls, 0)
        self.assertEqual(self.region.cleanup_calls, 1)
        self.worker.shutdown()
        self.assertEqual(self.region.cleanup_calls, 1)
        self.assertEqual(self.worker.pending, set())
        self.assertEqual(self.worker._inflight, {})
        for operation in (
            self.submit,
            self.worker.get_finished,
            lambda: self.worker.wait({1}),
        ):
            with self.assertRaisesRegex(RuntimeError, "closing or closed"):
                operation()

    def test_shutdown_failure_can_be_retried_without_unmapping(self):
        self.submit()
        self.runtime.fail_stream_sync = True
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "ownership retained"):
                self.worker.shutdown()
            self.assertTrue(self.worker.device_tensors)
            self.assertTrue(self.worker.cpu_tensors)
            self.assertIn(1, self.worker._inflight)
            self.assertEqual(self.region.cleanup_calls, 0)
        with self.assertRaisesRegex(RuntimeError, "closing or closed"):
            self.submit(2)
        self.runtime.fail_stream_sync = False
        self.worker.shutdown()
        self.worker.shutdown()
        self.assertEqual(self.region.cleanup_calls, 1)

    def test_cleanup_failure_keeps_admission_closed_and_can_retry(self):
        self.submit()
        self.region.fail_cleanup = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
            self.worker.shutdown()
        self.assertTrue(self.worker._stream_drained)
        with self.assertRaisesRegex(RuntimeError, "closing or closed"):
            self.submit(2)
        self.region.fail_cleanup = False
        self.worker.shutdown()
        self.worker.shutdown()
        self.assertEqual(self.region.cleanup_calls, 2)

    def test_empty_shutdown_needs_no_runtime_fence(self):
        self.worker.shutdown()
        self.worker.shutdown()
        self.assertEqual(self.runtime.stream_obj.sync_calls, 0)
        self.assertEqual(self.region.cleanup_calls, 1)

    def test_layout_snapshot_survives_failed_close_without_mutating_owner(self):
        self.worker.shutdown()
        self.region = FakeRegion(self.runtime)
        rows = [self.tensor]
        layout = SimpleNamespace(
            group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=4)]],
            layers=[(0, rows)],
            page_sizes=[4],
        )
        self.worker = self.module.AscendLayoutWorker(layout, 1, 4, self.region)
        self.runtime.worker = self.worker
        # Changes to container metadata must not invalidate the bound views.
        rows.clear()
        layout.layers.clear()
        layout.page_sizes.clear()
        self.submit()
        snapshot = self.worker.layers
        self.runtime.fail_stream_sync = True
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            self.worker.shutdown()
        self.assertIs(self.worker.layers, snapshot)
        self.assertEqual(self.region.cleanup_calls, 0)
        self.runtime.fail_stream_sync = False
        self.worker.shutdown()
        self.assertEqual(self.worker.layers, ())
        self.assertEqual(layout.layers, [])

    def test_shutdown_does_not_clear_callers_layout(self):
        self.worker.shutdown()
        self.region = FakeRegion(self.runtime)
        layers = [(0, [self.tensor])]
        layout = SimpleNamespace(group_data_refs=[], layers=layers, page_sizes=[4])
        self.worker = self.module.AscendLayoutWorker(layout, 1, 4, self.region)
        self.worker.shutdown()
        self.assertEqual(layout.layers, layers)
        self.assertEqual(len(layers), 1)
        self.assertIs(layers[0][1][0], self.tensor)


class ImportIsolationTests(unittest.TestCase):
    def test_import_stubs_restore_existing_modules_before_fixture_yields(self):
        names = (
            "torch",
            "vllm",
            "vllm.logger",
            "vllm.v1",
            "vllm.v1.kv_offload",
            "vllm.v1.kv_offload.base",
        )
        before = {name: sys.modules.get(name) for name in names}
        with loaded_worker(FakeRuntime()) as module:
            self.assertTrue(hasattr(module, "AscendCopyWorker"))
            self.assertEqual({name: sys.modules.get(name) for name in names}, before)
            self.assertNotIn("_isolated_async_worker_safety", sys.modules)
        self.assertEqual({name: sys.modules.get(name) for name in names}, before)


if __name__ == "__main__":
    unittest.main()
