"""Host-independent, non-skipped checks for the explicit Ascend hybrid guard."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_hust_kv_tiering import host_compat


class MambaSpec:
    pass


class AttentionSpec:
    pass


class UniformTypeKVCacheSpecs:
    def __init__(self, **children):
        self.kv_cache_specs = children


class SupportsHMA:
    pass


class SupportsNonuniformHybridLocalHits:
    def can_use_nonuniform_hybrid_local_hits(self, request):
        return False


SPECS = SimpleNamespace(
    MambaSpec=MambaSpec, UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs
)


def cache(*specs):
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    )


class TestHostCompatibilityGuard(unittest.TestCase):
    def check(self, backend, config, base):
        def load(name):
            if name == "vllm.v1.kv_cache_interface":
                return SPECS
            if name == "vllm.distributed.kv_transfer.kv_connector.v1.base":
                return base
            raise AssertionError(f"unexpected import: {name}")

        with patch.object(host_compat, "import_module", side_effect=load):
            host_compat.require_safe_ascend_hybrid_prefix_host(
                {"ascend_copy_backend": backend}, config
            )

    def test_legacy_hma_only_host_rejected_for_both_ascend_backends(self):
        for backend in ("torch_async", "torch_sync"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(RuntimeError, "SupportsHMA alone"):
                    self.check(
                        backend,
                        cache(MambaSpec()),
                        SimpleNamespace(SupportsHMA=SupportsHMA),
                    )

    def test_repaired_api_allows_hybrid_without_opting_in(self):
        for backend in ("torch_async", "torch_sync"):
            self.check(
                backend,
                cache(AttentionSpec(), MambaSpec()),
                SimpleNamespace(
                    SupportsNonuniformHybridLocalHits=SupportsNonuniformHybridLocalHits
                ),
            )
        self.assertFalse(
            SupportsNonuniformHybridLocalHits().can_use_nonuniform_hybrid_local_hits(
                None
            )
        )

    def test_marker_without_callable_request_method_rejected(self):
        for marker in (
            type("MissingMethod", (), {}),
            object(),
            type("WrongMethod", (), {"can_use_nonuniform_hybrid_local_hits": False}),
        ):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(RuntimeError, "callable"):
                    self.check(
                        "torch_async",
                        cache(MambaSpec()),
                        SimpleNamespace(SupportsNonuniformHybridLocalHits=marker),
                    )

    def test_nested_uniform_mamba_is_guarded(self):
        config = cache(
            UniformTypeKVCacheSpecs(
                fa=AttentionSpec(), nested=UniformTypeKVCacheSpecs(ssm=MambaSpec())
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Unsafe Ascend hybrid"):
            self.check("torch_async", config, SimpleNamespace())

    def test_pure_attention_never_imports_repair_api(self):
        def load(name):
            self.assertEqual(name, "vllm.v1.kv_cache_interface")
            return SPECS

        with patch.object(host_compat, "import_module", side_effect=load) as mocked:
            host_compat.require_safe_ascend_hybrid_prefix_host(
                {"ascend_copy_backend": "torch_async"},
                cache(AttentionSpec(), UniformTypeKVCacheSpecs(fa=AttentionSpec())),
            )
        self.assertEqual(mocked.call_count, 1)

    def test_generic_cuda_omitted_backend_never_imports_host(self):
        with patch.object(
            host_compat, "import_module", side_effect=AssertionError("import")
        ):
            host_compat.require_safe_ascend_hybrid_prefix_host({}, cache(MambaSpec()))

    def test_import_failure_fails_closed_and_names_separate_core_fix(self):
        def load(name):
            if name == "vllm.v1.kv_cache_interface":
                return SPECS
            raise ImportError("legacy ABI")

        with patch.object(host_compat, "import_module", side_effect=load):
            with self.assertRaisesRegex(RuntimeError, "separate core"):
                host_compat.require_safe_ascend_hybrid_prefix_host(
                    {"ascend_copy_backend": "torch_sync"}, cache(MambaSpec())
                )

    def test_spec_guard_precedes_storage_registration_and_host_super(self):
        source = Path(host_compat.__file__).with_name("spec.py").read_text()
        tree = ast.parse(source)
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HustTieringOffloadingSpec"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
        guard = next(
            n
            for n in calls
            if isinstance(n.func, ast.Name)
            and n.func.id == "require_safe_ascend_hybrid_prefix_host"
        )
        storage = next(
            n
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "_register_storage"
        )
        host_super = next(
            n
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "__init__"
        )
        self.assertLess(guard.lineno, storage.lineno)
        self.assertLess(guard.lineno, host_super.lineno)


if __name__ == "__main__":
    unittest.main()
