"""Fail closed for the known frozen Ascend hybrid-prefix scheduler defect.

This recognizes the separate core repair API, not arbitrary host correctness.
No accelerator import, transfer resource, or connector capability opt-in occurs.
"""

from importlib import import_module


def require_safe_ascend_hybrid_prefix_host(extra_config, kv_cache_config):
    """Check the explicit Ascend-copy profile before any storage initialization.

    Generic host/CUDA profiles and pure-attention caches are unchanged. HMA
    support alone does not promise restoration of omitted recurrent checkpoints.
    The plugin does not opt in to nonuniform hybrid local hits: it needs the
    repaired host's default unified-local-HMA lookup path.
    """
    if extra_config.get("ascend_copy_backend") not in ("torch_async", "torch_sync"):
        return

    specs = import_module("vllm.v1.kv_cache_interface")

    def contains_mamba(spec):
        if isinstance(spec, specs.MambaSpec):
            return True
        if isinstance(spec, specs.UniformTypeKVCacheSpecs):
            return any(contains_mamba(child) for child in spec.kv_cache_specs.values())
        return False

    if not any(
        contains_mamba(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups
    ):
        return

    message = (
        "Unsafe Ascend hybrid prefix-cache host: this explicit Ascend copy "
        "profile requires the separate core unified-HMA/nonuniform-local-hit "
        "repair API SupportsNonuniformHybridLocalHits with callable "
        "can_use_nonuniform_hybrid_local_hits(request), plus a matching Ascend "
        "runtime. MTP additionally needs separately qualified ABI and semantics. "
        "The legacy connector-present scheduler can "
        "advance cached tokens without restoring Mamba state, even with no "
        "transfers. SupportsHMA alone is insufficient. No bypass is provided. "
        "Recognizing this repair API is not certification of arbitrary hosts."
    )
    try:
        base = import_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
    except ImportError as exc:
        raise RuntimeError(message) from exc
    marker = getattr(base, "SupportsNonuniformHybridLocalHits", None)
    if not isinstance(marker, type) or not callable(
        getattr(marker, "can_use_nonuniform_hybrid_local_hits", None)
    ):
        raise RuntimeError(message)
