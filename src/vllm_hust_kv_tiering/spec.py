"""vLLM-HUST v1 spec entry point.

The v1 offloading factory loads this class from ``spec_module_path``.  The
base spec owns the stable connector protocol; this plugin supplies its own
secondary storage implementation through either ``type=hust_fs`` (after the
general plugin is loaded) or the portable ``module_path`` configuration.
"""

from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


class HustTieringOffloadingSpec(TieringOffloadingSpec):
    """Tiering spec pinned to the vLLM-HUST v1 offloading ABI."""

    pass
