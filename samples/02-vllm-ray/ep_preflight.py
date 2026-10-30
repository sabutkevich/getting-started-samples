"""Check the pinned vLLM Humming EP imports/device policy before loading weights.

This is not a replacement for inference: model-specific kernel execution is
verified only by the real generated response before READY.json is written.
"""
import os
from types import SimpleNamespace


def check_ep_backend():
    os.environ["VLLM_HUMMING_MOE_GEMM_TYPE"] = "grouped"
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingGroupedExperts, get_humming_moe_gemm_type,
    )
    from vllm.utils.humming import GemmType

    parallel = SimpleNamespace(use_ep=True)
    selected = get_humming_moe_gemm_type(SimpleNamespace(moe_parallel_config=parallel))
    if selected != "grouped_contiguous":
        raise RuntimeError(f"Expected grouped Humming MoE for EP, got {selected}")
    if not HummingGroupedExperts._supports_current_device():
        raise RuntimeError("Humming grouped MoE is unavailable on this CUDA device")
    if not HummingGroupedExperts._supports_parallel_config(parallel):
        raise RuntimeError("Humming grouped MoE rejected expert parallelism")
    if HummingGroupedExperts.humming_gemm_type() != GemmType.GROUPED_CONTIGUOUS:
        raise RuntimeError("Unexpected Humming grouped kernel type")
    return {"moe_backend": "humming", "gemm_type": selected,
            "all2all_backend": "allgather_reducescatter",
            "imports_and_device_policy": "PASSED", "model_kernel_execution": "Deferred to actual inference"}
