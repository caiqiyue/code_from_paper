from __future__ import annotations

import os
from typing import Any

from pretext_platform.core.run_state import PretextFailure


BYTES_PER_GIB = 1024**3


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mem_get_info(cuda: Any, device_index: int) -> tuple[int, int]:
    try:
        return cuda.mem_get_info(device_index)
    except TypeError:
        return cuda.mem_get_info()


def ensure_vllm_startup_memory(bootstrap_cfg: dict[str, Any]) -> dict[str, Any]:
    """Reject vLLM startup before allocation when the shared GPU is too full."""

    required_free_gib = _optional_float(bootstrap_cfg.get("startup_required_free_gb"))
    if required_free_gib is None:
        return {
            "required_free_gib": None,
            "observed_free_gib": None,
            "gpu_index": None,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    try:
        import torch
    except ImportError as exc:
        raise PretextFailure(
            "cuda_unavailable_for_stage2",
            "Stage 2 vLLM memory precheck requires PyTorch CUDA support.",
            phase="stage2_precheck",
            details={
                "required_free_gib": required_free_gib,
                "observed_free_gib": None,
                "gpu_index": None,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        ) from exc

    if not torch.cuda.is_available():
        raise PretextFailure(
            "cuda_unavailable_for_stage2",
            "Stage 2 vLLM memory precheck found no available CUDA device.",
            phase="stage2_precheck",
            details={
                "required_free_gib": required_free_gib,
                "observed_free_gib": None,
                "gpu_index": None,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )

    device_index = int(torch.cuda.current_device())
    free_bytes, total_bytes = _mem_get_info(torch.cuda, device_index)
    observed_free_gib = free_bytes / BYTES_PER_GIB
    observed_total_gib = total_bytes / BYTES_PER_GIB
    details = {
        "required_free_gib": required_free_gib,
        "observed_free_gib": round(observed_free_gib, 3),
        "observed_total_gib": round(observed_total_gib, 3),
        "gpu_index": device_index,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if observed_free_gib < required_free_gib:
        raise PretextFailure(
            "insufficient_free_gpu_memory_before_stage2",
            (
                f"free GPU memory {observed_free_gib:.2f} GiB is below "
                f"required {required_free_gib:.2f} GiB before Stage 2 vLLM startup"
            ),
            phase="stage2_precheck",
            details=details,
        )
    return details
