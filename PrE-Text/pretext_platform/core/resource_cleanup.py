from __future__ import annotations

import gc
from typing import Any


def release_gpu_memory(*objects: Any) -> None:
    """Release referenced model objects and flush CUDA cache when available."""

    for obj in objects:
        release = getattr(obj, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
