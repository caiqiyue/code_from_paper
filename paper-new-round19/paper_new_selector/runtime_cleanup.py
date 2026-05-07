from __future__ import annotations

import gc
from typing import Any


def release_runtime_memory(*objects: Any) -> None:
    for obj in objects:
        if obj is None:
            continue
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
