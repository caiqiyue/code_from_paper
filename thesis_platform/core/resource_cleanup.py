from __future__ import annotations

import gc
from typing import Any


def release_component_resources(*objects: Any) -> None:
    """Release nested model resources and flush CUDA cache when available."""

    seen: set[int] = set()
    for obj in objects:
        _release_one(obj, seen)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _release_one(obj: Any, seen: set[int]) -> None:
    if obj is None:
        return
    object_id = id(obj)
    if object_id in seen:
        return
    seen.add(object_id)

    release = getattr(obj, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass

    for attr_name in ("text_backend", "_text_backend", "embedder", "feature_encoder"):
        nested = getattr(obj, attr_name, None)
        if nested is not None:
            _release_one(nested, seen)
