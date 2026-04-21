from __future__ import annotations

from typing import Any


def model_device(model: Any) -> Any:
    """Return the device used by a model, even if it has no direct `.device` helper."""

    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "module") and hasattr(model.module, "device"):
        return model.module.device
    return next(model.parameters()).device


def move_batch_to_model_device(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    """Move tensor-like batch values onto the same device as the target model."""

    device = model_device(model)
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in batch.items()}
