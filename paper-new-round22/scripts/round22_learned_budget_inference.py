#!/usr/bin/env python3
"""Learned budget policy inference module for round22 runtime.

Loads the trained LightGBM model bundle and predicts optimal budget k ∈ {18,19,20,21,22}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb


BUDGETS = [18, 19, 20, 21, 22]


class ModelBundle:
    """Loads and manages a round22 learned budget policy bundle."""

    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        self.schema = self._load_schema()
        self.metadata = self._load_metadata()
        self.models = self._load_models()

    def _load_schema(self) -> dict[str, Any]:
        path = self.bundle_dir / "feature_schema.json"
        if not path.exists():
            raise FileNotFoundError(f"feature_schema.json not found at {path}")
        schema = json.loads(path.read_text())
        for field in ("version", "feature_names", "total_features"):
            if field not in schema:
                raise ValueError(f"feature_schema.json missing required field: {field}")
        return schema

    def _load_metadata(self) -> dict[str, Any]:
        path = self.bundle_dir / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"metadata.json not found at {path}")
        return json.loads(path.read_text())

    def _load_models(self) -> dict[int, lgb.LGBMModel]:
        models = {}
        for k in BUDGETS:
            model_path = self.bundle_dir / f"model_k{k}.txt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            models[k] = lgb.Booster(model_file=str(model_path))
        return models

    def predict(self, feature_vector: list[float]) -> dict[int, float]:
        """Predict reward for each budget k. Returns {k: predicted_reward}."""
        if len(feature_vector) != self.schema["total_features"]:
            raise ValueError(
                f"Feature vector length {len(feature_vector)} != "
                f"total_features {self.schema['total_features']}"
            )
        results = {}
        for k, model in self.models.items():
            pred = model.predict([feature_vector])[0]
            results[k] = float(pred)
        return results

    def predict_budget(self, feature_vector: list[float]) -> tuple[int, dict[int, float]]:
        """Predict optimal budget. Returns (best_k, {k: predicted_reward})."""
        rewards = self.predict(feature_vector)
        best_k = max(rewards, key=lambda k: rewards[k])
        return best_k, rewards


def run_inference(
    bundle_dir: str | Path,
    feature_vector: list[float],
) -> dict[str, Any]:
    """Main inference entry point. Returns dict with predicted_budget, predicted_rewards, metadata."""
    bundle = ModelBundle(bundle_dir)
    best_k, rewards = bundle.predict_budget(feature_vector)
    return {
        "predicted_budget": best_k,
        "predicted_rewards_by_budget": rewards,
        "feature_schema_version": bundle.schema["version"],
        "model_metadata": bundle.metadata,
        "total_features": bundle.schema["total_features"],
    }
