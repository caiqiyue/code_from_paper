#!/usr/bin/env python3
"""One-shot absolute-k controller inference for round23 runtime."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from round23_context_features import validate_feature_schema


BUDGETS = [18, 19, 20, 21, 22]
ACTION_TO_STEM = {budget: f"model_k{budget}" for budget in BUDGETS}


class _MockLinearModel:
    def __init__(self, payload: dict[str, Any]):
        self.bias = float(payload.get("bias", 0.0))
        self.weights = [float(weight) for weight in payload.get("weights", [])]

    def predict_one(self, feature_vector: list[float]) -> float:
        if self.weights and len(self.weights) != len(feature_vector):
            raise ValueError(
                f"Mock linear weight length {len(self.weights)} != feature length {len(feature_vector)}"
            )
        if not self.weights:
            return self.bias
        return self.bias + sum(weight * value for weight, value in zip(self.weights, feature_vector))


class AbsoluteKBundle:
    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        self.schema = self._load_schema()
        self.metadata = self._load_metadata()
        self.learner_family = str(self.metadata.get("learner_family", self.metadata.get("model_family", "lightgbm"))).lower()
        self.models = self._load_models()

    def _load_schema(self) -> dict[str, Any]:
        path = self.bundle_dir / "feature_schema.json"
        if not path.exists():
            raise FileNotFoundError(f"feature_schema.json not found at {path}")
        return validate_feature_schema(path)

    def _load_metadata(self) -> dict[str, Any]:
        path = self.bundle_dir / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"metadata.json not found at {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        budget_space = metadata.get("budget_space", BUDGETS)
        if list(budget_space) != BUDGETS:
            raise ValueError(f"Unexpected budget_space in metadata: {budget_space}")
        return metadata

    def _find_model_file(self, stem: str) -> Path:
        matches = sorted(self.bundle_dir.glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Model file not found for stem {stem} in {self.bundle_dir}")
        return matches[0]

    def _load_models(self) -> dict[int, Any]:
        if self.learner_family == "lightgbm":
            import lightgbm as lgb

            return {budget: lgb.Booster(model_file=str(self._find_model_file(stem))) for budget, stem in ACTION_TO_STEM.items()}
        if self.learner_family == "xgboost":
            import xgboost as xgb

            models: dict[int, Any] = {}
            for budget, stem in ACTION_TO_STEM.items():
                booster = xgb.Booster()
                booster.load_model(str(self._find_model_file(stem)))
                models[budget] = booster
            return models
        if self.learner_family == "mock_linear":
            models: dict[int, Any] = {}
            for budget, stem in ACTION_TO_STEM.items():
                payload = json.loads(self._find_model_file(stem).read_text(encoding="utf-8"))
                models[budget] = _MockLinearModel(payload)
            return models
        if self.learner_family == "catboost":
            from catboost import CatBoostRegressor

            models: dict[int, Any] = {}
            for budget, stem in ACTION_TO_STEM.items():
                model = CatBoostRegressor()
                model.load_model(str(self._find_model_file(stem)))
                models[budget] = model
            return models
        models: dict[int, Any] = {}
        for budget, stem in ACTION_TO_STEM.items():
            with self._find_model_file(stem).open("rb") as handle:
                models[budget] = pickle.load(handle)
        return models

    def predict(self, feature_vector: list[float]) -> dict[int, float]:
        if len(feature_vector) != int(self.schema["total_features"]):
            raise ValueError(
                f"Feature vector length {len(feature_vector)} != total_features {self.schema['total_features']}"
            )
        predictions: dict[int, float] = {}
        for budget, model in self.models.items():
            if self.learner_family == "lightgbm":
                predictions[budget] = float(model.predict([feature_vector])[0])
            elif self.learner_family == "xgboost":
                import numpy as np
                import xgboost as xgb

                matrix = xgb.DMatrix(np.asarray([feature_vector], dtype=float))
                predictions[budget] = float(model.predict(matrix)[0])
            elif self.learner_family == "mock_linear":
                predictions[budget] = float(model.predict_one(feature_vector))
            else:
                predictions[budget] = float(model.predict([feature_vector])[0])
        return predictions

    def predict_action(self, feature_vector: list[float], reference_budget: int = 20) -> dict[str, Any]:
        rewards = self.predict(feature_vector)
        best_budget = max(rewards, key=lambda budget: rewards[budget])
        sorted_rewards = sorted(rewards.items(), key=lambda item: item[1], reverse=True)
        confidence_margin = float(sorted_rewards[0][1] - sorted_rewards[1][1]) if len(sorted_rewards) > 1 else 0.0
        predicted_delta = int(best_budget) - int(reference_budget)
        return {
            "predicted_absolute_k": int(best_budget),
            "predicted_delta_k": predicted_delta,
            "predicted_target_budget": int(best_budget),
            "predicted_action_rewards": rewards,
            "confidence_margin": confidence_margin,
        }


def run_inference(bundle_dir: str | Path, feature_vector: list[float], reference_budget: int = 20) -> dict[str, Any]:
    bundle = AbsoluteKBundle(bundle_dir)
    result = bundle.predict_action(feature_vector, reference_budget=reference_budget)
    result["feature_schema_version"] = bundle.schema["feature_version"]
    result["model_metadata"] = bundle.metadata
    result["total_features"] = bundle.schema["total_features"]
    return result
