#!/usr/bin/env python3
"""Create a fake controller bundle for round23 local runtime testing."""
from __future__ import annotations

import json
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "controller_bundle" / "round23_mock_smoke_v1"
ACTIONS = [-2, -1, 0, 1, 2]


def _stem(delta_k: int) -> str:
    if delta_k < 0:
        return f"model_dk_neg{abs(delta_k)}.json"
    if delta_k > 0:
        return f"model_dk_pos{delta_k}.json"
    return "model_dk_0.json"


def create_bundle() -> Path:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    schema = {
        "version": "1.0",
        "feature_names": [
            "shape_score",
            "private_mean_length",
            "private_p75_length",
            "private_length_iqr",
            "support_mean_at_k20",
            "coverage_mean_at_k20",
            "coverage_p25_at_k20",
            "genericity_mean_at_k20",
            "redundancy_mean_at_k20",
        ],
        "include_dataset_onehot": True,
        "onehot_order": ["jobs", "congressional", "forums", "microblog"],
        "total_features": 13,
    }
    metadata = {
        "controller_version": "round23_mock_smoke_v1",
        "learner_family": "mock_linear",
        "training_data_version": "mock",
        "reference_budget": 20,
        "action_space": ACTIONS,
        "reward_formula": "mock_reward",
        "feature_names": schema["feature_names"],
    }
    (BUNDLE_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for delta_k in ACTIONS:
        payload = {"bias": -abs(delta_k) * 0.01}
        if delta_k == 0:
            payload["bias"] = 0.02
        (BUNDLE_DIR / _stem(delta_k)).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BUNDLE_DIR


if __name__ == "__main__":
    print(create_bundle())
