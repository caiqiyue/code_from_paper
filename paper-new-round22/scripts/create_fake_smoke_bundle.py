#!/usr/bin/env python3
"""Create a fake smoke test bundle for learned budget policy testing.

This script is for local testing only - do not use for production.
Run from paper-new-round22/scripts/ directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb  # noqa: F401 - available in conda experiment env

BUNDLE_DIR = Path(__file__).resolve().parents[2] / "artifacts/learned_budget_policy/round22_lgbm_smoke_v1"
BUDGETS = [18, 19, 20, 21, 22]


def create_smoke_bundle() -> Path:
    import numpy as np
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for k in BUDGETS:
        # Simple model: reward = k * 0.001 + noise
        X = [[0.5] * 12 for _ in range(10)]
        y = [float(k) * 0.001 for _ in range(10)]
        X_np = np.array(X, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)
        ds = lgb.Dataset(X_np, label=y_np)
        model = lgb.train({"objective": "regression", "verbose": -1}, ds, num_boost_round=3)
        model.save_model(str(BUNDLE_DIR / f"model_k{k}.txt"))

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
        ],
        "include_dataset_onehot": True,
        "onehot_order": ["jobs", "congressional", "forums", "microblog"],
        "total_features": 12,
    }
    (BUNDLE_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    metadata = {
        "version": "1.0",
        "exported_at": "2026-05-12T00:00:00",
        "training_data_version": "smoke_test",
        "reward_lambda": 0.002,
        "lightgbm_params": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
        },
        "training_seeds": [42],
        "model_train_git_commit": "smoke",
        "model_train_git_branch": "smoke",
    }
    (BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return BUNDLE_DIR


if __name__ == "__main__":
    bundle = create_smoke_bundle()
    print(f"[smoke] Fake bundle created at: {bundle}")
    print(f"[smoke] Contents: {list(bundle.iterdir())}")
