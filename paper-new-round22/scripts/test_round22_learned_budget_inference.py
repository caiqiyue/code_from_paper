"""Tests for round22_learned_budget_inference.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb

from round22_learned_budget_inference import ModelBundle, run_inference, BUDGETS


def make_fake_bundle(bundle_dir: Path) -> None:
    """Create a fake model bundle in the given directory (must not already exist)."""
    import numpy as np
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Create fake models (rewards proportional to k)
    for k in BUDGETS:
        X = np.array([[0.5] * 12 for _ in range(3)], dtype=np.float64)
        y = np.array([float(k) * 0.001 for _ in range(3)], dtype=np.float64)
        ds = lgb.Dataset(X, label=y)
        model = lgb.train({"objective": "regression", "verbose": -1}, ds, num_boost_round=1)
        model.save_model(str(bundle_dir / f"model_k{k}.txt"))

    schema = {
        "version": "1.0",
        "feature_names": [
            "shape_score", "private_mean_length", "private_p75_length", "private_length_iqr",
            "support_mean_at_k20", "coverage_mean_at_k20", "coverage_p25_at_k20", "genericity_mean_at_k20",
        ],
        "include_dataset_onehot": True,
        "onehot_order": ["jobs", "congressional", "forums", "microblog"],
        "total_features": 12,
    }
    (bundle_dir / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    metadata = {
        "version": "1.0",
        "training_data_version": "test",
        "reward_lambda": 0.002,
        "lightgbm_params": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200},
        "training_seeds": [42],
        "model_train_git_commit": "test",
        "model_train_git_branch": "test",
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_load_and_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        make_fake_bundle(bundle_dir)
        bundle = ModelBundle(bundle_dir)
        assert bundle.schema["total_features"] == 12
        assert len(bundle.models) == 5
        fv = [0.5] * 12
        best_k, rewards = bundle.predict_budget(fv)
        assert best_k in BUDGETS, f"best_k={best_k} not in {BUDGETS}"
        assert len(rewards) == 5


def test_wrong_feature_length_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        make_fake_bundle(bundle_dir)
        bundle = ModelBundle(bundle_dir)
        wrong_len_fv = [0.5] * 8
        try:
            bundle.predict(wrong_len_fv)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Feature vector length" in str(e)


def test_missing_schema_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_bundle = Path(tmpdir) / "empty_bundle"
        empty_bundle.mkdir()
        try:
            ModelBundle(empty_bundle)
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "feature_schema.json" in str(e)


def test_metadata_loaded():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        make_fake_bundle(bundle_dir)
        bundle = ModelBundle(bundle_dir)
        assert bundle.metadata["version"] == "1.0"
        assert "lightgbm_params" in bundle.metadata


def test_run_inference_returns_all_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        make_fake_bundle(bundle_dir)
        fv = [0.5] * 12
        result = run_inference(bundle_dir, fv)
        assert "predicted_budget" in result
        assert "predicted_rewards_by_budget" in result
        assert "feature_schema_version" in result
        assert "model_metadata" in result
        assert "total_features" in result
        assert result["total_features"] == 12
        assert result["predicted_budget"] in BUDGETS


def test_budgets_constant():
    assert BUDGETS == [18, 19, 20, 21, 22]


if __name__ == "__main__":
    tests = [
        ("ModelBundle.load_and_predict", test_load_and_predict),
        ("ModelBundle.wrong_length", test_wrong_feature_length_raises),
        ("ModelBundle.missing_schema", test_missing_schema_raises),
        ("ModelBundle.metadata", test_metadata_loaded),
        ("RunInference.all_fields", test_run_inference_returns_all_fields),
        ("Budgets.constant", test_budgets_constant),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    print("\nALL PHASE C TESTS PASSED")
