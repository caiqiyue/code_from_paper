"""Targeted tests for round23_controller_inference.py."""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from create_fake_round23_controller_bundle import create_bundle  # noqa: E402
from round23_controller_inference import ControllerBundle, run_inference  # noqa: E402


class _PicklePredictor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, rows):
        return [self.value for _ in rows]


def test_controller_bundle_loads_mock_bundle():
    bundle_dir = create_bundle()
    bundle = ControllerBundle(bundle_dir)
    assert bundle.learner_family == "mock_linear"
    assert set(bundle.models.keys()) == {-2, -1, 0, 1, 2}


def test_run_inference_prefers_delta_zero_in_mock_bundle():
    bundle_dir = create_bundle()
    result = run_inference(bundle_dir, [0.0] * 13, reference_budget=20)
    assert result["predicted_delta_k"] == 0
    assert result["predicted_target_budget"] == 20


def test_controller_bundle_loads_linear_baseline_pickle_bundle():
    with tempfile.TemporaryDirectory(prefix="round23_linear_bundle_") as tmp:
        bundle_dir = Path(tmp)
        schema = {
            "feature_version": "no_dataset",
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
            "include_dataset_onehot": False,
            "onehot_order": [],
            "total_features": 9,
        }
        metadata = {
            "learner_family": "linear_baseline",
            "action_space": [-2, -1, 0, 1, 2],
        }
        (bundle_dir / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (bundle_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        values = {-2: -0.2, -1: -0.1, 0: 0.5, 1: 0.1, 2: 0.0}
        stems = {
            -2: "model_dk_neg2.pkl",
            -1: "model_dk_neg1.pkl",
            0: "model_dk_0.pkl",
            1: "model_dk_pos1.pkl",
            2: "model_dk_pos2.pkl",
        }
        for delta_k, filename in stems.items():
            with (bundle_dir / filename).open("wb") as handle:
                pickle.dump(_PicklePredictor(values[delta_k]), handle)
        result = run_inference(bundle_dir, [0.0] * 9, reference_budget=20)
        assert result["predicted_delta_k"] == 0
        assert result["predicted_target_budget"] == 20
        assert result["predicted_action_rewards"][0] == 0.5


if __name__ == "__main__":
    tests = [
        ("bundle_load", test_controller_bundle_loads_mock_bundle),
        ("inference", test_run_inference_prefers_delta_zero_in_mock_bundle),
        ("linear_baseline_bundle", test_controller_bundle_loads_linear_baseline_pickle_bundle),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as exc:
            failures += 1
            print(f"  {name}: FAILED - {exc}")
    if failures:
        raise SystemExit(1)
    print("\nALL ROUND23 CONTROLLER INFERENCE TESTS PASSED")
