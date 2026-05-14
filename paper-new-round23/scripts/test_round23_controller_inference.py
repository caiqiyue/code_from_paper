"""Targeted tests for round23_controller_inference.py."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from create_fake_round23_controller_bundle import create_bundle  # noqa: E402
from round23_controller_inference import ControllerBundle, run_inference  # noqa: E402


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


if __name__ == "__main__":
    tests = [
        ("bundle_load", test_controller_bundle_loads_mock_bundle),
        ("inference", test_run_inference_prefers_delta_zero_in_mock_bundle),
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
