"""Targeted tests for run_round23_with_dynamic_controller.py."""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_round23_with_dynamic_controller import (  # noqa: E402
    generate_override_config,
    resolve_controller_model_dir,
    write_runtime_sidecar,
)
from round23_runtime_utils import (  # noqa: E402
    DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE,
    DEFAULT_ROUND23_CONTROLLER_SCOPE,
    build_round19_subprocess_env,
    collect_runtime_artifacts,
    load_yaml_with_inherits,
)


def test_build_round19_subprocess_env_sets_cuda_order_when_needed():
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_order = os.environ.get("CUDA_DEVICE_ORDER")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ.pop("CUDA_DEVICE_ORDER", None)
        env = build_round19_subprocess_env()
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        if old_order is None:
            os.environ.pop("CUDA_DEVICE_ORDER", None)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = old_order


def test_generate_override_config_pins_delta_k_and_output_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "meta:",
                    "  experiment_id: r23_test",
                    "selector:",
                    "  seed_top_k: 20",
                    "  seed_budget_rule:",
                    "    enabled: true",
                    "paths:",
                    "  output_root: ./outputs/original",
                ]
            ),
            encoding="utf-8",
        )
        override_path, experiment_id = generate_override_config(
            original_config_path=config_path,
            predicted_target_budget=18,
            predicted_delta_k=-2,
            model_dir=root / "bundle",
            output_root=root / "runtime",
        )
        merged = load_yaml_with_inherits(override_path)
        assert experiment_id == "r23_test"
        assert int(merged["selector"]["seed_top_k"]) == 18
        assert int(merged["meta"]["dynamic_budget_runtime"]["predicted_delta_k"]) == -2
        assert Path(merged["paths"]["output_root"]).resolve() == (root / "runtime").resolve()


def test_collect_runtime_artifacts_requires_stage1_and_eval_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "stage1_summary.json").write_text("{}", encoding="utf-8")
        eval_dir = root / "eval"
        eval_dir.mkdir()
        (eval_dir / "downstream_eval_summary.json").write_text('{"best_top1": 0.30}', encoding="utf-8")
        artifacts = collect_runtime_artifacts(root)
        assert artifacts["stage1_summary_path"].endswith("stage1_summary.json")
        assert artifacts["eval_summary"]["best_top1"] == 0.30


def test_default_controller_bundle_points_to_all6_extratrees_bundle():
    assert DEFAULT_ROUND23_CONTROLLER_SCOPE == "all6"
    assert DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE.name == (
        "round23_controller_1200_all6_top1_delta_m0005_extratrees_broad_no_dataset"
    )
    assert resolve_controller_model_dir(None) == DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE.resolve()


def test_runtime_sidecar_records_controller_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        override = root / "override.yaml"
        override.write_text("{}", encoding="utf-8")
        inference = {
            "predicted_delta_k": 1,
            "predicted_target_budget": 21,
            "predicted_action_rewards": {-2: 0.1, -1: 0.2, 0: 0.3, 1: 0.9, 2: 0.4},
            "confidence_margin": 0.5,
            "feature_schema_version": "no_dataset",
            "model_metadata": {
                "bundle_version": "bundle-v1",
                "learner_family": "extratrees",
                "feature_version": "no_dataset",
                "target_mode": "top1_delta",
                "target_field": "target_value_for_training",
                "reference_budget": 20,
            },
        }
        sidecar = write_runtime_sidecar(
            root,
            "r23_test",
            inference,
            [0.1, 0.2],
            {
                "reference_budget": 20,
                "reference_output_root": str(root / "reference"),
                "reference_stage1_summary_path": str(root / "reference" / "stage1_summary.json"),
            },
            override,
            {
                "runtime_output_root": str(root),
                "stage1_summary_path": str(root / "stage1_summary.json"),
                "eval_summary_path": str(root / "eval" / "downstream_eval_summary.json"),
            },
            root / "bundle-v1",
        )
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["controller_scope"] == "all6"
        assert payload["bundle_version"] == "bundle-v1"
        assert payload["learner_family"] == "extratrees"
        assert payload["feature_version"] == "no_dataset"
        assert payload["target_mode"] == "top1_delta"
        assert payload["target_field"] == "target_value_for_training"


if __name__ == "__main__":
    tests = [
        ("env.cuda_order", test_build_round19_subprocess_env_sets_cuda_order_when_needed),
        ("override_config", test_generate_override_config_pins_delta_k_and_output_root),
        ("collect_runtime_artifacts", test_collect_runtime_artifacts_requires_stage1_and_eval_summary),
        ("default_controller_bundle", test_default_controller_bundle_points_to_all6_extratrees_bundle),
        ("runtime_sidecar_metadata", test_runtime_sidecar_records_controller_metadata),
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
    print("\nALL ROUND23 RUNTIME WRAPPER TESTS PASSED")
