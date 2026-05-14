#!/usr/bin/env python3
"""Run a single round23 experiment with dynamic delta-k controller."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from round23_context_features import build_feature_vector, validate_feature_schema
from round23_controller_inference import run_inference
from round23_runtime_utils import (
    PAPER_NEW_ROUND23_ROOT,
    collect_runtime_artifacts,
    deep_merge,
    load_yaml_with_inherits,
    run_round19_selector_subprocess,
    build_round19_subprocess_env,
)


def run_reference_k20_feature_subprocess(
    *,
    config_path: str | Path,
    output_root: str | Path,
    timeout_seconds: int = 7200,
    reference_budget: int = 20,
) -> dict[str, Any]:
    result_json = Path(output_root) / "_k20_reference_features.json"
    helper_script = PAPER_NEW_ROUND23_ROOT / "scripts" / "round23_reference_stage0_features.py"
    result = subprocess.run(
        [
            sys.executable,
            str(helper_script),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--result-json",
            str(result_json),
            "--reference-budget",
            str(reference_budget),
        ],
        capture_output=True,
        text=True,
        cwd=str(PAPER_NEW_ROUND23_ROOT),
        env=build_round19_subprocess_env(),
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "reference stage0 feature subprocess failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not result_json.exists():
        raise FileNotFoundError(f"Expected reference feature JSON at {result_json}")
    return json.loads(result_json.read_text(encoding="utf-8"))


def generate_override_config(
    original_config_path: str | Path,
    predicted_target_budget: int,
    predicted_delta_k: int,
    model_dir: str | Path,
    output_root: str | Path,
    reference_budget: int = 20,
) -> tuple[Path, str]:
    original_cfg = load_yaml_with_inherits(original_config_path)
    experiment_id = str(original_cfg.get("meta", {}).get("experiment_id", Path(original_config_path).stem))
    override = {
        "meta": {
            "dynamic_budget_runtime": {
                "enabled": True,
                "reference_budget": int(reference_budget),
                "predicted_delta_k": int(predicted_delta_k),
                "predicted_target_budget": int(predicted_target_budget),
                "model_dir": str(Path(model_dir).resolve()),
            }
        },
        "paths": {
            "output_root": str(Path(output_root).resolve()),
        },
        "selector": {
            "seed_top_k": int(predicted_target_budget),
            "seed_budget_rule": {
                "enabled": False,
                "mode": "hierarchical_shape_routing",
            },
        },
    }
    merged = deep_merge(original_cfg, override)
    override_path = Path(output_root) / f"{experiment_id}_dynamic_controller_override.yaml"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.dump(merged, handle)
    return override_path, experiment_id


def write_runtime_sidecar(
    output_root: str | Path,
    experiment_id: str,
    inference_result: dict[str, Any],
    feature_vector: list[float],
    reference_info: dict[str, Any],
    override_config_path: Path,
    runtime_artifacts: dict[str, Any],
    model_dir: str | Path,
) -> Path:
    sidecar = {
        "budget_policy_type": "dynamic_controller",
        "reference_budget": int(reference_info["reference_budget"]),
        "predicted_delta_k": int(inference_result["predicted_delta_k"]),
        "predicted_target_budget": int(inference_result["predicted_target_budget"]),
        "predicted_action_rewards": inference_result["predicted_action_rewards"],
        "controller_confidence_margin": float(inference_result.get("confidence_margin", 0.0)),
        "feature_vector": feature_vector,
        "feature_schema_version": inference_result["feature_schema_version"],
        "controller_model_dir": str(Path(model_dir).resolve()),
        "model_metadata": inference_result.get("model_metadata", {}),
        "reference_round_output_root": reference_info["reference_output_root"],
        "reference_stage1_summary_path": reference_info["reference_stage1_summary_path"],
        "reference_feature_json_path": str(Path(output_root) / "_k20_reference_features.json"),
        "reference_metrics_snapshot": reference_info.get("reference_metrics_snapshot", {}),
        "round1_output_root": runtime_artifacts["runtime_output_root"],
        "round1_stage1_summary_path": runtime_artifacts["stage1_summary_path"],
        "round1_eval_summary_path": runtime_artifacts["eval_summary_path"],
        "override_config_path": str(override_config_path),
        "runtime_artifacts": runtime_artifacts,
    }
    sidecar_path = Path(output_root) / f"{experiment_id}_dynamic_controller_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run round23 with dynamic delta-k controller")
    parser.add_argument("--config", required=True, help="Path to original round23 experiment config YAML")
    parser.add_argument("--model-dir", required=True, help="Path to controller bundle directory")
    parser.add_argument("--output-root", required=True, help="Output root directory for this run")
    parser.add_argument("--reference-budget", type=int, default=20, help="Reference round budget k0")
    parser.add_argument("--timeout-seconds", type=int, default=7200, help="Timeout for helper and final subprocesses")
    args = parser.parse_args()

    print("[round23] Running reference Stage1-only round...")
    reference_info = run_reference_k20_feature_subprocess(
        config_path=args.config,
        output_root=args.output_root,
        timeout_seconds=args.timeout_seconds,
        reference_budget=args.reference_budget,
    )
    schema = validate_feature_schema(Path(args.model_dir) / "feature_schema.json")
    feature_vector = build_feature_vector(
        context_features=reference_info["context_features"],
        dataset_name=reference_info["dataset_name"],
        schema=schema,
    )

    print("[round23] Running controller inference...")
    inference_result = run_inference(args.model_dir, feature_vector, reference_budget=args.reference_budget)
    predicted_delta_k = int(inference_result["predicted_delta_k"])
    predicted_target_budget = int(inference_result["predicted_target_budget"])
    print(f"[round23] Predicted Δk={predicted_delta_k}, target budget k1={predicted_target_budget}")

    print("[round23] Generating override config...")
    override_path, experiment_id = generate_override_config(
        original_config_path=args.config,
        predicted_target_budget=predicted_target_budget,
        predicted_delta_k=predicted_delta_k,
        model_dir=args.model_dir,
        output_root=args.output_root,
        reference_budget=args.reference_budget,
    )

    print(f"[round23] Calling round19 runtime with {override_path} ...")
    result = run_round19_selector_subprocess(config_path=override_path, timeout_seconds=args.timeout_seconds)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(f"[ERROR] round19 runtime failed:\n{result.stderr}", file=sys.stderr)
        return int(result.returncode)

    print(result.stdout, end="")
    runtime_artifacts = collect_runtime_artifacts(args.output_root)
    sidecar_path = write_runtime_sidecar(
        args.output_root,
        experiment_id,
        inference_result,
        feature_vector,
        reference_info,
        override_path,
        runtime_artifacts,
        args.model_dir,
    )
    print(f"[round23] Done. Sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
