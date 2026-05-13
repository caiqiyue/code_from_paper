#!/usr/bin/env python3
"""Run a single round22 experiment with learned budget policy."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PAPER_NEW_ROUND22_ROOT = Path(__file__).resolve().parents[1]
PAPER_NEW_ROUND19_ROOT = (PAPER_NEW_ROUND22_ROOT / "../paper-new-round19").resolve()

from round22_context_features import (
    build_feature_vector,
    validate_feature_schema,
)
from round22_learned_budget_inference import run_inference


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_yaml_with_inherits(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    inherits = data.pop("inherits", []) or []
    merged: dict[str, Any] = {}
    for parent in inherits:
        parent_path = (Path(path).parent / str(parent)).resolve()
        merged = deep_merge(merged, load_yaml_with_inherits(parent_path))
    return deep_merge(merged, data)


def build_round19_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(PAPER_NEW_ROUND19_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if env.get("CUDA_VISIBLE_DEVICES") and not env.get("CUDA_DEVICE_ORDER"):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def run_round19_selector_subprocess(
    *,
    config_path: str | Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "paper_new_selector.run_selector_single_node", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(PAPER_NEW_ROUND19_ROOT),
        env=build_round19_subprocess_env(),
        timeout=timeout_seconds,
    )


def run_reference_k20_feature_subprocess(
    *,
    config_path: str | Path,
    output_root: str | Path,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    result_json = Path(output_root) / "_k20_reference_features.json"
    helper_script = PAPER_NEW_ROUND22_ROOT / "scripts" / "round22_reference_stage1_features.py"
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
        ],
        capture_output=True,
        text=True,
        cwd=str(PAPER_NEW_ROUND22_ROOT),
        env=build_round19_subprocess_env(),
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "reference k=20 Stage1 feature subprocess failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not result_json.exists():
        raise FileNotFoundError(f"Expected reference feature JSON at {result_json}")
    return json.loads(result_json.read_text(encoding="utf-8"))


def generate_override_config(
    original_config_path: str | Path,
    predicted_k: int,
    model_dir: str | Path,
    output_root: str | Path,
) -> tuple[Path, str]:
    """Generate override YAML that injects predicted_k into round19 runtime."""
    orig_cfg = load_yaml_with_inherits(original_config_path)
    experiment_id = str(orig_cfg.get("meta", {}).get("experiment_id", Path(original_config_path).stem))

    override = {
        "meta": {
            "learned_budget_runtime": {
                "enabled": True,
                "predicted_budget": int(predicted_k),
                "model_dir": str(Path(model_dir).resolve()),
            }
        },
        "selector": {
            "seed_top_k": int(predicted_k),
            "seed_budget_rule": {
                "enabled": False,
                "mode": "hierarchical_shape_routing",
            },
        },
    }

    merged = deep_merge(orig_cfg, override)
    override_path = Path(output_root) / f"{experiment_id}_learned_override.yaml"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.dump(merged, handle)
    return override_path, experiment_id


def write_learned_runtime_sidecar(
    output_root: str | Path,
    experiment_id: str,
    inference_result: dict[str, Any],
    feature_vector: list[float],
    reference_info: dict[str, Any],
    override_config_path: Path,
) -> Path:
    sidecar = {
        "budget_policy_type": "learned",
        "learned_budget_enabled": True,
        "predicted_budget": int(inference_result["predicted_budget"]),
        "predicted_rewards_by_budget": inference_result["predicted_rewards_by_budget"],
        "feature_vector": feature_vector,
        "feature_schema_version": inference_result["feature_schema_version"],
        "model_dir": str(reference_info["model_dir"]),
        "model_metadata": inference_result.get("model_metadata", {}),
        "reference_budget": int(reference_info["reference_budget"]),
        "reference_stage1_output": {
            "dataset_name": reference_info["dataset_name"],
            "output_root": reference_info["reference_output_root"],
            "selected_count": int(reference_info.get("selected_count", 0)),
        },
        "override_config_path": str(override_config_path),
    }

    path = Path(output_root) / f"{experiment_id}_learned_budget_policy_runtime.json"
    path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run round22 with learned budget policy")
    parser.add_argument("--config", type=str, required=True, help="Path to original round22 experiment config YAML")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to learned policy model bundle directory")
    parser.add_argument("--output-root", type=str, required=True, help="Output root directory for this run")
    args = parser.parse_args()

    print("[learned] Running k=20 reference Stage1 for feature extraction...")
    reference_info = run_reference_k20_feature_subprocess(
        config_path=args.config,
        output_root=args.output_root,
    )
    reference_info["model_dir"] = str(Path(args.model_dir).resolve())

    schema = validate_feature_schema(Path(args.model_dir) / "feature_schema.json")
    feature_vector = build_feature_vector(
        context_features=reference_info["context_features"],
        dataset_name=reference_info["dataset_name"],
        schema=schema,
    )

    print("[learned] Running learned policy inference...")
    inference_result = run_inference(args.model_dir, feature_vector)
    predicted_k = int(inference_result["predicted_budget"])
    print(f"[learned] Predicted budget: k={predicted_k}")

    print("[learned] Generating override config...")
    override_path, experiment_id = generate_override_config(
        args.config,
        predicted_k,
        args.model_dir,
        args.output_root,
    )

    print("[learned] Writing sidecar summary...")
    sidecar_path = write_learned_runtime_sidecar(
        args.output_root,
        experiment_id,
        inference_result,
        feature_vector,
        reference_info,
        override_path,
    )

    print(f"[learned] Calling round19 runtime with {override_path} ...")
    result = run_round19_selector_subprocess(config_path=override_path, timeout_seconds=7200)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(f"[ERROR] round19 runtime failed:\n{result.stderr}", file=sys.stderr)
        return int(result.returncode)

    print(result.stdout, end="")
    print(f"[learned] Done. Sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
