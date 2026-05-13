#!/usr/bin/env python3
"""Run a single round22 experiment with learned budget policy.

This wrapper keeps round19 runtime intact and only replaces the budget decision:
1. Run a reference k=20 Stage1 pass
2. Extract the 8 runtime features required by the learned policy
3. Load the LightGBM bundle and predict the best budget in {18,19,20,21,22}
4. Write a temporary override config that pins selector.seed_top_k to the predicted budget
5. Call round19 runtime with the override config
6. Write a sidecar JSON for audit/debugging
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

PAPER_NEW_ROUND22_ROOT = Path(__file__).resolve().parents[1]
PAPER_NEW_ROUND19_ROOT = (PAPER_NEW_ROUND22_ROOT / "../paper-new-round19").resolve()

if str(PAPER_NEW_ROUND19_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_NEW_ROUND19_ROOT))

from paper_new_selector.runtime_cleanup import release_runtime_memory
from paper_new_selector.stage1_runner import run_stage1_with_runtime
from paper_new_selector.thesis_bridge import build_embedder_from_config, load_text_samples

from round22_context_features import (
    _percentile_nearest_rank,
    build_feature_vector,
    compute_coverage_metrics,
    compute_shape_descriptor,
    compute_shape_score,
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


def run_reference_k20_and_extract_features(
    original_config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run a k=20 reference Stage1 pass and compute learned-policy features."""
    orig_cfg = load_yaml_with_inherits(original_config_path)
    orig_cfg["selector"]["seed_top_k"] = 20
    orig_cfg["selector"]["seed_budget_rule"]["enabled"] = False

    router_cfg = dict(orig_cfg.get("selector", {}).get("seed_budget_rule", {}).get("router", {}))
    tail_threshold = int(router_cfg.get("tail_threshold", 350))
    short_threshold = int(router_cfg.get("short_threshold", 120))

    ref_output_root = Path(output_root) / "_k20_reference"
    orig_cfg["paths"]["output_root"] = str(ref_output_root)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.dump(orig_cfg, tmp)
        ref_config_path = tmp.name

    generator_handle = None
    embedder = None
    try:
        stage1_summary, runtime = run_stage1_with_runtime(ref_config_path, validate_only=False)
        generator_handle = runtime.get("generator_handle")
        embedder = runtime.get("embedder") or build_embedder_from_config(ref_config_path)

        sample_bundle = load_text_samples(ref_config_path)
        private_texts = [sample.render_text() for sample in sample_bundle["train_samples"]]
        private_lengths = [len(text.split()) for text in private_texts]
        private_vectors = [list(map(float, vector)) for vector in embedder.embed_texts(private_texts)]

        decision = dict(stage1_summary.get("decision", {}))
        candidate_records = list(decision.get("candidate_records", []))
        selected_indices = {int(index) for index in decision.get("selected_indices", [])}
        selected_records = [
            record
            for record in candidate_records
            if int(record.get("index", -1)) in selected_indices
        ]
        selected_vectors = [list(map(float, record["vector"])) for record in selected_records]

        support_mean_at_k20 = (
            float(sum(float(record["private_support"]) for record in selected_records) / len(selected_records))
            if selected_records
            else 0.0
        )
        genericity_mean_at_k20 = (
            float(sum(float(record["genericity_penalty"]) for record in selected_records) / len(selected_records))
            if selected_records
            else 0.0
        )
        coverage_mean_at_k20, coverage_p25_at_k20 = compute_coverage_metrics(
            private_vectors=private_vectors,
            selected_vectors=selected_vectors,
        )

        descriptor = compute_shape_descriptor(
            private_lengths,
            tail_threshold=tail_threshold,
            short_threshold=short_threshold,
        )
        shape_score = compute_shape_score(descriptor, router_cfg)
        mean_length = float(sum(private_lengths) / len(private_lengths)) if private_lengths else 0.0
        p75_length = float(_percentile_nearest_rank(private_lengths, 75)) if private_lengths else 0.0
        q1 = float(_percentile_nearest_rank(private_lengths, 25)) if private_lengths else 0.0
        q3 = float(_percentile_nearest_rank(private_lengths, 75)) if private_lengths else 0.0
        length_iqr = q3 - q1

        return {
            "context_features": [
                shape_score,
                mean_length,
                p75_length,
                length_iqr,
                support_mean_at_k20,
                coverage_mean_at_k20,
                coverage_p25_at_k20,
                genericity_mean_at_k20,
            ],
            "dataset_name": str(orig_cfg.get("meta", {}).get("dataset_name", "")),
            "reference_budget": 20,
            "reference_output_root": str(ref_output_root),
            "selected_count": len(selected_records),
        }
    finally:
        release_runtime_memory(
            getattr(generator_handle, "text_backend", None),
            embedder,
        )
        Path(ref_config_path).unlink(missing_ok=True)


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
    reference_info = run_reference_k20_and_extract_features(args.config, args.output_root)
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
