#!/usr/bin/env python3
"""Run reference k=20 Stage1 in an isolated subprocess and emit learned-policy features."""
from __future__ import annotations

import argparse
import json
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
    compute_coverage_metrics,
    compute_shape_descriptor,
    compute_shape_score,
)


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


def compute_reference_features(
    *,
    original_config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit learned-policy reference k=20 features as JSON")
    parser.add_argument("--config", required=True, help="Original round22 config")
    parser.add_argument("--output-root", required=True, help="Runtime output root")
    parser.add_argument("--result-json", required=True, help="Path to write computed features JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_reference_features(
        original_config_path=args.config,
        output_root=args.output_root,
    )
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
