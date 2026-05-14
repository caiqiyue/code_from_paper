#!/usr/bin/env python3
"""Run reference k0=20 Stage1 in an isolated subprocess and emit controller features."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from round23_context_features import (
    compute_selected_redundancy_mean,
    extract_context_features,
)
from round23_runtime_utils import PAPER_NEW_ROUND19_ROOT, deep_merge, load_yaml_with_inherits

if str(PAPER_NEW_ROUND19_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_NEW_ROUND19_ROOT))

from paper_new_selector.runtime_cleanup import release_runtime_memory
from paper_new_selector.stage1_runner import run_stage1_with_runtime
from paper_new_selector.thesis_bridge import build_embedder_from_config, load_text_samples


def compute_reference_features(
    *,
    original_config_path: str | Path,
    output_root: str | Path,
    reference_budget: int = 20,
) -> dict[str, Any]:
    orig_cfg = load_yaml_with_inherits(original_config_path)
    orig_cfg["selector"]["seed_top_k"] = int(reference_budget)
    if "seed_budget_rule" in orig_cfg.get("selector", {}):
        orig_cfg["selector"]["seed_budget_rule"]["enabled"] = False

    router_cfg = dict(orig_cfg.get("selector", {}).get("seed_budget_rule", {}).get("router", {}))
    ref_output_root = Path(output_root) / "_k20_reference"
    orig_cfg["paths"]["output_root"] = str(ref_output_root)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.dump(orig_cfg, tmp)
        ref_config_path = tmp.name

    generator_handle = None
    embedder = None
    try:
        stage1_summary, runtime = run_stage1_with_runtime(ref_config_path, validate_only=False)
        ref_output_root.mkdir(parents=True, exist_ok=True)
        stage1_summary_path = ref_output_root / "stage1_summary.json"
        stage1_summary_path.write_text(
            json.dumps(stage1_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
            record for record in candidate_records if int(record.get("index", -1)) in selected_indices
        ]
        selected_vectors = [list(map(float, record["vector"])) for record in selected_records]

        support_mean = (
            float(sum(float(record.get("private_support", 0.0)) for record in selected_records) / len(selected_records))
            if selected_records
            else 0.0
        )
        genericity_mean = (
            float(
                sum(float(record.get("genericity_penalty", 0.0)) for record in selected_records) / len(selected_records)
            )
            if selected_records
            else 0.0
        )
        redundancy_values = [
            float(record.get("redundancy_penalty", 0.0))
            for record in selected_records
            if "redundancy_penalty" in record
        ]
        redundancy_mean = (
            float(sum(redundancy_values) / len(redundancy_values))
            if redundancy_values
            else compute_selected_redundancy_mean(selected_vectors)
        )

        context_features = extract_context_features(
            private_lengths=private_lengths,
            private_vectors=private_vectors,
            selected_vectors_k20=selected_vectors,
            support_mean_at_k20=support_mean,
            genericity_mean_at_k20=genericity_mean,
            redundancy_mean_at_k20=redundancy_mean,
            router_cfg=router_cfg,
        )
        metrics_snapshot = {
            "support_mean_at_k20": support_mean,
            "genericity_mean_at_k20": genericity_mean,
            "redundancy_mean_at_k20": redundancy_mean,
        }
        return {
            "context_features": context_features,
            "dataset_name": str(
                orig_cfg.get("data", {}).get(
                    "dataset_name",
                    orig_cfg.get("meta", {}).get("dataset_name", ""),
                )
            ),
            "reference_budget": int(reference_budget),
            "reference_output_root": str(ref_output_root),
            "reference_stage1_summary_path": str(stage1_summary_path),
            "selected_count": len(selected_records),
            "reference_metrics_snapshot": metrics_snapshot,
        }
    finally:
        release_runtime_memory(getattr(generator_handle, "text_backend", None), embedder)
        Path(ref_config_path).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit round23 controller reference features as JSON")
    parser.add_argument("--config", required=True, help="Original round23 config")
    parser.add_argument("--output-root", required=True, help="Runtime output root")
    parser.add_argument("--result-json", required=True, help="Path to write computed features JSON")
    parser.add_argument("--reference-budget", type=int, default=20, help="Reference budget k0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_reference_features(
        original_config_path=args.config,
        output_root=args.output_root,
        reference_budget=args.reference_budget,
    )
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
