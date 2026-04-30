#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from paper_new_selector.thesis_bridge import load_yaml_config


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: append_round182_summary.py SUMMARY_PATH EXPERIMENT STATUS CONFIG_PATH OUTPUT_DIR"
        )

    summary_path = Path(sys.argv[1])
    experiment = sys.argv[2]
    status = sys.argv[3]
    config_path = Path(sys.argv[4])
    output_dir = Path(sys.argv[5])

    config = load_yaml_config(config_path)
    stage1 = _read_json(output_dir / "stage1_summary.json")
    evaluation = _read_json(output_dir / "eval" / "downstream_eval_summary.json")

    seed_budget = stage1.get("seed_budget") or {}
    metrics = evaluation.get("metrics") or {}
    bootstrap_cfg = dict(config.get("bootstrap", {}))
    rule_cfg = dict(config.get("selector", {}).get("seed_budget_rule", {}))
    if bool(rule_cfg.get("enabled", True)):
        raise RuntimeError("Round18.2 summary expects seed_budget_rule.enabled=false")
    if str(seed_budget.get("mode", "")) != "disabled":
        raise RuntimeError("Round18.2 summary expects seed_budget.mode=disabled")
    if seed_budget.get("configured_seed_top_k") != seed_budget.get("resolved_seed_top_k"):
        raise RuntimeError(
            "Round18.2 summary expects configured_seed_top_k == resolved_seed_top_k"
        )

    row = [
        experiment,
        status,
        seed_budget.get("configured_seed_top_k", "NA"),
        seed_budget.get("resolved_seed_top_k", "NA"),
        seed_budget.get("mode", "NA"),
        bootstrap_cfg.get("max_tokens", "NA"),
        metrics.get("best_top1", "NA"),
        metrics.get("best_top3", "NA"),
        metrics.get("best_top5", "NA"),
        metrics.get("best_top10", "NA"),
    ]

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
