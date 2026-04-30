#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _stringify(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: append_round181_summary.py SUMMARY_PATH EXPERIMENT STATUS OUTPUT_DIR"
        )

    summary_path = Path(sys.argv[1])
    experiment = sys.argv[2]
    status = sys.argv[3]
    output_dir = Path(sys.argv[4])

    calibration = _read_json(output_dir / "stage1_budget_calibration.json")
    evaluation = _read_json(output_dir / "eval" / "downstream_eval_summary.json")

    coverage_constraint = calibration.get("coverage_constraint") or {}
    constrained_recheck = calibration.get("constrained_recheck") or {}
    metrics = evaluation.get("metrics") or {}

    row = [
        experiment,
        status,
        calibration.get("resolved_seed_top_k", "NA"),
        calibration.get("runner_up_seed_top_k", "NA"),
        calibration.get("utility_gap", "NA"),
        _stringify(coverage_constraint.get("feasible_budgets")),
        constrained_recheck.get("candidate_budget", "NA"),
        constrained_recheck.get("promoted_budget", "NA"),
        constrained_recheck.get("pass_recheck", "NA"),
        constrained_recheck.get("support_drop", "NA"),
        constrained_recheck.get("support_drop_normalized", "NA"),
        constrained_recheck.get("coverage_mean_gain", "NA"),
        constrained_recheck.get("coverage_p25_gain", "NA"),
        constrained_recheck.get("coverage_min_gain", "NA"),
        constrained_recheck.get("family_score_gain", "NA"),
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
