#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
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
            "usage: append_round183_summary.py SUMMARY_PATH EXPERIMENT STATUS OUTPUT_DIR"
        )

    summary_path = Path(sys.argv[1])
    experiment = sys.argv[2]
    status = sys.argv[3]
    output_dir = Path(sys.argv[4])

    calibration = _read_json(output_dir / "stage1_budget_calibration.json")
    evaluation = _read_json(output_dir / "eval" / "downstream_eval_summary.json")

    if str(calibration.get("mode", "")) != "hybrid_length_family_constrained":
        raise RuntimeError(
            "Round18.3 summary expects stage1_budget_calibration.json from hybrid mode"
        )

    coverage_constraint = calibration.get("coverage_constraint") or {}
    metrics = evaluation.get("metrics") or {}

    row = [
        experiment,
        status,
        calibration.get("selection_source", "NA"),
        calibration.get("configured_seed_top_k", "NA"),
        calibration.get("resolved_seed_top_k", "NA"),
        calibration.get("length_family_resolved_seed_top_k", "NA"),
        calibration.get("runner_up_seed_top_k", "NA"),
        calibration.get("selection_stage", "NA"),
        calibration.get("fallback_used", "NA"),
        _stringify(coverage_constraint.get("feasible_budgets")),
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
