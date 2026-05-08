#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_mechanism(
    calibration: dict,
    expected_regime: str | None,
    expected_stage: str | None,
    expected_triggered: str | None,
) -> str:
    mismatches: list[str] = []
    if expected_regime is not None and calibration.get("regime") != expected_regime:
        mismatches.append(f"regime={calibration.get('regime')}")
    if (
        expected_stage is not None
        and calibration.get("selection_stage") != expected_stage
    ):
        mismatches.append(f"selection_stage={calibration.get('selection_stage')}")
    if expected_triggered is not None:
        expected_bool = expected_triggered.lower() == "true"
        if bool(calibration.get("arbitration_triggered")) != expected_bool:
            mismatches.append(
                f"arbitration_triggered={calibration.get('arbitration_triggered')}"
            )
    if mismatches:
        return "MISMATCH:" + ",".join(mismatches)
    return "OK"


def main() -> int:
    if len(sys.argv) not in {5, 8}:
        raise SystemExit(
            "usage: append_round20_uncertain_summary.py SUMMARY_PATH EXPERIMENT STATUS "
            "OUTPUT_DIR [EXPECTED_REGIME EXPECTED_STAGE EXPECTED_TRIGGERED]"
        )

    summary_path = Path(sys.argv[1])
    experiment = sys.argv[2]
    status = sys.argv[3]
    output_dir = Path(sys.argv[4])
    expected_regime = sys.argv[5] if len(sys.argv) == 8 else None
    expected_stage = sys.argv[6] if len(sys.argv) == 8 else None
    expected_triggered = sys.argv[7] if len(sys.argv) == 8 else None

    calibration = _read_json(output_dir / "stage1_budget_calibration.json")
    evaluation = _read_json(output_dir / "eval" / "downstream_eval_summary.json")
    metrics = evaluation.get("metrics") or {}
    validation_status = _validate_mechanism(
        calibration,
        expected_regime,
        expected_stage,
        expected_triggered,
    )

    row = [
        experiment,
        status,
        calibration.get("regime", "NA"),
        calibration.get("resolved_seed_top_k", "NA"),
        calibration.get("selection_stage", "NA"),
        calibration.get("arbitration_triggered", "NA"),
        calibration.get("arbitration_winner_policy", "NA"),
        calibration.get("arbitration_reason", "NA"),
        metrics.get("best_top1", "NA"),
        validation_status,
    ]

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")
    return 0 if validation_status == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
