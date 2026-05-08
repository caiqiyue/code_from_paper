from pathlib import Path
import json

from paper_new_selector.repeat15_runner import REPEAT15_SUMMARY_HEADER


def test_append_round19_summary_reads_best_topk_fields(tmp_path: Path):
    output_dir = tmp_path / "exp"
    (output_dir / "eval").mkdir(parents=True)
    (output_dir / "stage1_budget_calibration.json").write_text(
        json.dumps({"mode": "disabled", "configured_seed_top_k": 6, "resolved_seed_top_k": 6}),
        encoding="utf-8",
    )
    (output_dir / "eval" / "downstream_eval_summary.json").write_text(
        json.dumps({"metrics": {"best_top1": 0.1, "best_top3": 0.2, "best_top5": 0.3, "best_top10": 0.4}}),
        encoding="utf-8",
    )

    assert "best_top1" in REPEAT15_SUMMARY_HEADER
    assert "best_top10" in REPEAT15_SUMMARY_HEADER

