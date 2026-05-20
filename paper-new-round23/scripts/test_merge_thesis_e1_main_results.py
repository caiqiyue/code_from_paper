"""Targeted tests for thesis E1 main result merging."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import merge_thesis_e1_main_results as merge_e1  # noqa: E402


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_merge_synthetic_round19_and_round23_summaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        round19_summary = root / "paper-new-round19" / "logs" / "thesis_e1_main_seen_pilot_summary.tsv"
        round23_summary = root / "paper-new-round23" / "logs" / "round23_thesis_main_seen_pilot_summary.tsv"
        _write_tsv(
            round19_summary,
            [
                {
                    "experiment_id": "pretext_jobs_seed42",
                    "method": "pretext",
                    "method_display_name": "PrE-Text",
                    "dataset": "jobs",
                    "seed": 42,
                    "status": "success",
                    "best_top1": 0.10,
                    "best_top3": 0.20,
                    "best_top5": 0.30,
                    "best_top10": 0.40,
                    "mapping_status": "needs_verification",
                    "implementation_key": "expand_private",
                },
                {
                    "experiment_id": "wasp_jobs_seed42",
                    "method": "wasp",
                    "method_display_name": "WASP",
                    "dataset": "jobs",
                    "seed": 42,
                    "status": "failed",
                    "best_top1": "",
                    "best_top3": "",
                    "best_top5": "",
                    "best_top10": "",
                    "mapping_status": "",
                    "implementation_key": "wasp",
                },
            ],
        )
        _write_tsv(
            round23_summary,
            [
                {
                    "experiment_id": "r23_jobs_seed42",
                    "dataset_name": "jobs",
                    "meta_seed": 42,
                    "status": "success",
                    "best_top1": 0.50,
                    "best_top3": 0.60,
                    "best_top5": 0.70,
                    "best_top10": 0.80,
                    "controller_scope": "all6",
                    "bundle_version": "bundle-v1",
                }
            ],
        )

        outputs = merge_e1.merge_mode_results(
            mode="thesis_main_seen_pilot",
            round19_summary=round19_summary,
            round23_summary=round23_summary,
            output_dir=root / "paper-new-round23" / "logs",
        )

        method_rows = list(csv.DictReader(outputs["method_dataset_summary"].open("r", encoding="utf-8"), delimiter="\t"))
        table_rows = list(csv.DictReader(outputs["paper_table"].open("r", encoding="utf-8"), delimiter="\t"))
        round23_row = next(row for row in method_rows if row["Method"] == "round23")
        self_note = round23_row["Note"]
        assert "6-dataset controller" in self_note
        assert "bundle-v1" in self_note
        assert round23_row["n_success"] == "1"
        assert table_rows[0]["Method"] == "PrE-Text"
        assert table_rows[-1]["Method"] == "round23"


if __name__ == "__main__":
    tests = [("merge_synthetic", test_merge_synthetic_round19_and_round23_summaries)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as exc:
            failures += 1
            print(f"  {name}: FAILED - {exc}")
    if failures:
        raise SystemExit(1)
    print("\nALL THESIS E1 MERGE TESTS PASSED")
