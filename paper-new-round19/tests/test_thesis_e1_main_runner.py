from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_new_selector import thesis_e1_main_runner as e1


class ThesisE1MainRunnerTests(unittest.TestCase):
    def test_builds_expected_e1_specs_and_method_labels(self) -> None:
        specs = e1.build_e1_run_specs("thesis_main_seen_pilot")
        self.assertEqual(len(specs), 48)
        self.assertEqual(
            {spec.method_display_name for spec in specs},
            {"PrE-Text", "round19", "WASP", "DPGA-TextSyn"},
        )
        pretext = next(spec for spec in specs if spec.method == "pretext")
        self.assertEqual(pretext.mapping_status, "needs_verification")
        self.assertEqual(pretext.implementation_key, "expand_private")

    def test_builds_smoke_and_repeat15_experiment_counts(self) -> None:
        smoke_specs = e1.build_e1_run_specs("thesis_main_seen_smoke")
        repeat15_specs = e1.build_e1_run_specs("thesis_main_seen_repeat15")
        self.assertEqual(len(smoke_specs), 16)
        self.assertEqual(len(repeat15_specs), 240)
        self.assertEqual({spec.seed for spec in smoke_specs}, {42})

    def test_materializes_manifest_with_registry_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated = e1.materialize_e1_configs(root, mode="thesis_main_seen_pilot")
            self.assertEqual(len(generated), 48)
            manifest = root / "configs" / "experiments" / "thesis_e1_main_seen_pilot" / "thesis_e1_main_seen_pilot_manifest.tsv"
            with manifest.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 48)
            self.assertEqual(rows[0]["method_display_name"], "PrE-Text")
            self.assertEqual(rows[0]["mapping_status"], "needs_verification")
            self.assertEqual(rows[0]["implementation_key"], "expand_private")
            self.assertTrue(rows[0]["output_root"].startswith("paper-new-round19/outputs/"))
            self.assertTrue((root / rows[0]["config_path"]).exists())

            generated_repeat15 = e1.materialize_e1_configs(root, mode="thesis_main_seen_repeat15")
            self.assertEqual(len(generated_repeat15), 240)
            repeat15_manifest = root / "configs" / "experiments" / "thesis_e1_main_seen_repeat15" / "thesis_e1_main_seen_repeat15_manifest.tsv"
            with repeat15_manifest.open("r", encoding="utf-8") as handle:
                repeat15_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(repeat15_rows), 240)

    def test_dry_run_summary_shape_supports_e1_table_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            e1.materialize_e1_configs(root, mode="thesis_main_seen_pilot")
            summary = e1.run_e1_batch(root, mode="thesis_main_seen_pilot", dry_run=True, limit=2)
            self.assertEqual(summary["pending_count"], 2)
            self.assertEqual(summary["mode"], "thesis_main_seen_pilot")
            self.assertEqual(summary["summary_tsv"], str((root / "logs" / "thesis_e1_main_seen_pilot_summary.tsv").resolve()))

    def test_append_summary_reads_nested_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = e1.build_e1_run_specs("thesis_main_seen_pilot")[0]
            output_dir = e1.resolve_project_output_path(root, spec.relative_output_root)
            (output_dir / "eval").mkdir(parents=True)
            (output_dir / "eval" / "downstream_eval_summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "metrics": {
                            "best_top1": 0.1,
                            "best_top3": 0.2,
                            "best_top5": 0.3,
                            "best_top10": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = e1.build_summary_row(spec, project_root=root, status="success", error_excerpt="")
            self.assertEqual(row["method_display_name"], "PrE-Text")
            self.assertEqual(row["best_top1"], 0.1)
            self.assertEqual(row["best_top10"], 0.4)


if __name__ == "__main__":
    unittest.main()
