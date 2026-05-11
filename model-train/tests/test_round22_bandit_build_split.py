from __future__ import annotations

import json
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent
if str(MODEL_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAIN_ROOT))

from build_round22_bandit_dataset import build_dataset
from split_round22_bandit_dataset import build_splits


class Round22BanditBuildSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="round22_bandit_test_"))
        self.summary = THIS_DIR / "fixtures" / "round22_synthetic_summary.jsonl"
        self.schema = (
            Path(__file__).resolve().parents[2]
            / "paper-new-round22"
            / "configs"
            / "experiments"
            / "bandit_data_collection"
            / "round22_bandit_record_schema.yaml"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_dataset_creates_complete_tables(self) -> None:
        report = build_dataset(
            summary_jsonl=self.summary,
            schema_yaml=self.schema,
            output_dir=self.tmpdir / "datasets",
        )
        self.assertEqual(report["action_sample_count"], 20)
        self.assertEqual(report["context_count"], 4)
        context_table = self.tmpdir / "datasets" / "round22_context_table.jsonl"
        rows = [json.loads(line) for line in context_table.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 4)
        jobs_row = next(row for row in rows if row["dataset_name"] == "jobs")
        self.assertEqual(jobs_row["oracle_best_k"], 19)

    def test_split_dataset_respects_dataset_holdout_counts(self) -> None:
        build_dataset(
            summary_jsonl=self.summary,
            schema_yaml=self.schema,
            output_dir=self.tmpdir / "datasets",
        )
        report = build_splits(
            context_table_path=self.tmpdir / "datasets" / "round22_context_table.jsonl",
            output_dir=self.tmpdir / "splits",
            random_seed=7,
            fold_count=2,
            final_test_counts={
                "jobs": 1,
                "forums": 1,
                "congressional": 1,
                "microblog": 1,
            },
        )
        self.assertEqual(report["context_count"], 4)
        self.assertEqual(report["final_test_count"], 4)
        final_test = json.loads((self.tmpdir / "splits" / "round22_final_test_contexts.json").read_text(encoding="utf-8"))
        self.assertEqual(len(final_test["final_test_context_ids"]), 4)

    def test_split_cli_accepts_inline_final_test_counts_override(self) -> None:
        build_dataset(
            summary_jsonl=self.summary,
            schema_yaml=self.schema,
            output_dir=self.tmpdir / "datasets",
        )
        script = MODEL_TRAIN_ROOT / "split_round22_bandit_dataset.py"
        output_dir = self.tmpdir / "cli_splits"
        override = json.dumps(
            {
                "jobs": 1,
                "forums": 1,
                "congressional": 1,
                "microblog": 1,
            }
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--context-table",
                str(self.tmpdir / "datasets" / "round22_context_table.jsonl"),
                "--output-dir",
                str(output_dir),
                "--random-seed",
                "7",
                "--fold-count",
                "2",
                "--final-test-counts-json",
                override,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("SPLIT contexts=4 final_test=4 dev=0 folds=2", proc.stdout)
        final_test = json.loads((output_dir / "round22_final_test_contexts.json").read_text(encoding="utf-8"))
        self.assertEqual(len(final_test["final_test_context_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
