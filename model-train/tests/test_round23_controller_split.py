from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent
if str(MODEL_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAIN_ROOT))

from split_round23_controller_dataset import build_controller_splits


class Round23ControllerSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="round23_controller_split_"))
        self.context_table_path = self.tmpdir / "round23_controller_context_table.jsonl"
        rows = []
        for dataset_name, seed_start in (
            ("jobs", 1),
            ("forums", 10),
            ("congressional", 20),
            ("microblog", 30),
        ):
            for offset in range(2):
                rows.append(
                    {
                        "context_id": f"{dataset_name}_seed{seed_start + offset}",
                        "dataset_name": dataset_name,
                        "meta_seed": seed_start + offset,
                        "oracle_best_delta_k": 0,
                        "oracle_best_target_budget": 20,
                    }
                )
        self.context_table_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_controller_splits_keeps_contexts_grouped(self) -> None:
        report = build_controller_splits(
            context_table_path=self.context_table_path,
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
        self.assertEqual(report["context_count"], 8)
        self.assertEqual(report["final_test_count"], 4)
        folds = json.loads((self.tmpdir / "splits" / "round23_cv_folds.json").read_text(encoding="utf-8"))["folds"]
        all_val_ids = {context_id for fold in folds for context_id in fold["validation_context_ids"]}
        self.assertEqual(len(all_val_ids), 4)
        for fold in folds:
            self.assertTrue(set(fold["validation_context_ids"]).isdisjoint(set(fold["training_context_ids"])))

    def test_build_controller_splits_separates_unseen_datasets(self) -> None:
        rows = []
        for dataset_name, seed_start in (
            ("jobs", 1),
            ("forums", 10),
            ("congressional", 20),
            ("microblog", 30),
            ("imdb", 40),
            ("openreview", 50),
        ):
            for offset in range(2):
                rows.append(
                    {
                        "context_id": f"{dataset_name}_seed{seed_start + offset}",
                        "dataset_name": dataset_name,
                        "meta_seed": seed_start + offset,
                        "oracle_best_delta_k": 0,
                        "oracle_best_target_budget": 20,
                    }
                )
        self.context_table_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        report = build_controller_splits(
            context_table_path=self.context_table_path,
            output_dir=self.tmpdir / "formal_splits",
            random_seed=13,
            fold_count=2,
            final_test_counts={
                "jobs": 1,
                "forums": 1,
                "congressional": 1,
                "microblog": 1,
            },
        )
        unseen_payload = json.loads(
            (self.tmpdir / "formal_splits" / "round23_unseen_test_contexts.json").read_text(encoding="utf-8")
        )
        final_test_payload = json.loads(
            (self.tmpdir / "formal_splits" / "round23_final_test_contexts.json").read_text(encoding="utf-8")
        )
        cv_payload = json.loads(
            (self.tmpdir / "formal_splits" / "round23_cv_folds.json").read_text(encoding="utf-8")
        )

        unseen_ids = set(unseen_payload["unseen_test_context_ids"])
        self.assertEqual(unseen_ids, {"imdb_seed40", "imdb_seed41", "openreview_seed50", "openreview_seed51"})
        self.assertEqual(report["train_dataset_context_count"], 8)
        self.assertEqual(report["unseen_test_context_count"], 4)
        self.assertTrue(unseen_ids.isdisjoint(set(final_test_payload["final_test_context_ids"])))
        for fold in cv_payload["folds"]:
            self.assertTrue(unseen_ids.isdisjoint(set(fold["validation_context_ids"])))
            self.assertTrue(unseen_ids.isdisjoint(set(fold["training_context_ids"])))


if __name__ == "__main__":
    unittest.main()
