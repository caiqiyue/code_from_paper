from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent
if str(MODEL_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAIN_ROOT))

from eval_round23_controller import evaluate_controller
from train_round23_controller import train_controller_models


class _DummyRegressor:
    def fit(self, x, y) -> None:
        self._mean = sum(y) / len(y) if y else 0.0

    def predict(self, x):
        return [getattr(self, "_mean", 0.0) for _ in x]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


class Round23TrainingEvalPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="round23_train_eval_paths_"))
        self.controller_samples = self.tmpdir / "round23_controller_samples.jsonl"
        self.context_table = self.tmpdir / "round23_controller_context_table.jsonl"
        self.final_test = self.tmpdir / "round23_final_test_contexts.json"
        self.unseen_test = self.tmpdir / "round23_unseen_test_contexts.json"
        self.cv_folds = self.tmpdir / "round23_cv_folds.json"
        self.config = self.tmpdir / "train_round23_controller.yaml"
        self.model_dir = self.tmpdir / "models"
        self.report_dir = self.tmpdir / "reports"

        actions = [-2, -1, 0, 1, 2]
        sample_rows = []
        for context_id, dataset_name, reward_base in (
            ("jobs_seed1", "jobs", 1.0),
            ("jobs_seed2", "jobs", 2.0),
            ("imdb_seed1", "imdb", 10.0),
        ):
            for delta_k in actions:
                sample_rows.append(
                    {
                        "context_id": context_id,
                        "dataset_name": dataset_name,
                        "action_delta_k": delta_k,
                        "shape_score": 0.1,
                        "private_mean_length": 10.0,
                        "private_p75_length": 11.0,
                        "private_length_iqr": 2.0,
                        "support_mean_at_k20": 0.5,
                        "coverage_mean_at_k20": 0.6,
                        "coverage_p25_at_k20": 0.4,
                        "genericity_mean_at_k20": 0.3,
                        "redundancy_mean_at_k20": 0.2,
                        "reward_round23_controller": reward_base + delta_k * 0.01,
                        "target_value_for_training": reward_base + delta_k * 0.02,
                        "label_target_mode": "top1_delta",
                        "tie_margin": 0.0005,
                    }
                )
        _write_jsonl(self.controller_samples, sample_rows)

        context_rows = []
        for context_id, dataset_name, oracle_delta in (
            ("jobs_seed1", "jobs", 0),
            ("jobs_seed2", "jobs", 1),
            ("imdb_seed1", "imdb", -1),
        ):
            row = {
                "context_id": context_id,
                "dataset_name": dataset_name,
                "oracle_best_delta_k": oracle_delta,
                "oracle_best_controller_reward": 1.0,
                "oracle_best_top1": 0.55,
                "label_target_mode": "top1_delta",
                "tie_margin": 0.0005,
                "shape_score": 0.1,
                "private_mean_length": 10.0,
                "private_p75_length": 11.0,
                "private_length_iqr": 2.0,
                "support_mean_at_k20": 0.5,
                "coverage_mean_at_k20": 0.6,
                "coverage_p25_at_k20": 0.4,
                "genericity_mean_at_k20": 0.3,
                "redundancy_mean_at_k20": 0.2,
            }
            for delta_k in actions:
                suffix = f"neg{abs(delta_k)}" if delta_k < 0 else ("0" if delta_k == 0 else f"pos{delta_k}")
                row[f"controller_reward_dk_{suffix}"] = 0.8 + delta_k * 0.05
                row[f"best_top1_dk_{suffix}"] = 0.5 + delta_k * 0.01
            context_rows.append(row)
        _write_jsonl(self.context_table, context_rows)

        _write_json(self.final_test, {"final_test_context_ids": ["jobs_seed2"]})
        _write_json(self.unseen_test, {"unseen_test_context_ids": ["imdb_seed1"]})
        _write_json(
            self.cv_folds,
            {
                "folds": [
                    {
                        "fold_index": 0,
                        "training_context_ids": ["jobs_seed1"],
                        "validation_context_ids": [],
                    }
                ]
            },
        )
        self.config.write_text(
            """
model:
  delta_actions: [-2, -1, 0, 1, 2]
  feature_version: no_dataset
  feature_fields:
    - shape_score
    - private_mean_length
    - private_p75_length
    - private_length_iqr
    - support_mean_at_k20
    - coverage_mean_at_k20
    - coverage_p25_at_k20
    - genericity_mean_at_k20
    - redundancy_mean_at_k20
  include_dataset_one_hot: false
training:
  param_candidates:
    linear_baseline:
      - name: ridge_default
        alpha: 1.0
""".strip(),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_train_controller_models_excludes_unseen_rows_from_final_retrain(self) -> None:
        with (
            patch("train_round23_controller.build_regressor", return_value=_DummyRegressor()),
            patch("train_round23_controller.save_regressor", lambda family, model, path: Path(path).write_text("dummy", encoding="utf-8")),
        ):
            report = train_controller_models(
                controller_samples_path=self.controller_samples,
                final_test_path=self.final_test,
                unseen_test_path=self.unseen_test,
                cv_folds_path=self.cv_folds,
                config_path=self.config,
                model_output_dir=self.model_dir,
                model_family="linear_baseline",
                feature_version="no_dataset",
                target_field="target_value_for_training",
            )
        self.assertEqual(report["train_context_count"], 1)
        self.assertEqual(report["final_test_context_count"], 1)
        self.assertEqual(report["unseen_test_context_count"], 1)
        self.assertEqual(report["excluded_context_count"], 2)
        self.assertEqual(report["train_row_count"], 5)
        self.assertEqual(report["target_field"], "target_value_for_training")
        self.assertEqual(report["target_mode"], "top1_delta")

    def test_evaluate_controller_supports_unseen_split_payload(self) -> None:
        class _EvalRegressor:
            def predict(self, x):
                return [0.0 for _ in x]

        with patch("eval_round23_controller.load_regressor", return_value=_EvalRegressor()):
            report = evaluate_controller(
                controller_context_table_path=self.context_table,
                context_split_path=self.unseen_test,
                context_split_key="unseen_test_context_ids",
                config_path=self.config,
                model_dir=self.model_dir,
                report_dir=self.report_dir,
                model_family="linear_baseline",
                feature_version="no_dataset",
                round19_replay_path=None,
                target_field="target_value_for_training",
            )
        self.assertEqual(report["context_count"], 1)
        self.assertEqual(report["context_split_key"], "unseen_test_context_ids")
        self.assertEqual(report["per_context"][0]["context_id"], "imdb_seed1")
        self.assertEqual(report["target_field"], "target_value_for_training")
        self.assertIn("mean_best_top1_regret", report)


if __name__ == "__main__":
    unittest.main()
