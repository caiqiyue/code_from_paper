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

from eval_round23_e4_round_count import (
    POLICY_ABSOLUTE_K,
    POLICY_KEEP,
    POLICY_ROUND23,
    evaluate_e4_round_count,
)
from train_round23_absolute_k_controller import train_absolute_k_models


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


class Round23E4AbsoluteKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="round23_e4_absk_"))
        self.controller_samples = self.tmpdir / "round23_controller_samples.jsonl"
        self.context_table = self.tmpdir / "round23_controller_context_table.jsonl"
        self.final_test = self.tmpdir / "round23_final_test_contexts.json"
        self.unseen_test = self.tmpdir / "round23_unseen_test_contexts.json"
        self.cv_folds = self.tmpdir / "round23_cv_folds.json"
        self.config = self.tmpdir / "train_round23_absolute_k_controller.yaml"
        self.model_dir = self.tmpdir / "models"
        self.report_dir = self.tmpdir / "reports"
        self.round19_table = self.tmpdir / "round19_replay_table.jsonl"
        self.round23_eval_report = self.tmpdir / "round23_eval_report.json"

        budgets = [18, 19, 20, 21, 22]
        sample_rows = []
        for context_id, dataset_name, reward_base in (
            ("jobs_seed1", "jobs", 0.10),
            ("jobs_seed2", "jobs", 0.20),
            ("forums_seed1", "forums", 0.30),
        ):
            for budget in budgets:
                sample_rows.append(
                    {
                        "context_id": context_id,
                        "dataset_name": dataset_name,
                        "target_budget": budget,
                        "shape_score": 0.1,
                        "private_mean_length": 10.0,
                        "private_p75_length": 11.0,
                        "private_length_iqr": 2.0,
                        "support_mean_at_k20": 0.5,
                        "coverage_mean_at_k20": 0.6,
                        "coverage_p25_at_k20": 0.4,
                        "genericity_mean_at_k20": 0.3,
                        "redundancy_mean_at_k20": 0.2,
                        "reward_round23_controller": reward_base + budget * 0.001,
                        "target_value_for_training": reward_base + budget * 0.002,
                        "label_target_mode": "top1_delta",
                        "tie_margin": 0.0005,
                    }
                )
        _write_jsonl(self.controller_samples, sample_rows)

        context_rows = []
        for context_id, dataset_name, oracle_delta in (
            ("jobs_seed1", "jobs", 0),
            ("jobs_seed2", "jobs", 1),
            ("forums_seed1", "forums", -1),
        ):
            row = {
                "context_id": context_id,
                "dataset_name": dataset_name,
                "oracle_best_delta_k": oracle_delta,
                "oracle_best_controller_reward": 0.0,
                "oracle_best_top1": 0.0,
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
            for delta_k in (-2, -1, 0, 1, 2):
                suffix = f"neg{abs(delta_k)}" if delta_k < 0 else ("0" if delta_k == 0 else f"pos{delta_k}")
                row[f"controller_reward_dk_{suffix}"] = 0.5 + delta_k * 0.05
                row[f"best_top1_dk_{suffix}"] = 0.4 + delta_k * 0.02
            row["keep_k0_reward"] = row["controller_reward_dk_0"]
            row["keep_k0_training_value"] = row["best_top1_dk_0"]
            oracle_suffix = "0" if oracle_delta == 0 else (f"neg{abs(oracle_delta)}" if oracle_delta < 0 else f"pos{oracle_delta}")
            row["oracle_best_controller_reward"] = row[f"controller_reward_dk_{oracle_suffix}"]
            row["oracle_best_top1"] = row[f"best_top1_dk_{oracle_suffix}"]
            context_rows.append(row)
        _write_jsonl(self.context_table, context_rows)

        _write_jsonl(
            self.round19_table,
            [
                {
                    "context_id": "jobs_seed1",
                    "round19_predicted_delta_k": 0,
                    "round19_predicted_budget": 20,
                    "round19_replay_reward": 0.50,
                    "round19_replay_best_top1": 0.40,
                },
                {
                    "context_id": "jobs_seed2",
                    "round19_predicted_delta_k": 1,
                    "round19_predicted_budget": 21,
                    "round19_replay_reward": 0.55,
                    "round19_replay_best_top1": 0.42,
                },
                {
                    "context_id": "forums_seed1",
                    "round19_predicted_delta_k": -1,
                    "round19_predicted_budget": 19,
                    "round19_replay_reward": 0.45,
                    "round19_replay_best_top1": 0.38,
                },
            ],
        )

        _write_json(
            self.round23_eval_report,
            {
                "per_context": [
                    {
                        "context_id": "jobs_seed1",
                        "predicted_delta_k": 0,
                        "predicted_target_budget": 20,
                        "predicted_rewards": {"-2": 0.1, "-1": 0.2, "0": 0.8, "1": 0.3, "2": 0.1},
                    },
                    {
                        "context_id": "jobs_seed2",
                        "predicted_delta_k": 1,
                        "predicted_target_budget": 21,
                        "predicted_rewards": {"-2": 0.1, "-1": 0.2, "0": 0.3, "1": 0.9, "2": 0.4},
                    },
                    {
                        "context_id": "forums_seed1",
                        "predicted_delta_k": -1,
                        "predicted_target_budget": 19,
                        "predicted_rewards": {"-2": 0.1, "-1": 0.7, "0": 0.2, "1": 0.1, "2": 0.0},
                    },
                ]
            },
        )

        _write_json(self.final_test, {"final_test_context_ids": ["jobs_seed2"]})
        _write_json(self.unseen_test, {"unseen_test_context_ids": ["forums_seed1"]})
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
  absolute_budgets: [18, 19, 20, 21, 22]
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

    def test_train_absolute_k_models_excludes_unseen_rows_from_final_retrain(self) -> None:
        with (
            patch("train_round23_absolute_k_controller.build_regressor", return_value=_DummyRegressor()),
            patch(
                "train_round23_absolute_k_controller.save_regressor",
                lambda family, model, path: Path(path).write_text("dummy", encoding="utf-8"),
            ),
        ):
            report = train_absolute_k_models(
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
        self.assertEqual(report["absolute_budgets"], [18, 19, 20, 21, 22])

    def test_evaluate_e4_round_count_includes_absolute_k_and_round23(self) -> None:
        class _EvalRegressor:
            def __init__(self, value: float):
                self._value = value

            def predict(self, x):
                return [self._value for _ in x]

        values_by_budget = {
            "model_k18.pkl": 0.1,
            "model_k19.pkl": 0.2,
            "model_k20.pkl": 0.3,
            "model_k21.pkl": 0.9,
            "model_k22.pkl": 0.4,
        }

        def _fake_load_regressor(*, family, path):
            return _EvalRegressor(values_by_budget[Path(path).name])

        with patch("eval_round23_e4_round_count.load_regressor", side_effect=_fake_load_regressor):
            report = evaluate_e4_round_count(
                controller_context_table=self.context_table,
                round19_replay_table=self.round19_table,
                output_dir=self.report_dir,
                absolute_k_model_dir=self.model_dir,
                absolute_k_model_family="linear_baseline",
                absolute_k_feature_version="no_dataset",
                absolute_k_config_path=self.config,
                round23_eval_report=self.round23_eval_report,
                scope="seen4",
            )
        overall = report["tables"]["e4_table_policy_quality"]
        policies = {row["policy"] for row in overall}
        self.assertEqual(
            policies,
            {POLICY_KEEP, POLICY_ABSOLUTE_K, POLICY_ROUND23},
        )
        absk_row = next(row for row in overall if row["policy"] == POLICY_ABSOLUTE_K)
        self.assertEqual(absk_row["contexts"], 3)
        datasetwise = report["tables"]["e4_table_datasetwise_policy_quality"]
        self.assertTrue(any(row["dataset_name"] == "jobs" for row in datasetwise))


if __name__ == "__main__":
    unittest.main()
