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

from build_round23_controller_dataset import build_controller_dataset
from common import BUDGETS


class Round23ControllerDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="round23_controller_dataset_"))
        self.action_samples_path = self.tmpdir / "all_action_samples.jsonl"
        self.contexts_path = self.tmpdir / "all_contexts.jsonl"
        self._write_fixture_data()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_fixture_data(self) -> None:
        context_rows = [
            {
                "context_id": "jobs_seed1",
                "dataset_name": "jobs",
                "meta_seed": 1,
                "shape_score": 1.5,
                "private_mean_length": 120.0,
                "private_p75_length": 140.0,
                "private_length_iqr": 60.0,
                "support_mean_at_k20": 10.0,
                "coverage_mean_at_k20": 0.30,
                "coverage_p25_at_k20": 0.20,
                "genericity_mean_at_k20": 0.05,
                "redundancy_mean_at_k20": 0.01,
                "reward_k18": 0.46,
                "reward_k19": 0.49,
                "reward_k20": 0.50,
                "reward_k21": 0.53,
                "reward_k22": 0.45,
                "best_top1_k18": 0.46,
                "best_top1_k19": 0.495,
                "best_top1_k20": 0.501,
                "best_top1_k21": 0.545,
                "best_top1_k22": 0.47,
                "coverage_p25_k18": 0.18,
                "coverage_p25_k19": 0.19,
                "coverage_p25_k20": 0.20,
                "coverage_p25_k21": 0.24,
                "coverage_p25_k22": 0.17,
                "support_mean_k18": 10.4,
                "support_mean_k19": 10.2,
                "support_mean_k20": 10.0,
                "support_mean_k21": 9.8,
                "support_mean_k22": 9.5,
                "oracle_best_k": 21,
                "oracle_best_reward": 0.53,
            },
            {
                "context_id": "forums_seed2",
                "dataset_name": "forums",
                "meta_seed": 2,
                "shape_score": 3.5,
                "private_mean_length": 220.0,
                "private_p75_length": 260.0,
                "private_length_iqr": 90.0,
                "support_mean_at_k20": 12.0,
                "coverage_mean_at_k20": 0.28,
                "coverage_p25_at_k20": 0.16,
                "genericity_mean_at_k20": 0.04,
                "redundancy_mean_at_k20": 0.03,
                "reward_k18": 0.31,
                "reward_k19": 0.34,
                "reward_k20": 0.35,
                "reward_k21": 0.33,
                "reward_k22": 0.30,
                "best_top1_k18": 0.31,
                "best_top1_k19": 0.342,
                "best_top1_k20": 0.351,
                "best_top1_k21": 0.336,
                "best_top1_k22": 0.308,
                "coverage_p25_k18": 0.14,
                "coverage_p25_k19": 0.15,
                "coverage_p25_k20": 0.16,
                "coverage_p25_k21": 0.15,
                "coverage_p25_k22": 0.13,
                "support_mean_k18": 12.3,
                "support_mean_k19": 12.1,
                "support_mean_k20": 12.0,
                "support_mean_k21": 11.6,
                "support_mean_k22": 11.5,
                "oracle_best_k": 20,
                "oracle_best_reward": 0.35,
            },
        ]
        action_rows = []
        for context in context_rows:
            for budget in (18, 19, 20, 21, 22):
                action_rows.append(
                    {
                        "experiment_id": f"r22dc_{context['dataset_name']}_seed{context['meta_seed']}_k{budget}",
                        "context_id": context["context_id"],
                        "dataset_name": context["dataset_name"],
                        "meta_seed": context["meta_seed"],
                        "action_budget": budget,
                        "normalized_budget_cost": {18: 0.0, 19: 0.25, 20: 0.5, 21: 0.75, 22: 1.0}[budget],
                        "reward": context[f"reward_k{budget}"],
                        "best_top1": context[f"best_top1_k{budget}"],
                        "shape_score": context["shape_score"],
                        "private_mean_length": context["private_mean_length"],
                        "private_p75_length": context["private_p75_length"],
                        "private_length_iqr": context["private_length_iqr"],
                        "support_mean_at_k20": context["support_mean_at_k20"],
                        "coverage_mean_at_k20": context["coverage_mean_at_k20"],
                        "coverage_p25_at_k20": context["coverage_p25_at_k20"],
                        "genericity_mean_at_k20": context["genericity_mean_at_k20"],
                        "redundancy_mean_at_k20": context["redundancy_mean_at_k20"],
                    }
                )
        self.action_samples_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in action_rows) + "\n",
            encoding="utf-8",
        )
        self.contexts_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in context_rows) + "\n",
            encoding="utf-8",
        )

    def test_build_controller_dataset_expands_each_context_to_five_actions(self) -> None:
        report = build_controller_dataset(
            action_samples_path=self.action_samples_path,
            context_table_path=self.contexts_path,
            output_dir=self.tmpdir / "outputs",
        )
        self.assertEqual(report["controller_sample_count"], 10)
        self.assertEqual(report["context_count"], 2)
        rows = [
            json.loads(line)
            for line in (self.tmpdir / "outputs" / "round23_controller_samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        jobs_rows = [row for row in rows if row["context_id"] == "jobs_seed1"]
        self.assertEqual(len(jobs_rows), 5)
        self.assertEqual(sorted(row["action_delta_k"] for row in jobs_rows), [-2, -1, 0, 1, 2])

    def test_build_controller_dataset_computes_reward_formula(self) -> None:
        build_controller_dataset(
            action_samples_path=self.action_samples_path,
            context_table_path=self.contexts_path,
            output_dir=self.tmpdir / "outputs",
        )
        rows = [
            json.loads(line)
            for line in (self.tmpdir / "outputs" / "round23_controller_samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        row = next(
            item
            for item in rows
            if item["context_id"] == "jobs_seed1" and int(item["target_budget"]) == 21
        )
        expected = (
            1.0 * (0.545 - 0.501)
            + 0.25 * (0.24 - 0.20)
            - 0.20 * max(0.0, 10.0 - 9.8)
            - 0.02 * 1.0
        )
        self.assertAlmostEqual(row["reward_round23_controller"], expected, places=9)

    def test_build_controller_dataset_sets_oracle_best_delta(self) -> None:
        build_controller_dataset(
            action_samples_path=self.action_samples_path,
            context_table_path=self.contexts_path,
            output_dir=self.tmpdir / "outputs",
        )
        rows = [
            json.loads(line)
            for line in (self.tmpdir / "outputs" / "round23_controller_context_table.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        jobs_row = next(row for row in rows if row["context_id"] == "jobs_seed1")
        self.assertEqual(jobs_row["oracle_best_delta_k"], 0)
        self.assertEqual(jobs_row["oracle_best_target_budget"], 20)

    def test_build_controller_dataset_from_collection_schema_outputs_round19_replay(self) -> None:
        collection_root = self.tmpdir / "collection_fixture"
        outputs_root = collection_root / "outputs"
        manifest_path = collection_root / "manifest.tsv"
        outputs_root.mkdir(parents=True, exist_ok=True)

        context_specs = [
            {
                "dataset_name": "jobs",
                "meta_seed": 42,
                "shape_score": None,
                "private_mean_length": 140.0,
                "private_p75_length": 180.0,
                "private_length_iqr": 55.0,
                "median_len": 135.0,
                "tail_ratio": 0.18,
                "short_ratio": 0.05,
                "private_lengths": [80, 120, 135, 170, 220],
                "best_top1": {18: 0.41, 19: 0.46, 20: 0.50, 21: 0.53, 22: 0.49},
                "support_mean": {18: 8.8, 19: 8.5, 20: 8.2, 21: 8.0, 22: 7.7},
                "coverage_mean": {18: 0.32, 19: 0.35, 20: 0.37, 21: 0.39, 22: 0.34},
                "coverage_p25": {18: 0.21, 19: 0.24, 20: 0.26, 21: 0.29, 22: 0.22},
                "genericity_mean": {18: 0.08, 19: 0.09, 20: 0.10, 21: 0.11, 22: 0.10},
                "redundancy_mean": {18: 0.01, 19: 0.02, 20: 0.03, 21: 0.04, 22: 0.05},
            },
            {
                "dataset_name": "imdb",
                "meta_seed": 55,
                "shape_score": 2.0,
                "private_mean_length": 210.0,
                "private_p75_length": 250.0,
                "private_length_iqr": 75.0,
                "median_len": 205.0,
                "tail_ratio": 0.30,
                "short_ratio": 0.02,
                "private_lengths": [120, 180, 205, 260, 310],
                "best_top1": {18: 0.37, 19: 0.40, 20: 0.43, 21: 0.45, 22: 0.41},
                "support_mean": {18: 9.5, 19: 9.1, 20: 8.8, 21: 8.4, 22: 8.1},
                "coverage_mean": {18: 0.28, 19: 0.30, 20: 0.31, 21: 0.33, 22: 0.29},
                "coverage_p25": {18: 0.17, 19: 0.19, 20: 0.20, 21: 0.22, 22: 0.18},
                "genericity_mean": {18: 0.06, 19: 0.07, 20: 0.08, 21: 0.09, 22: 0.08},
                "redundancy_mean": {18: 0.02, 19: 0.03, 20: 0.04, 21: 0.05, 22: 0.06},
            },
        ]

        manifest_lines = ["experiment_id\tdataset_name\tmeta_seed\tbudget_k\tconfig_path\toutput_root\tgroup_name"]
        for spec in context_specs:
            dataset_name = spec["dataset_name"]
            meta_seed = spec["meta_seed"]
            for budget_k in BUDGETS:
                output_root = outputs_root / dataset_name / f"seed{meta_seed}" / f"k{budget_k}"
                collection_dir = output_root / "collection"
                collection_dir.mkdir(parents=True, exist_ok=True)
                context_summary = {
                    "dataset_name": dataset_name,
                    "meta_seed": meta_seed,
                    "context_id": f"{dataset_name}_seed{meta_seed}",
                    "shape_score": spec["shape_score"],
                    "private_mean_length": spec["private_mean_length"],
                    "private_p75_length": spec["private_p75_length"],
                    "private_length_iqr": spec["private_length_iqr"],
                    "median_len": spec["median_len"],
                    "tail_ratio": spec["tail_ratio"],
                    "short_ratio": spec["short_ratio"],
                    "private_lengths": spec["private_lengths"],
                }
                budget_rows = []
                for candidate_budget in BUDGETS:
                    budget_rows.append(
                        {
                            "context_id": f"{dataset_name}_seed{meta_seed}",
                            "dataset_name": dataset_name,
                            "meta_seed": meta_seed,
                            "budget_k": candidate_budget,
                            "budget_cost": float(candidate_budget - 18),
                            "normalized_budget_cost": float(candidate_budget - 18) / 4.0,
                            "selected_seed_count": candidate_budget,
                            "support_mean_k": spec["support_mean"][candidate_budget],
                            "coverage_mean_k": spec["coverage_mean"][candidate_budget],
                            "coverage_p25_k": spec["coverage_p25"][candidate_budget],
                            "genericity_mean_k": spec["genericity_mean"][candidate_budget],
                            "redundancy_mean_k": spec["redundancy_mean"][candidate_budget],
                        }
                    )
                final_result_summary = {
                    "best_top1": spec["best_top1"][budget_k],
                    "best_top3": spec["best_top1"][budget_k] + 0.05,
                    "best_top5": spec["best_top1"][budget_k] + 0.08,
                    "best_top10": spec["best_top1"][budget_k] + 0.10,
                }
                (collection_dir / "context_summary.json").write_text(
                    json.dumps(context_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (collection_dir / "budget_table.jsonl").write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in budget_rows) + "\n",
                    encoding="utf-8",
                )
                (collection_dir / "final_result_summary.json").write_text(
                    json.dumps(final_result_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                manifest_lines.append(
                    "\t".join(
                        [
                            f"r19_r23c_{dataset_name}_seed{meta_seed}_k{budget_k}",
                            dataset_name,
                            str(meta_seed),
                            str(budget_k),
                            f"configs/{dataset_name}/seed{meta_seed}/k{budget_k}.yaml",
                            str(output_root).replace("\\", "/"),
                            "round19_round23_collection_repeat40",
                        ]
                    )
                )

        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        report = build_controller_dataset(
            collection_manifest_path=manifest_path,
            output_dir=self.tmpdir / "formal_outputs",
        )

        self.assertEqual(report["build_mode"], "round19_round23_collection_repeat40")
        self.assertEqual(report["controller_sample_count"], 10)
        self.assertEqual(report["context_count"], 2)
        self.assertEqual(report["dataset_partition_summary"]["train"]["context_count"], 1)
        self.assertEqual(report["dataset_partition_summary"]["unseen_test"]["context_count"], 1)

        sample_rows = [
            json.loads(line)
            for line in (self.tmpdir / "formal_outputs" / "round23_controller_samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        k20_row = next(
            row
            for row in sample_rows
            if row["context_id"] == "jobs_seed42" and int(row["action_delta_k"]) == 0
        )
        self.assertEqual(k20_row["best_top1_at_k20"], 0.50)
        self.assertIn("target_value_for_training", k20_row)
        self.assertIn("reward_round23_controller_old", k20_row)
        self.assertEqual(k20_row["label_target_mode"], "top1_delta")
        self.assertIsInstance(k20_row["shape_score"], float)
        self.assertNotIn("best_top1_at_k20", report["feature_fields"])

        context_rows = [
            json.loads(line)
            for line in (self.tmpdir / "formal_outputs" / "round23_controller_context_table.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        jobs_context = next(row for row in context_rows if row["context_id"] == "jobs_seed42")
        self.assertEqual(jobs_context["oracle_best_delta_k"], 1)
        self.assertEqual(jobs_context["oracle_best_target_budget"], 21)
        self.assertIn("best_top1_dk_pos1", jobs_context)
        self.assertIn("controller_old_reward_dk_0", jobs_context)

        replay_rows = [
            json.loads(line)
            for line in (self.tmpdir / "formal_outputs" / "round19_replay_table.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(replay_rows), 2)
        for row in replay_rows:
            self.assertIn(int(row["round19_predicted_budget"]), BUDGETS)
            self.assertIn("round19_replay_reward", row)
            self.assertIn("round19_reward_delta_vs_keep_k20", row)


if __name__ == "__main__":
    unittest.main()
