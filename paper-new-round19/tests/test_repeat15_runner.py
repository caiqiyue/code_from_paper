import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from paper_new_selector.repeat15_runner import (
    REPEAT15_SUMMARY_HEADER,
    append_repeat15_summary_row,
    build_repeat15_run_specs,
    resolve_repeat15_project_root,
    run_repeat15_batch,
    write_repeat15_config,
)
from paper_new_selector.thesis_bridge import load_yaml_config


class Repeat15RunnerTests(unittest.TestCase):
    def test_resolve_repeat15_project_root_anchors_to_repo_not_resource_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "paper-new-round19"
            module_path = repo_root / "paper_new_selector" / "repeat15_runner.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# test", encoding="utf-8")

            self.assertEqual(resolve_repeat15_project_root(module_path), repo_root.resolve())

    def test_build_repeat15_run_specs_expands_four_datasets_across_fifteen_seeds(self):
        specs = build_repeat15_run_specs()

        self.assertEqual(len(specs), 60)

        first = specs[0]
        self.assertEqual(first.dataset, "jobs")
        self.assertEqual(first.seed, 1)
        self.assertEqual(first.experiment_id, "r19_repeat15_round01_jobs_seed01")
        self.assertEqual(
            first.base_config.as_posix(),
            "configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml",
        )
        self.assertEqual(
            first.relative_output_root.as_posix(),
            "paper-new-round19/outputs/repeat15_rounds/r19_repeat15_round01_jobs_seed01",
        )

        last = specs[-1]
        self.assertEqual(last.dataset, "microblog")
        self.assertEqual(last.seed, 15)
        self.assertEqual(last.experiment_id, "r19_repeat15_round15_microblog_seed15")
        self.assertEqual(
            last.base_config.as_posix(),
            "configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml",
        )

    def test_write_repeat15_config_only_overrides_seed_experiment_and_output_root(self):
        spec = build_repeat15_run_specs()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            generated = Path(temp_dir) / "repeat15_jobs_seed01.yaml"
            write_repeat15_config(spec, generated)

            config = load_yaml_config(generated)
            self.assertEqual(config["meta"]["seed"], 1)
            self.assertEqual(config["meta"]["experiment_id"], "r19_repeat15_round01_jobs_seed01")
            self.assertEqual(
                config["paths"]["output_root"],
                "paper-new-round19/outputs/repeat15_rounds/r19_repeat15_round01_jobs_seed01",
            )
            self.assertEqual(config["meta"]["stage"], "single_node_tuning_round19")
            self.assertEqual(config["data"]["dataset_name"], "jobs")
            self.assertEqual(config["selector"]["seed_budget_rule"]["mode"], "hierarchical_shape_routing")

            raw = yaml.safe_load(generated.read_text(encoding="utf-8"))
            self.assertEqual(raw["meta"]["seed"], 1)
            self.assertEqual(raw["meta"]["experiment_id"], "r19_repeat15_round01_jobs_seed01")
            self.assertEqual(
                raw["paths"]["output_root"],
                "paper-new-round19/outputs/repeat15_rounds/r19_repeat15_round01_jobs_seed01",
            )

    def test_append_repeat15_summary_row_includes_dataset_seed_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            summary_path = temp_root / "round19_full_repeat15_summary.tsv"
            summary_path.write_text("\t".join(REPEAT15_SUMMARY_HEADER) + "\n", encoding="utf-8")

            outdir = temp_root / "outputs" / "exp1"
            (outdir / "eval").mkdir(parents=True)
            (outdir / "stage1_budget_calibration.json").write_text(
                json.dumps(
                    {
                        "mode": "hierarchical_shape_routing",
                        "regime": "broad_tail",
                        "configured_seed_top_k": 20,
                        "resolved_seed_top_k": 22,
                        "runner_up_seed_top_k": 21,
                        "selection_stage": "broad_tail_policy",
                        "fallback_used": False,
                        "policy_fallback_used": False,
                        "feasible_budgets": [21, 22],
                        "shape_score": 0.61,
                    }
                ),
                encoding="utf-8",
            )
            (outdir / "eval" / "downstream_eval_summary.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "best_top1": 0.27,
                            "best_top3": 0.42,
                            "best_top5": 0.49,
                            "best_top10": 0.57,
                        }
                    }
                ),
                encoding="utf-8",
            )

            append_repeat15_summary_row(summary_path, "exp1", "jobs", 3, 0, outdir)

            rows = summary_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[1],
                "exp1\tjobs\t3\t0\thierarchical_shape_routing\tbroad_tail\t20\t22\t21\tbroad_tail_policy\tFalse\tFalse\t21,22\t0.61\t0.27\t0.42\t0.49\t0.57",
            )

    def test_run_repeat15_batch_uses_project_root_for_artifacts_and_subprocess_cwd(self):
        spec = build_repeat15_run_specs()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "paper-new-round19"
            temp_root.mkdir()

            completed = []

            def fake_write_config(run_spec, target_path):
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                Path(target_path).write_text("meta:\n  seed: 1\n", encoding="utf-8")
                return Path(target_path)

            def fake_subprocess(cmd, cwd, stdout, stderr, check):
                completed.append((cmd, cwd, Path(stdout.name)))
                return type("Completed", (), {"returncode": 1})()

            with patch("paper_new_selector.repeat15_runner.build_repeat15_run_specs", return_value=[spec]), patch(
                "paper_new_selector.repeat15_runner.write_repeat15_config",
                side_effect=fake_write_config,
            ), patch(
                "paper_new_selector.repeat15_runner.subprocess.run",
                side_effect=fake_subprocess,
            ), patch(
                "paper_new_selector.repeat15_runner.time.sleep",
                return_value=None,
            ):
                status = run_repeat15_batch(temp_root)

            self.assertEqual(status, 1)
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0][1], temp_root.resolve())
            self.assertEqual(
                completed[0][2],
                (temp_root / "logs" / "r19_repeat15_round01_jobs_seed01.log").resolve(),
            )
            self.assertTrue((temp_root / "tmp_round19_repeat15" / "r19_repeat15_round01_jobs_seed01.yaml").exists())
            self.assertTrue((temp_root / "logs" / "round19_full_repeat15_summary.tsv").exists())


if __name__ == "__main__":
    unittest.main()
