import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from paper_new_selector.round20_repeat5_runner import (
    ROUND20_REPEAT5_SUMMARY_HEADER,
    append_round20_repeat5_summary_row,
    build_round20_repeat5_run_specs,
    resolve_round20_repeat5_project_root,
    resolve_round20_repeat5_runtime_output_dir,
    run_round20_repeat5_batch,
    write_round20_repeat5_config,
)
from paper_new_selector.thesis_bridge import load_yaml_config


class Round20Repeat5RunnerTests(unittest.TestCase):
    def test_resolve_round20_repeat5_project_root_anchors_to_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "paper-new-round20"
            module_path = repo_root / "paper_new_selector" / "round20_repeat5_runner.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# test", encoding="utf-8")

            self.assertEqual(resolve_round20_repeat5_project_root(module_path), repo_root.resolve())

    def test_build_round20_repeat5_run_specs_expands_twenty_configs_across_five_repeats(self):
        specs = build_round20_repeat5_run_specs()
        self.assertEqual(len(specs), 100)

        first = specs[0]
        self.assertEqual(first.base_experiment_id, "r20_scan_jobs_delta180_seed42")
        self.assertEqual(first.dataset, "jobs")
        self.assertEqual(first.base_seed, 42)
        self.assertEqual(first.repeat_index, 1)
        self.assertEqual(first.experiment_id, "r20_scan_jobs_delta180_seed42_repeat01")
        self.assertEqual(
            first.relative_output_root.as_posix(),
            "paper-new-round20/outputs/r20_repeat5/r20_scan_jobs_delta180_seed42_repeat01",
        )

        last = specs[-1]
        self.assertEqual(last.base_experiment_id, "r20_microblog_tau300_seed456_fallback")
        self.assertEqual(last.dataset, "microblog")
        self.assertEqual(last.repeat_index, 5)
        self.assertEqual(last.experiment_id, "r20_microblog_tau300_seed456_fallback_repeat05")

    def test_write_round20_repeat5_config_only_overrides_experiment_and_output_root(self):
        spec = build_round20_repeat5_run_specs()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            generated = Path(temp_dir) / "round20_repeat5.yaml"
            write_round20_repeat5_config(spec, generated)

            config = load_yaml_config(generated)
            self.assertEqual(config["meta"]["seed"], 42)
            self.assertEqual(config["meta"]["experiment_id"], "r20_scan_jobs_delta180_seed42_repeat01")
            self.assertEqual(
                config["paths"]["output_root"],
                "paper-new-round20/outputs/r20_repeat5/r20_scan_jobs_delta180_seed42_repeat01",
            )
            self.assertEqual(config["selector"]["seed_budget_rule"]["router"]["delta_router"], 1.8)
            self.assertEqual(
                config["selector"]["seed_budget_rule"]["policies"]["uncertain"]["arbitration_enabled"],
                False,
            )

            raw = yaml.safe_load(generated.read_text(encoding="utf-8"))
            self.assertEqual(raw["meta"]["experiment_id"], "r20_scan_jobs_delta180_seed42_repeat01")

    def test_resolve_round20_repeat5_runtime_output_dir(self):
        spec = build_round20_repeat5_run_specs()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "paper-new-round20"
            repo_root.mkdir()

            self.assertEqual(
                resolve_round20_repeat5_runtime_output_dir(repo_root, spec),
                repo_root / "outputs" / "r20_repeat5" / "r20_scan_jobs_delta180_seed42_repeat01",
            )

    def test_append_round20_repeat5_summary_row_includes_repeat_metadata_and_metrics(self):
        spec = build_round20_repeat5_run_specs()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            summary_path = temp_root / "round20_repeat5_summary.tsv"
            summary_path.write_text("\t".join(ROUND20_REPEAT5_SUMMARY_HEADER) + "\n", encoding="utf-8")

            outdir = temp_root / "outputs" / "exp1"
            (outdir / "eval").mkdir(parents=True)
            (outdir / "stage1_budget_calibration.json").write_text(
                json.dumps(
                    {
                        "mode": "hierarchical_shape_routing",
                        "regime": "uncertain",
                        "configured_seed_top_k": 20,
                        "resolved_seed_top_k": 21,
                        "selection_stage": "uncertainty_policy_arbitration",
                        "arbitration_triggered": True,
                        "arbitration_winner_policy": "broad_tail",
                    }
                ),
                encoding="utf-8",
            )
            (outdir / "eval" / "downstream_eval_summary.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "best_top1": 0.29,
                            "best_top3": 0.43,
                            "best_top5": 0.50,
                            "best_top10": 0.58,
                        }
                    }
                ),
                encoding="utf-8",
            )

            append_round20_repeat5_summary_row(summary_path, spec, 0, outdir)
            rows = summary_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[1],
                "r20_scan_jobs_delta180_seed42_repeat01\tr20_scan_jobs_delta180_seed42\tjobs\t42\t1\t0\thierarchical_shape_routing\tuncertain\t20\t21\tuncertainty_policy_arbitration\tTrue\tbroad_tail\t0.29\t0.43\t0.5\t0.58",
            )

    def test_run_round20_repeat5_batch_uses_project_root_and_serial_subprocesses(self):
        spec = build_round20_repeat5_run_specs()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "paper-new-round20"
            temp_root.mkdir()
            completed = []

            def fake_write_config(run_spec, target_path):
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                Path(target_path).write_text("meta:\n  experiment_id: x\n", encoding="utf-8")
                return Path(target_path)

            def fake_subprocess(cmd, cwd, stdout, stderr, check, **kwargs):
                completed.append((cmd, cwd, Path(stdout.name)))
                return type("Completed", (), {"returncode": 1})()

            with patch(
                "paper_new_selector.round20_repeat5_runner.build_round20_repeat5_run_specs",
                return_value=[spec],
            ), patch(
                "paper_new_selector.round20_repeat5_runner.write_round20_repeat5_config",
                side_effect=fake_write_config,
            ), patch(
                "paper_new_selector.round20_repeat5_runner.subprocess.run",
                side_effect=fake_subprocess,
            ), patch(
                "paper_new_selector.round20_repeat5_runner.time.sleep",
                return_value=None,
            ):
                status = run_round20_repeat5_batch(temp_root, repeat_count=5)

            self.assertEqual(status, 1)
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0][1], temp_root.resolve())
            self.assertEqual(
                completed[0][2],
                (temp_root / "logs" / "r20_scan_jobs_delta180_seed42_repeat01.log").resolve(),
            )
            self.assertTrue(
                (temp_root / "tmp_round20_repeat5" / "r20_scan_jobs_delta180_seed42_repeat01.yaml").exists()
            )
            self.assertTrue((temp_root / "logs" / "round20_repeat5_summary.tsv").exists())

    def test_run_round20_repeat5_batch_pins_cuda_device_order(self):
        spec = build_round20_repeat5_run_specs()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "paper-new-round20"
            temp_root.mkdir()
            completed = []

            def fake_write_config(run_spec, target_path):
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                Path(target_path).write_text("meta:\n  experiment_id: x\n", encoding="utf-8")
                return Path(target_path)

            def fake_subprocess(cmd, cwd, stdout, stderr, check, **kwargs):
                completed.append(dict(kwargs.get("env", {})))
                return type("Completed", (), {"returncode": 1})()

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False), patch(
                "paper_new_selector.round20_repeat5_runner.build_round20_repeat5_run_specs",
                return_value=[spec],
            ), patch(
                "paper_new_selector.round20_repeat5_runner.write_round20_repeat5_config",
                side_effect=fake_write_config,
            ), patch(
                "paper_new_selector.round20_repeat5_runner.subprocess.run",
                side_effect=fake_subprocess,
            ), patch(
                "paper_new_selector.round20_repeat5_runner.time.sleep",
                return_value=None,
            ):
                status = run_round20_repeat5_batch(temp_root, repeat_count=5)

            self.assertEqual(status, 1)
            self.assertEqual(completed[0]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(completed[0]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")


if __name__ == "__main__":
    unittest.main()
