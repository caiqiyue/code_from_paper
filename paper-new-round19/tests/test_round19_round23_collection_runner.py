import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_new_selector.round19_round23_collection_runner import (
    COLLECTION_SUMMARY_HEADER,
    Round19Round23CollectionSpec,
    append_round19_round23_collection_summary_row,
    build_round19_round23_collection_specs,
    normalize_collection_output_root,
    resolve_round19_round23_collection_project_root,
    run_round19_round23_collection_batch,
    validate_collection_output_dir,
)


class Round19Round23CollectionRunnerTests(unittest.TestCase):
    def test_resolve_project_root_anchors_to_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "paper-new-round19"
            module_path = repo_root / "paper_new_selector" / "round19_round23_collection_runner.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_text("# test", encoding="utf-8")
            self.assertEqual(resolve_round19_round23_collection_project_root(module_path), repo_root.resolve())

    def test_build_specs_reads_1200_specs_from_manifest(self):
        specs = build_round19_round23_collection_specs(
            project_root=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(len(specs), 1200)
        self.assertEqual(specs[0].experiment_id, "r19_r23c_jobs_seed42_k18")
        self.assertEqual(specs[0].budget_k, 18)
        self.assertEqual(specs[-1].dataset_name, "openreview")
        self.assertEqual(specs[-1].budget_k, 22)

    def test_normalize_output_root_strips_repo_name_prefix(self):
        root = Path("D:/tmp/paper-new-round19").resolve()
        normalized = normalize_collection_output_root(root, "paper-new-round19/outputs/round23_collection_repeat40/jobs/seed42/k18")
        self.assertEqual(
            normalized,
            root / "outputs" / "round23_collection_repeat40" / "jobs" / "seed42" / "k18",
        )

    def test_validate_collection_output_dir_requires_metrics_and_nonempty_budget_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir) / "exp"
            (outdir / "collection").mkdir(parents=True)
            (outdir / "eval").mkdir(parents=True)
            (outdir / "collection" / "context_summary.json").write_text(
                json.dumps({"dataset_name": "jobs", "meta_seed": 42}),
                encoding="utf-8",
            )
            (outdir / "collection" / "budget_table.jsonl").write_text(
                json.dumps({"budget_k": 18}) + "\n",
                encoding="utf-8",
            )
            metrics = {
                "best_top1": 0.27,
                "best_top3": 0.42,
                "best_top5": 0.49,
                "best_top10": 0.57,
            }
            (outdir / "collection" / "final_result_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
            (outdir / "eval" / "downstream_eval_summary.json").write_text(
                json.dumps({"metrics": metrics}),
                encoding="utf-8",
            )
            code, status, returned_metrics = validate_collection_output_dir(outdir)
            self.assertEqual(code, 0)
            self.assertEqual(status, "completed")
            self.assertEqual(returned_metrics["best_top1"], 0.27)

    def test_append_summary_row_records_attempts_status_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "summary.tsv"
            summary.write_text("\t".join(COLLECTION_SUMMARY_HEADER) + "\n", encoding="utf-8")
            spec = Round19Round23CollectionSpec(
                experiment_id="exp1",
                dataset_name="jobs",
                meta_seed=42,
                budget_k=18,
                config_path=Path("cfg.yaml"),
                output_root=Path("out"),
                group_name="g",
            )
            append_round19_round23_collection_summary_row(
                summary,
                spec,
                attempts=3,
                status="failed",
                returncode=1,
                error_class="retryable_vllm_cache",
                metrics={"best_top1": 0.1, "best_top3": 0.2, "best_top5": 0.3, "best_top10": 0.4},
            )
            rows = summary.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(
                rows[1],
                "exp1\tjobs\t42\t18\t3\tfailed\t1\tretryable_vllm_cache\t0.1\t0.2\t0.3\t0.4",
            )

    def test_run_batch_retries_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "paper-new-round19"
            (root / "logs").mkdir(parents=True)
            spec_fail = Round19Round23CollectionSpec(
                experiment_id="exp_fail",
                dataset_name="jobs",
                meta_seed=42,
                budget_k=22,
                config_path=root / "cfg_fail.yaml",
                output_root=root / "outputs" / "exp_fail",
                group_name="g",
            )
            spec_ok = Round19Round23CollectionSpec(
                experiment_id="exp_ok",
                dataset_name="jobs",
                meta_seed=43,
                budget_k=18,
                config_path=root / "cfg_ok.yaml",
                output_root=root / "outputs" / "exp_ok",
                group_name="g",
            )
            spec_fail.config_path.write_text("x", encoding="utf-8")
            spec_ok.config_path.write_text("x", encoding="utf-8")

            call_counter = {"exp_fail": 0, "exp_ok": 0}

            def fake_run(cmd, cwd, stdout, stderr, check, env):
                exp = "exp_fail" if "cfg_fail.yaml" in cmd[-1] else "exp_ok"
                call_counter[exp] += 1
                if exp == "exp_fail":
                    stdout.write("# GPU blocks: 0\nNo available memory for the cache blocks\n")
                    return type("Completed", (), {"returncode": 1})()
                stdout.write("ok\n")
                return type("Completed", (), {"returncode": 0})()

            def fake_validate(output_dir):
                if "exp_ok" in str(output_dir):
                    return 0, "completed", {
                        "best_top1": 0.27,
                        "best_top3": 0.42,
                        "best_top5": 0.49,
                        "best_top10": 0.57,
                    }
                return 91, "missing_required_artifact", {}

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False), patch(
                "paper_new_selector.round19_round23_collection_runner.build_round19_round23_collection_specs",
                return_value=[spec_fail, spec_ok],
            ), patch(
                "paper_new_selector.round19_round23_collection_runner.wait_for_collection_vllm_capacity",
                return_value=None,
            ), patch(
                "paper_new_selector.round19_round23_collection_runner.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "paper_new_selector.round19_round23_collection_runner.validate_collection_output_dir",
                side_effect=fake_validate,
            ), patch(
                "paper_new_selector.round19_round23_collection_runner.time.sleep",
                return_value=None,
            ):
                status = run_round19_round23_collection_batch(root)

            self.assertEqual(status, 1)
            self.assertEqual(call_counter["exp_fail"], 3)
            self.assertEqual(call_counter["exp_ok"], 1)
            summary = (root / "logs" / "round19_round23_collection_repeat40_summary.tsv").read_text(encoding="utf-8")
            self.assertIn("exp_fail\tjobs\t42\t22\t3\tfailed\t1\tretryable_vllm_cache", summary)
            self.assertIn("exp_ok\tjobs\t43\t18\t1\tcompleted\t0\tcompleted\t0.27", summary)


if __name__ == "__main__":
    unittest.main()
