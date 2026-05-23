from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

import yaml

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

    def test_builds_e2_extra_unseen_specs_without_external_baselines(self) -> None:
        smoke_specs = e1.build_e1_run_specs("thesis_e2_extra_unseen_smoke")
        repeat15_specs = e1.build_e1_run_specs("thesis_e2_extra_unseen_repeat15")
        self.assertEqual(len(smoke_specs), 6)
        self.assertEqual(len(repeat15_specs), 90)
        self.assertEqual({spec.method for spec in repeat15_specs}, {"pretext", "round19"})
        self.assertEqual(
            {spec.dataset for spec in repeat15_specs},
            {"bioarxiv", "rotten_tomatoes", "twitter_emotion_binary"},
        )
        self.assertEqual({spec.seed for spec in smoke_specs}, {42})

    def test_builds_controller_dev_extra_repeat15_with_external_baselines(self) -> None:
        specs = e1.build_e1_run_specs("thesis_main_controller_dev_extra_repeat15")
        self.assertEqual(len(specs), 120)
        self.assertEqual({spec.method for spec in specs}, {"pretext", "round19", "wasp", "dpga"})
        self.assertEqual({spec.dataset for spec in specs}, {"imdb", "openreview"})
        self.assertEqual({spec.seed for spec in specs}, set(e1.E1_REPEAT15_SEEDS))

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

    def test_materializes_e2_extra_unseen_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated = e1.materialize_e1_configs(root, mode="thesis_e2_extra_unseen_repeat15")
            self.assertEqual(len(generated), 90)
            manifest = root / "configs" / "experiments" / "thesis_e2_extra_unseen_repeat15" / "thesis_e2_extra_unseen_repeat15_manifest.tsv"
            with manifest.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 90)
            self.assertEqual({row["method"] for row in rows}, {"pretext", "round19"})
            self.assertEqual(
                {row["dataset"] for row in rows},
                {"bioarxiv", "rotten_tomatoes", "twitter_emotion_binary"},
            )
            self.assertTrue((root / rows[0]["config_path"]).exists())
            self.assertTrue(rows[0]["output_root"].startswith("paper-new-round19/outputs/thesis_e2_extra_unseen_repeat15/"))

    def test_materializes_controller_dev_extra_repeat15_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated = e1.materialize_e1_configs(root, mode="thesis_main_controller_dev_extra_repeat15")
            self.assertEqual(len(generated), 120)
            manifest = (
                root
                / "configs"
                / "experiments"
                / "thesis_e1_controller_dev_extra_repeat15"
                / "thesis_e1_controller_dev_extra_repeat15_manifest.tsv"
            )
            with manifest.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 120)
            self.assertEqual({row["method"] for row in rows}, {"pretext", "round19", "wasp", "dpga"})
            self.assertEqual({row["dataset"] for row in rows}, {"imdb", "openreview"})
            self.assertTrue(rows[0]["output_root"].startswith("paper-new-round19/outputs/thesis_e1_controller_dev_extra_repeat15/"))

            pretext_config = yaml.safe_load(
                (root / "configs" / "experiments" / "thesis_e1_controller_dev_extra_repeat15" / "pretext_imdb_seed42.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                pretext_config["inherits"],
                [
                    "../single_run_baseline_screening/_base_single_run_expand_private.yaml",
                    "../single_node_tuning_round19/_data_imdb.yaml",
                ],
            )
            wasp_config = yaml.safe_load(
                (root / "configs" / "experiments" / "thesis_e1_controller_dev_extra_repeat15" / "wasp_imdb_seed42.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                wasp_config["inherits"],
                [
                    "../single_run_baseline_screening/_base_single_run_wasp.yaml",
                    "../single_node_tuning_round19/_data_imdb.yaml",
                ],
            )
            dpga_config = yaml.safe_load(
                (root / "configs" / "experiments" / "thesis_e1_controller_dev_extra_repeat15" / "dpga_imdb_seed42.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                dpga_config["inherits"],
                [
                    "../single_run_baseline_screening/_base_single_run_dpga.yaml",
                    "../single_node_tuning_round19/_data_imdb.yaml",
                ],
            )

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

    def test_retry_all_failures_attempts_three_times_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            attempts: list[tuple[str, int]] = []
            original_materialize = e1.materialize_e1_configs
            original_load_manifest = e1.load_manifest
            original_wait = e1.wait_for_vllm_capacity
            original_build_command = e1.build_e1_command
            original_run = e1.subprocess.run
            try:
                specs = [
                    e1.E1BaselineSpec(
                        method="pretext",
                        method_display_name="PrE-Text",
                        dataset="bioarxiv",
                        seed=42,
                        experiment_id="pretext_bioarxiv_seed42",
                        kind="selector_single_node",
                        relative_config_path=Path("configs/experiments/e2/pretext_bioarxiv_seed42.yaml"),
                        relative_output_root=Path("paper-new-round19/outputs/e2/pretext/bioarxiv/seed42"),
                        source_artifact_path=None,
                        implementation_key="expand_private",
                        pretext_template_key="expand_private",
                        mapping_status="needs_verification",
                    ),
                    e1.E1BaselineSpec(
                        method="round19",
                        method_display_name="round19",
                        dataset="rotten_tomatoes",
                        seed=42,
                        experiment_id="round19_rotten_tomatoes_seed42",
                        kind="selector_single_node",
                        relative_config_path=Path("configs/experiments/e2/round19_rotten_tomatoes_seed42.yaml"),
                        relative_output_root=Path("paper-new-round19/outputs/e2/round19/rotten_tomatoes/seed42"),
                        source_artifact_path=None,
                        implementation_key="round19",
                        pretext_template_key="",
                        mapping_status="canonical_round19",
                    ),
                ]
                e1.materialize_e1_configs = lambda project_root, mode="thesis_main_seen_pilot": []
                e1.load_manifest = lambda project_root, mode: specs
                e1.wait_for_vllm_capacity = lambda *args, **kwargs: None
                e1.build_e1_command = lambda spec, config_path: ["fake", spec.experiment_id]
                per_spec_attempts = {"pretext_bioarxiv_seed42": 0, "round19_rotten_tomatoes_seed42": 0}

                def fake_run(command, **kwargs):
                    exp_id = command[-1]
                    per_spec_attempts[exp_id] += 1
                    attempts.append((exp_id, per_spec_attempts[exp_id]))
                    return type(
                        "Completed",
                        (),
                        {
                            "returncode": 1 if exp_id == "pretext_bioarxiv_seed42" else 0,
                            "stdout": "",
                            "stderr": "deterministic non-resource failure"
                            if exp_id == "pretext_bioarxiv_seed42"
                            else "",
                        },
                    )()

                e1.subprocess.run = fake_run
                result = e1.run_e1_batch(
                    root,
                    mode="thesis_e2_extra_unseen_repeat15",
                    dry_run=False,
                    max_attempts=3,
                    retry_delay_seconds=0.0,
                    min_free_gb_for_vllm=2.0,
                    retry_all_failures=True,
                    reset_summary=True,
                )

                self.assertEqual(result["status"], "failed")
                self.assertEqual(
                    attempts,
                    [
                        ("pretext_bioarxiv_seed42", 1),
                        ("pretext_bioarxiv_seed42", 2),
                        ("pretext_bioarxiv_seed42", 3),
                        ("round19_rotten_tomatoes_seed42", 1),
                    ],
                )
            finally:
                e1.materialize_e1_configs = original_materialize
                e1.load_manifest = original_load_manifest
                e1.wait_for_vllm_capacity = original_wait
                e1.build_e1_command = original_build_command
                e1.subprocess.run = original_run


if __name__ == "__main__":
    unittest.main()
