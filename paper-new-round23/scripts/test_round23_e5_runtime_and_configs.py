from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import generate_round23_e5_experiment_configs as e5_config_gen  # noqa: E402
import merge_round23_e5_anchor_results as merge_e5  # noqa: E402
import round23_dynamic_experiment_runner as runner  # noqa: E402


def test_generate_e5_configs_create_expected_counts_and_relative_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e5_config_gen.CONFIG_ROOT
        original_specs = {mode: dict(spec) for mode, spec in e5_config_gen.MODE_SPECS.items()}
        try:
            e5_config_gen.CONFIG_ROOT = root / "configs"
            e5_config_gen.create_base_and_data_stubs()
            e5_config_gen.create_mode_configs("e5_anchor_k19_keepk0_all6_smoke")
            e5_config_gen.create_mode_configs("e5_anchor_k21_round23_all6_repeat5")

            smoke_manifest = (
                e5_config_gen.CONFIG_ROOT
                / "e5_anchor_k19_keepk0_all6_smoke"
                / "round23_e5_anchor_k19_keepk0_all6_smoke_manifest.tsv"
            )
            repeat_manifest = (
                e5_config_gen.CONFIG_ROOT
                / "e5_anchor_k21_round23_all6_repeat5"
                / "round23_e5_anchor_k21_round23_all6_repeat5_manifest.tsv"
            )
            with smoke_manifest.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle, delimiter="\t"))
            with repeat_manifest.open("r", encoding="utf-8", newline="") as handle:
                repeat_rows = list(csv.DictReader(handle, delimiter="\t"))

            assert len(smoke_rows) == 6
            assert len(repeat_rows) == 30
            assert {row["reference_budget"] for row in smoke_rows} == {"19"}
            assert {row["reference_budget"] for row in repeat_rows} == {"21"}
            assert {row["method"] for row in smoke_rows} == {"round23_keepk0"}
            assert {row["method"] for row in repeat_rows} == {"round23"}
            assert all(not Path(row["config_path"]).is_absolute() for row in smoke_rows + repeat_rows)
            assert all(":" not in row["config_path"] for row in smoke_rows + repeat_rows)
        finally:
            e5_config_gen.CONFIG_ROOT = original_root
            e5_config_gen.MODE_SPECS = original_specs


def test_runner_supports_e5_modes_and_manifest_reference_budget():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = root / "manifest.tsv"
        manifest.write_text(
            "\t".join(
                [
                    "experiment_id",
                    "dataset",
                    "seed",
                    "config_path",
                    "output_root",
                    "method",
                    "controller_scope",
                    "controller_bundle",
                    "reference_budget",
                ]
            )
            + "\n"
            + "\t".join(
                [
                    "r23_e5_test",
                    "jobs",
                    "42",
                    "configs/experiments/single_node_tuning_round23_dynamic/e5_anchor_k19_round23_all6_smoke/r23_e5_test.yaml",
                    "outputs/e5_anchor/jobs/seed42",
                    "round23",
                    "all6",
                    "round23_controller_1200_all6_top1_delta_m0005_extratrees_broad_no_dataset",
                    "19",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        specs = runner.load_manifest(manifest)
        assert len(specs) == 1
        assert specs[0].reference_budget == 19
        assert runner.resolve_mode_paths("e5_anchor_k21_keepk0_all6_repeat5")["dataset_split"] == "all6"


def test_run_single_experiment_passes_reference_budget_to_wrapper():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = runner.ExperimentSpec(
            experiment_id="r23_e5_budget",
            dataset_name="jobs",
            meta_seed=42,
            config_path=root / "cfg.yaml",
            output_root="outputs/test/jobs/seed42",
            method="round23_keepk0",
            reference_budget=21,
        )
        captured: dict[str, object] = {}
        original_run = runner.subprocess.run
        original_root = runner.ROUND23_ROOT

        class DummyResult:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            captured["command"] = command
            return DummyResult()

        try:
            runner.subprocess.run = fake_run
            runner.ROUND23_ROOT = root
            code, _, _, _ = runner.run_single_experiment(
                spec,
                model_dir=None,
                timeout_seconds=10,
                log_dir=root,
            )
        finally:
            runner.subprocess.run = original_run
            runner.ROUND23_ROOT = original_root

        assert code == 0
        command = captured["command"]
        assert "--reference-budget" in command
        idx = command.index("--reference-budget")
        assert command[idx + 1] == "21"


def test_merge_e5_anchor_results_builds_gain_vs_keepk0_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        summary = root / "summary.tsv"
        rows = [
            {
                "method": "round23_keepk0",
                "dataset_name": "jobs",
                "status": "success",
                "reference_budget": "19",
                "best_top1": "0.20",
                "best_top3": "0.30",
                "best_top5": "0.40",
                "best_top10": "0.50",
            },
            {
                "method": "round23",
                "dataset_name": "jobs",
                "status": "success",
                "reference_budget": "19",
                "best_top1": "0.25",
                "best_top3": "0.35",
                "best_top5": "0.45",
                "best_top10": "0.55",
            },
        ]
        with summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        outputs = merge_e5.merge_results(
            summary_paths=[summary],
            output_dir=root / "out",
            output_prefix="e5_unit",
        )
        seen4_rows = list(csv.DictReader(outputs["seen4_table"].open("r", encoding="utf-8"), delimiter="\t"))
        round23_k19 = next(row for row in seen4_rows if row["anchor"] == "19" and row["method"] == "round23")
        keepk0_k19 = next(row for row in seen4_rows if row["anchor"] == "19" and row["method"] == "round23_keepk0")
        assert keepk0_k19["Avg."] == "0.200000"
        assert round23_k19["Avg."] == "0.250000"
        assert round23_k19["Gain vs keep-k0"] == "0.050000"


def test_e5_sequential_scripts_reset_summary_and_pin_gpu_index():
    for path in (
        Path(__file__).parent / "run_round23_e5_anchor_smoke24_sequential.sh",
        Path(__file__).parent / "run_round23_e5_anchor_repeat5_120_sequential.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert "RESET_SUMMARY" in text
        assert "--reset-summary" in text
        assert "--target-gpu-index" in text
        assert "--retry-delay-seconds 10" in text
