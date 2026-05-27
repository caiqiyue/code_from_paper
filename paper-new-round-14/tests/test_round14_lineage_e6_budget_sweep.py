from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_round14_lineage_e6_budget_sweep_configs as e6_config_gen  # noqa: E402
import run_round14_lineage_manifest as e6_runner  # noqa: E402


def test_generate_e6_configs_create_expected_counts_and_output_roots():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e6_config_gen.CONFIG_ROOT
        try:
            e6_config_gen.CONFIG_ROOT = root / "configs"
            e6_config_gen.create_mode_configs("round14_lineage_e6_smoke90")
            e6_config_gen.create_mode_configs("round14_lineage_e6_repeat2_180")

            smoke_manifest = (
                e6_config_gen.CONFIG_ROOT
                / "smoke90"
                / "round14_lineage_e6_smoke90_manifest.tsv"
            )
            repeat_manifest = (
                e6_config_gen.CONFIG_ROOT
                / "repeat2_180"
                / "round14_lineage_e6_repeat2_180_manifest.tsv"
            )
            with smoke_manifest.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle, delimiter="\t"))
            with repeat_manifest.open("r", encoding="utf-8", newline="") as handle:
                repeat_rows = list(csv.DictReader(handle, delimiter="\t"))

            expected_datasets = {"jobs", "congressional", "forums", "microblog", "imdb", "openreview"}
            expected_budgets = {6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62}

            assert len(smoke_rows) == 90
            assert len(repeat_rows) == 180
            assert {row["dataset"] for row in smoke_rows} == expected_datasets
            assert {int(row["budget"]) for row in smoke_rows} == expected_budgets
            assert {int(row["seed"]) for row in smoke_rows} == {42}
            assert {int(row["seed"]) for row in repeat_rows} == {42, 123}
            assert all(not Path(row["config_path"]).is_absolute() for row in smoke_rows + repeat_rows)
            assert all(":" not in row["config_path"] for row in smoke_rows + repeat_rows)
            assert all(
                row["output_root"].startswith("paper-new-round-14/outputs/round14_lineage_e6_budget_sweep")
                for row in smoke_rows + repeat_rows
            )
        finally:
            e6_config_gen.CONFIG_ROOT = original_root


def test_generated_base_config_uses_formal_vllm_and_large_candidate_pool():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e6_config_gen.CONFIG_ROOT
        try:
            e6_config_gen.CONFIG_ROOT = root / "configs"
            base_path = e6_config_gen.create_base_config()
            text = base_path.read_text(encoding="utf-8")
        finally:
            e6_config_gen.CONFIG_ROOT = original_root

    assert "candidate_count: 96" in text
    assert "generated_per_round: 24" in text
    assert "max_rounds: 8" in text
    assert "gpu_memory_utilization: 0.55" in text
    assert "startup_required_free_gb: 26" in text


def test_run_single_experiment_extracts_eval_metrics_from_success_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = e6_runner.ExperimentSpec(
            experiment_id="e6_jobs_k10_seed42",
            dataset_name="jobs",
            budget=10,
            meta_seed=42,
            config_path=root / "cfg.yaml",
            output_root="paper-new-round-14/outputs/round14_lineage_e6_budget_sweep/jobs/k10/seed42",
        )
        captured: dict[str, object] = {}
        original_run = e6_runner.subprocess.run

        class DummyResult:
            returncode = 0
            stderr = ""
            stdout = (
                "{\n"
                '  "eval": {\n'
                '    "status": "completed",\n'
                '    "metrics": {\n'
                '      "best_top1": 0.31,\n'
                '      "best_top3": 0.42,\n'
                '      "best_top5": 0.53,\n'
                '      "best_top10": 0.64\n'
                "    }\n"
                "  }\n"
                "}\n"
            )

        def fake_run(command, **kwargs):
            captured["command"] = command
            return DummyResult()

        try:
            e6_runner.subprocess.run = fake_run
            result = e6_runner.run_single_experiment(
                spec,
                python_executable="python",
                timeout_seconds=30,
            )
        finally:
            e6_runner.subprocess.run = original_run

        assert result.returncode == 0
        assert result.metrics["best_top1"] == 0.31
        assert result.metrics["best_top10"] == 0.64
        command = captured["command"]
        assert "--config" in command
        assert str(spec.config_path) in command


def test_sequential_scripts_pin_retry_policy():
    for path in (
        ROOT / "scripts" / "run_round14_lineage_e6_budget_sweep_smoke90_sequential.sh",
        ROOT / "scripts" / "run_round14_lineage_e6_budget_sweep_repeat2_180_sequential.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert "--max-attempts 3" in text
        assert "--retry-delay-seconds 10" in text
        assert "--reset-summary" in text
        assert "TARGET_GPU_INDEX" in text
        assert "--min-free-gb-for-vllm 26" in text
        assert "--gpu-wait-poll-seconds 60" in text
        assert "--gpu-wait-timeout-seconds 43200" in text
        assert "--target-gpu-index" in text
