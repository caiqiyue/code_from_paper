"""Targeted tests for round23 config generation and batch runner."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import generate_round23_experiment_configs as config_gen  # noqa: E402
import round23_dynamic_experiment_runner as runner  # noqa: E402


def test_generate_configs_creates_unseen_manifest_and_six_dataset_stubs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = config_gen.CONFIG_ROOT
        original_base = config_gen.BASE_FILE
        original_mode_specs = {
            mode: dict(spec)
            for mode, spec in config_gen.MODE_SPECS.items()
        }
        try:
            config_gen.CONFIG_ROOT = root / "configs"
            config_gen.BASE_FILE = config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            config_gen.MODE_SPECS["quick_compare"]["seeds"] = [42, 3141]
            config_gen.MODE_SPECS["unseen_dataset_final_eval"]["seeds"] = [42, 3141]
            config_gen.MODE_SPECS["thesis_main_seen_smoke"]["seeds"] = [42]
            config_gen.MODE_SPECS["thesis_main_seen_pilot"]["seeds"] = [42, 123]
            config_gen.MODE_SPECS["thesis_main_seen_repeat10"]["seeds"] = [42, 123]
            config_gen.MODE_SPECS["thesis_main_seen_repeat15"]["seeds"] = [42, 123]
            config_gen.MODE_SPECS["thesis_main_seen_repeat30"]["seeds"] = [42, 123]
            config_gen.create_base_and_data_stubs()
            config_gen.create_real_smoke()
            config_gen.create_quick_compare()
            config_gen.create_unseen_dataset_final_eval()
            config_gen.create_mode_configs("thesis_main_seen_smoke")
            config_gen.create_mode_configs("thesis_main_seen_pilot")

            assert (config_gen.CONFIG_ROOT / "_data_imdb.yaml").exists()
            assert (config_gen.CONFIG_ROOT / "_data_openreview.yaml").exists()

            manifest_path = (
                config_gen.CONFIG_ROOT
                / "unseen_dataset_final_eval_repeat40"
                / "round23_unseen_dataset_final_eval_repeat40_manifest.tsv"
            )
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

            assert len(rows) == 4
            assert {row["dataset"] for row in rows} == {"imdb", "openreview"}
            assert all(
                "outputs/unseen_dataset_final_eval_repeat40/" in row["output_root"]
                for row in rows
            )

            pilot_manifest = (
                config_gen.CONFIG_ROOT
                / "thesis_main_seen_pilot"
                / "round23_thesis_main_seen_pilot_manifest.tsv"
            )
            with pilot_manifest.open("r", encoding="utf-8", newline="") as handle:
                pilot_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(pilot_rows) == 8
            assert pilot_rows[0]["method"] == "round23"
            assert pilot_rows[0]["controller_scope"] == "all6"
            assert pilot_rows[0]["controller_bundle"] == (
                "round23_controller_1200_all6_top1_delta_m0005_extratrees_broad_no_dataset"
            )

            smoke_manifest = (
                config_gen.CONFIG_ROOT
                / "thesis_main_seen_smoke"
                / "round23_thesis_main_seen_smoke_manifest.tsv"
            )
            with smoke_manifest.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(smoke_rows) == 4
            assert {row["seed"] for row in smoke_rows} == {"42"}
            assert not Path(smoke_rows[0]["config_path"]).is_absolute()
            assert smoke_rows[0]["config_path"].startswith(
                "configs/experiments/single_node_tuning_round23_dynamic/"
            )
        finally:
            config_gen.CONFIG_ROOT = original_root
            config_gen.BASE_FILE = original_base
            config_gen.MODE_SPECS = original_mode_specs


def test_resolve_mode_paths_supports_unseen_dataset_final_eval():
    paths = runner.resolve_mode_paths("unseen_dataset_final_eval")
    assert paths["manifest_relpath"] == (
        "unseen_dataset_final_eval_repeat40/"
        "round23_unseen_dataset_final_eval_repeat40_manifest.tsv"
    )
    assert paths["log_stem"] == "round23_unseen_dataset_final_eval_repeat40"


def test_resolve_mode_paths_supports_thesis_main_seen_pilot():
    paths = runner.resolve_mode_paths("thesis_main_seen_pilot")
    assert paths["manifest_relpath"] == (
        "thesis_main_seen_pilot/"
        "round23_thesis_main_seen_pilot_manifest.tsv"
    )
    assert paths["log_stem"] == "round23_thesis_main_seen_pilot"
    assert paths["dataset_split"] == "seen"


def test_resolve_mode_paths_supports_thesis_main_seen_smoke_and_repeat15():
    smoke_paths = runner.resolve_mode_paths("thesis_main_seen_smoke")
    assert smoke_paths["manifest_relpath"] == (
        "thesis_main_seen_smoke/"
        "round23_thesis_main_seen_smoke_manifest.tsv"
    )
    assert smoke_paths["log_stem"] == "round23_thesis_main_seen_smoke"

    repeat15_paths = runner.resolve_mode_paths("thesis_main_seen_repeat15")
    assert repeat15_paths["manifest_relpath"] == (
        "thesis_main_seen_repeat15/"
        "round23_thesis_main_seen_repeat15_manifest.tsv"
    )
    assert repeat15_paths["log_stem"] == "round23_thesis_main_seen_repeat15"


def test_generate_configs_creates_e2_extra_unseen_smoke_and_repeat15():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = config_gen.CONFIG_ROOT
        original_base = config_gen.BASE_FILE
        original_mode_specs = {
            mode: dict(spec)
            for mode, spec in config_gen.MODE_SPECS.items()
        }
        try:
            config_gen.CONFIG_ROOT = root / "configs"
            config_gen.BASE_FILE = config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            config_gen.create_base_and_data_stubs()
            config_gen.create_mode_configs("thesis_e2_extra_unseen_smoke")
            config_gen.create_mode_configs("thesis_e2_extra_unseen_repeat15")

            for dataset in ("bioarxiv", "rotten_tomatoes", "twitter_emotion_binary"):
                assert (config_gen.CONFIG_ROOT / f"_data_{dataset}.yaml").exists()

            smoke_manifest = (
                config_gen.CONFIG_ROOT
                / "thesis_e2_extra_unseen_smoke"
                / "round23_thesis_e2_extra_unseen_smoke_manifest.tsv"
            )
            with smoke_manifest.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(smoke_rows) == 3
            assert {row["dataset"] for row in smoke_rows} == {
                "bioarxiv",
                "rotten_tomatoes",
                "twitter_emotion_binary",
            }
            assert {row["seed"] for row in smoke_rows} == {"42"}
            assert all(row["controller_scope"] == "all6" for row in smoke_rows)

            repeat15_manifest = (
                config_gen.CONFIG_ROOT
                / "thesis_e2_extra_unseen_repeat15"
                / "round23_thesis_e2_extra_unseen_repeat15_manifest.tsv"
            )
            with repeat15_manifest.open("r", encoding="utf-8", newline="") as handle:
                repeat15_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(repeat15_rows) == 45
            assert {row["dataset"] for row in repeat15_rows} == {
                "bioarxiv",
                "rotten_tomatoes",
                "twitter_emotion_binary",
            }
            assert all(
                "outputs/thesis_e2_extra_unseen_repeat15/" in row["output_root"]
                for row in repeat15_rows
            )
        finally:
            config_gen.CONFIG_ROOT = original_root
            config_gen.BASE_FILE = original_base
            config_gen.MODE_SPECS = original_mode_specs


def test_resolve_mode_paths_supports_e2_extra_unseen_smoke_and_repeat15():
    smoke_paths = runner.resolve_mode_paths("thesis_e2_extra_unseen_smoke")
    assert smoke_paths["manifest_relpath"] == (
        "thesis_e2_extra_unseen_smoke/"
        "round23_thesis_e2_extra_unseen_smoke_manifest.tsv"
    )
    assert smoke_paths["log_stem"] == "round23_thesis_e2_extra_unseen_smoke"
    assert smoke_paths["dataset_split"] == "extra_unseen"

    repeat15_paths = runner.resolve_mode_paths("thesis_e2_extra_unseen_repeat15")
    assert repeat15_paths["manifest_relpath"] == (
        "thesis_e2_extra_unseen_repeat15/"
        "round23_thesis_e2_extra_unseen_repeat15_manifest.tsv"
    )
    assert repeat15_paths["log_stem"] == "round23_thesis_e2_extra_unseen_repeat15"
    assert repeat15_paths["dataset_split"] == "extra_unseen"


def test_summary_row_reads_nested_metrics_and_controller_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = runner.ExperimentSpec(
            experiment_id="r23_test",
            dataset_name="jobs",
            meta_seed=42,
            config_path=root / "r23_test.yaml",
            output_root="outputs/thesis_main_seen_pilot/jobs/seed42",
        )
        sidecar_root = root / runner.normalize_output_root(spec.output_root)
        sidecar_root.mkdir(parents=True, exist_ok=True)
        (sidecar_root / "r23_test_dynamic_controller_runtime.json").write_text(
            json.dumps(
                {
                    "controller_scope": "all6",
                    "bundle_version": "bundle-v1",
                    "learner_family": "extratrees",
                    "feature_version": "no_dataset",
                    "target_mode": "top1_delta",
                    "target_field": "target_value_for_training",
                    "reference_budget": 20,
                    "predicted_delta_k": -1,
                    "predicted_target_budget": 19,
                    "runtime_artifacts": {
                        "eval_summary": {
                            "metrics": {
                                "best_top1": 0.11,
                                "best_top3": 0.22,
                                "best_top5": 0.33,
                                "best_top10": 0.44,
                            }
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        original_root = runner.ROUND23_ROOT
        try:
            runner.ROUND23_ROOT = root
            row = runner.build_summary_row(
                spec,
                mode="thesis_main_seen_pilot",
                dataset_split="seen",
                status="success",
                attempt=1,
                duration_seconds=0.1,
            )
        finally:
            runner.ROUND23_ROOT = original_root

        assert row["controller_scope"] == "all6"
        assert row["bundle_version"] == "bundle-v1"
        assert row["learner_family"] == "extratrees"
        assert row["feature_version"] == "no_dataset"
        assert row["target_mode"] == "top1_delta"
        assert row["target_field"] == "target_value_for_training"
        assert row["reference_budget"] == 20
        assert row["best_top1"] == 0.11
        assert row["best_top3"] == 0.22
        assert row["best_top5"] == 0.33
        assert row["best_top10"] == 0.44


def test_runner_retries_failed_item_without_stopping_following_items():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        specs = [
            runner.ExperimentSpec(
                experiment_id="exp_fail",
                dataset_name="jobs",
                meta_seed=42,
                config_path=root / "exp_fail.yaml",
                output_root="outputs/real_smoke/jobs/seed42",
            ),
            runner.ExperimentSpec(
                experiment_id="exp_success",
                dataset_name="imdb",
                meta_seed=3141,
                config_path=root / "exp_success.yaml",
                output_root="outputs/unseen_dataset_final_eval_repeat40/imdb/seed3141",
            ),
        ]
        attempts: list[tuple[str, int]] = []
        model_dir = root / "bundle"
        model_dir.mkdir()

        original_root = runner.ROUND23_ROOT
        original_load_manifest = runner.load_manifest
        original_run_single_experiment = runner.run_single_experiment
        original_parse_args = runner.parse_args
        original_wait_for_vllm_capacity = runner.wait_for_vllm_capacity
        try:
            runner.ROUND23_ROOT = root
            runner.load_manifest = lambda manifest_path: specs

            per_spec_attempts = {"exp_fail": 0, "exp_success": 0}

            def fake_run_single_experiment(spec, *, model_dir, timeout_seconds, log_dir):
                per_spec_attempts[spec.experiment_id] += 1
                attempts.append((spec.experiment_id, per_spec_attempts[spec.experiment_id]))
                if spec.experiment_id == "exp_fail":
                    return 1, "", "No available memory for the cache blocks", 0.01
                sidecar_root = root / runner.normalize_output_root(spec.output_root)
                sidecar_root.mkdir(parents=True, exist_ok=True)
                sidecar_path = sidecar_root / f"{spec.experiment_id}_dynamic_controller_runtime.json"
                sidecar_path.write_text(
                    json.dumps(
                        {
                            "predicted_delta_k": 1,
                            "predicted_target_budget": 21,
                            "runtime_artifacts": {
                                "eval_summary": {
                                    "best_top1": 0.55,
                                }
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return 0, "", "", 0.02

            runner.run_single_experiment = fake_run_single_experiment
            runner.parse_args = lambda: argparse.Namespace(
                mode="unseen_dataset_final_eval",
                model_dir=str(model_dir),
                timeout_seconds=1,
                max_attempts=2,
                retry_delay_seconds=0.0,
                retry_all_failures=False,
                target_gpu_name_token="RTX A6000",
                min_free_gb_for_vllm=0.0,
                gpu_wait_poll_seconds=0.0,
                gpu_wait_timeout_seconds=1.0,
                reset_summary=False,
                dry_run=False,
                limit=0,
            )
            runner.wait_for_vllm_capacity = lambda *args, **kwargs: None

            exit_code = runner.main()

            assert exit_code == 1
            assert attempts == [
                ("exp_fail", 1),
                ("exp_fail", 2),
                ("exp_success", 1),
            ]

            summary_jsonl = root / "logs" / "round23_unseen_dataset_final_eval_repeat40_summary.jsonl"
            rows = [
                json.loads(line)
                for line in summary_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert [row["experiment_id"] for row in rows] == [
                "exp_fail",
                "exp_fail",
                "exp_success",
            ]
            assert rows[-1]["status"] == "success"
        finally:
            runner.ROUND23_ROOT = original_root
            runner.load_manifest = original_load_manifest
            runner.run_single_experiment = original_run_single_experiment
            runner.parse_args = original_parse_args
            runner.wait_for_vllm_capacity = original_wait_for_vllm_capacity


def test_runner_retry_all_failures_attempts_three_times_and_continues():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        specs = [
            runner.ExperimentSpec(
                experiment_id="exp_nonretry_fail",
                dataset_name="bioarxiv",
                meta_seed=42,
                config_path=root / "exp_nonretry_fail.yaml",
                output_root="outputs/thesis_e2_extra_unseen_repeat15/bioarxiv/seed42",
            ),
            runner.ExperimentSpec(
                experiment_id="exp_after_failure",
                dataset_name="rotten_tomatoes",
                meta_seed=42,
                config_path=root / "exp_after_failure.yaml",
                output_root="outputs/thesis_e2_extra_unseen_repeat15/rotten_tomatoes/seed42",
            ),
        ]
        attempts: list[tuple[str, int]] = []
        model_dir = root / "bundle"
        model_dir.mkdir()

        original_root = runner.ROUND23_ROOT
        original_load_manifest = runner.load_manifest
        original_run_single_experiment = runner.run_single_experiment
        original_parse_args = runner.parse_args
        original_wait_for_vllm_capacity = runner.wait_for_vllm_capacity
        try:
            runner.ROUND23_ROOT = root
            runner.load_manifest = lambda manifest_path: specs
            per_spec_attempts = {"exp_nonretry_fail": 0, "exp_after_failure": 0}

            def fake_run_single_experiment(spec, *, model_dir, timeout_seconds, log_dir):
                per_spec_attempts[spec.experiment_id] += 1
                attempts.append((spec.experiment_id, per_spec_attempts[spec.experiment_id]))
                if spec.experiment_id == "exp_nonretry_fail":
                    return 1, "", "deterministic non-resource failure", 0.01
                sidecar_root = root / runner.normalize_output_root(spec.output_root)
                sidecar_root.mkdir(parents=True, exist_ok=True)
                (sidecar_root / f"{spec.experiment_id}_dynamic_controller_runtime.json").write_text(
                    json.dumps(
                        {
                            "predicted_delta_k": 0,
                            "predicted_target_budget": 20,
                            "runtime_artifacts": {"eval_summary": {"best_top1": 0.5}},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return 0, "", "", 0.01

            runner.run_single_experiment = fake_run_single_experiment
            runner.parse_args = lambda: argparse.Namespace(
                mode="thesis_e2_extra_unseen_repeat15",
                model_dir=str(model_dir),
                timeout_seconds=1,
                max_attempts=3,
                retry_delay_seconds=0.0,
                retry_all_failures=True,
                target_gpu_name_token="RTX A6000",
                min_free_gb_for_vllm=2.0,
                gpu_wait_poll_seconds=0.0,
                gpu_wait_timeout_seconds=1.0,
                reset_summary=True,
                dry_run=False,
                limit=0,
            )
            runner.wait_for_vllm_capacity = lambda *args, **kwargs: None

            exit_code = runner.main()

            assert exit_code == 1
            assert attempts == [
                ("exp_nonretry_fail", 1),
                ("exp_nonretry_fail", 2),
                ("exp_nonretry_fail", 3),
                ("exp_after_failure", 1),
            ]
        finally:
            runner.ROUND23_ROOT = original_root
            runner.load_manifest = original_load_manifest
            runner.run_single_experiment = original_run_single_experiment
            runner.parse_args = original_parse_args
            runner.wait_for_vllm_capacity = original_wait_for_vllm_capacity


if __name__ == "__main__":
    tests = [
        ("config_generation", test_generate_configs_creates_unseen_manifest_and_six_dataset_stubs),
        ("resolve_mode_paths", test_resolve_mode_paths_supports_unseen_dataset_final_eval),
        ("resolve_thesis_mode_paths", test_resolve_mode_paths_supports_thesis_main_seen_pilot),
        ("resolve_thesis_smoke_repeat15_paths", test_resolve_mode_paths_supports_thesis_main_seen_smoke_and_repeat15),
        ("summary_nested_metrics", test_summary_row_reads_nested_metrics_and_controller_metadata),
        ("runner_continuation", test_runner_retries_failed_item_without_stopping_following_items),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as exc:
            failures += 1
            print(f"  {name}: FAILED - {exc}")
    if failures:
        raise SystemExit(1)
    print("\nALL ROUND23 CONFIG/RUNNER TESTS PASSED")
