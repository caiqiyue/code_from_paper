from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import generate_round23_experiment_configs as main_config_gen  # noqa: E402
import generate_round23_e4_experiment_configs as e4_config_gen  # noqa: E402
import round23_dynamic_experiment_runner as runner  # noqa: E402
from run_round23_with_absolute_k_controller import generate_override_config as generate_absk_override_config  # noqa: E402
from round23_runtime_utils import load_yaml_with_inherits  # noqa: E402


def test_generate_absolute_k_override_config_pins_predicted_budget_and_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "meta:",
                    "  experiment_id: r23_e4_absk_test",
                    "selector:",
                    "  seed_top_k: 20",
                    "paths:",
                    "  output_root: ./outputs/original",
                ]
            ),
            encoding="utf-8",
        )
        override_path, experiment_id = generate_absk_override_config(
            original_config_path=config_path,
            predicted_absolute_k=21,
            predicted_delta_k=1,
            model_dir=root / "bundle",
            output_root=root / "runtime",
        )
        payload = json.loads(
            json.dumps(__import__("yaml").safe_load(override_path.read_text(encoding="utf-8")))
        )
        assert experiment_id == "r23_e4_absk_test"
        assert int(payload["selector"]["seed_top_k"]) == 21
        assert int(payload["meta"]["absolute_k_runtime"]["predicted_absolute_k"]) == 21
        assert int(payload["meta"]["absolute_k_runtime"]["predicted_delta_k"]) == 1


def test_generate_e4_configs_creates_all6_manifests_with_relative_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e4_config_gen.CONFIG_ROOT
        original_base = e4_config_gen.BASE_FILE
        original_mode_specs = {mode: dict(spec) for mode, spec in e4_config_gen.MODE_SPECS.items()}
        try:
            e4_config_gen.CONFIG_ROOT = root / "configs"
            e4_config_gen.BASE_FILE = e4_config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            e4_config_gen.MODE_SPECS["e4_a_oneshot_all6_smoke"]["seeds"] = [42]
            e4_config_gen.MODE_SPECS["e4_a_oneshot_all6_repeat15"]["seeds"] = [42, 123]
            e4_config_gen.create_base_and_data_stubs()
            e4_config_gen.create_mode_configs("e4_a_oneshot_all6_smoke")
            e4_config_gen.create_mode_configs("e4_a_oneshot_all6_repeat15")

            smoke_manifest = (
                e4_config_gen.CONFIG_ROOT
                / "e4_a_oneshot_all6_smoke"
                / "round23_e4_a_oneshot_all6_smoke_manifest.tsv"
            )
            with smoke_manifest.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(smoke_rows) == 6
            assert {row["method"] for row in smoke_rows} == {"round23_absk_oneshot"}
            assert {row["dataset"] for row in smoke_rows} == {
                "jobs",
                "congressional",
                "forums",
                "microblog",
                "imdb",
                "openreview",
            }
            assert not Path(smoke_rows[0]["config_path"]).is_absolute()

            repeat_manifest = (
                e4_config_gen.CONFIG_ROOT
                / "e4_a_oneshot_all6_repeat15"
                / "round23_e4_a_oneshot_all6_repeat15_manifest.tsv"
            )
            with repeat_manifest.open("r", encoding="utf-8", newline="") as handle:
                repeat_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(repeat_rows) == 12
            assert all(
                row["output_root"].startswith("outputs/e4_a_oneshot_all6_repeat15/")
                for row in repeat_rows
            )
        finally:
            e4_config_gen.CONFIG_ROOT = original_root
            e4_config_gen.BASE_FILE = original_base
            e4_config_gen.MODE_SPECS = original_mode_specs


def test_generate_e4_configs_marks_three_round_manifest_as_nonformal():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e4_config_gen.CONFIG_ROOT
        original_base = e4_config_gen.BASE_FILE
        original_mode_specs = {mode: dict(spec) for mode, spec in e4_config_gen.MODE_SPECS.items()}
        try:
            e4_config_gen.CONFIG_ROOT = root / "configs"
            e4_config_gen.BASE_FILE = e4_config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            e4_config_gen.MODE_SPECS["e4_c_three_round_stress_all6_repeat15"]["seeds"] = [42]
            e4_config_gen.create_base_and_data_stubs()
            e4_config_gen.create_mode_configs("e4_c_three_round_stress_all6_repeat15")

            manifest = (
                e4_config_gen.CONFIG_ROOT
                / "e4_c_three_round_stress_all6_repeat15"
                / "round23_e4_c_three_round_stress_all6_repeat15_manifest.tsv"
            )
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            assert rows
            assert rows[0]["method"] == "round23_3round_stress"
            assert "non-formal" in rows[0]["note"].lower()
        finally:
            e4_config_gen.CONFIG_ROOT = original_root
            e4_config_gen.BASE_FILE = original_base
            e4_config_gen.MODE_SPECS = original_mode_specs


def test_generate_e4_main_defaults_to_all6_formal_modes_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e4_config_gen.CONFIG_ROOT
        original_base = e4_config_gen.BASE_FILE
        original_argv = sys.argv[:]
        try:
            e4_config_gen.CONFIG_ROOT = root / "configs"
            e4_config_gen.BASE_FILE = e4_config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            sys.argv = ["generate_round23_e4_experiment_configs.py"]
            exit_code = e4_config_gen.main()
            assert exit_code == 0
            assert (e4_config_gen.CONFIG_ROOT / "e4_a_oneshot_all6_smoke").exists()
            assert (e4_config_gen.CONFIG_ROOT / "e4_c_three_round_stress_all6_repeat15").exists()
            assert not (e4_config_gen.CONFIG_ROOT / "e4_a_oneshot_seen_smoke").exists()
            assert not (e4_config_gen.CONFIG_ROOT / "e4_c_three_round_stress_pilot").exists()
        finally:
            e4_config_gen.CONFIG_ROOT = original_root
            e4_config_gen.BASE_FILE = original_base
            sys.argv = original_argv


def test_generated_dynamic_base_inherits_formal_vllm_startup_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = main_config_gen.CONFIG_ROOT
        original_base = main_config_gen.BASE_FILE
        try:
            main_config_gen.CONFIG_ROOT = root / "configs"
            main_config_gen.BASE_FILE = main_config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            main_config_gen.create_base_and_data_stubs()

            generated_base = main_config_gen.BASE_FILE.read_text(encoding="utf-8")

            assert "llm:" in generated_base
            assert "generator:" in generated_base
            assert "gpu_memory_utilization: 0.55" in generated_base
            assert "startup_required_free_gb: 26" in generated_base
        finally:
            main_config_gen.CONFIG_ROOT = original_root
            main_config_gen.BASE_FILE = original_base


def test_repo_dynamic_base_resolves_to_formal_vllm_startup_guard():
    repo_base = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "single_node_tuning_round23_dynamic"
        / "_base_selector_tuning_round23_dynamic.yaml"
    )
    merged = load_yaml_with_inherits(repo_base)
    llm_generator = merged["llm"]["generator"]

    assert float(llm_generator["gpu_memory_utilization"]) == 0.55
    assert float(llm_generator["startup_required_free_gb"]) == 26


def test_runner_supports_e4_modes_and_method_specific_sidecars():
    paths = runner.resolve_mode_paths("e4_a_oneshot_all6_repeat15")
    assert paths["manifest_relpath"] == (
        "e4_a_oneshot_all6_repeat15/round23_e4_a_oneshot_all6_repeat15_manifest.tsv"
    )
    assert paths["dataset_split"] == "all6"
    assert runner.sidecar_suffix_for_method("round23_absk_oneshot") == "_absolute_k_controller_runtime.json"
    assert runner.sidecar_suffix_for_method("round23_keepk0") == "_keep_k0_runtime.json"
    assert runner.resolve_mode_paths("e4_b_keepk0_all6_repeat15")["dataset_split"] == "all6"
    assert runner.resolve_mode_paths("e4_c_three_round_stress_all6_repeat15")["dataset_split"] == "all6"


def test_parse_nvidia_smi_memory_report_prefers_explicit_gpu_index():
    report = "\n".join(
        [
            "0, NVIDIA RTX A6000, 10240",
            "1, NVIDIA RTX A6000, 20480",
        ]
    )
    index, free_gb = runner.parse_nvidia_smi_memory_report(
        report,
        target_name_token="RTX A6000",
        preferred_index="1",
    )
    assert index == "1"
    assert round(free_gb, 2) == 20.0


def test_e4_all6_sequential_scripts_reset_summary_and_pin_gpu_index():
    for path in (
        Path(__file__).parent / "run_round23_e4_all6_smoke18_sequential.sh",
        Path(__file__).parent / "run_round23_e4_all6_repeat15_270_sequential.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert "RESET_SUMMARY" in text
        assert "--reset-summary" in text
        assert "--target-gpu-index" in text
