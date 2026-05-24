from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import generate_round23_e4_experiment_configs as e4_config_gen  # noqa: E402
import round23_dynamic_experiment_runner as runner  # noqa: E402
from run_round23_with_absolute_k_controller import generate_override_config as generate_absk_override_config  # noqa: E402


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


def test_generate_e4_configs_creates_seen4_manifests_with_relative_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e4_config_gen.CONFIG_ROOT
        original_base = e4_config_gen.BASE_FILE
        original_mode_specs = {mode: dict(spec) for mode, spec in e4_config_gen.MODE_SPECS.items()}
        try:
            e4_config_gen.CONFIG_ROOT = root / "configs"
            e4_config_gen.BASE_FILE = e4_config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            e4_config_gen.MODE_SPECS["e4_a_oneshot_seen_smoke"]["seeds"] = [42]
            e4_config_gen.MODE_SPECS["e4_a_oneshot_seen_repeat15"]["seeds"] = [42, 123]
            e4_config_gen.create_base_and_data_stubs()
            e4_config_gen.create_mode_configs("e4_a_oneshot_seen_smoke")
            e4_config_gen.create_mode_configs("e4_a_oneshot_seen_repeat15")

            smoke_manifest = (
                e4_config_gen.CONFIG_ROOT
                / "e4_a_oneshot_seen_smoke"
                / "round23_e4_a_oneshot_seen_smoke_manifest.tsv"
            )
            with smoke_manifest.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(smoke_rows) == 4
            assert {row["method"] for row in smoke_rows} == {"round23_absk_oneshot"}
            assert not Path(smoke_rows[0]["config_path"]).is_absolute()

            repeat_manifest = (
                e4_config_gen.CONFIG_ROOT
                / "e4_a_oneshot_seen_repeat15"
                / "round23_e4_a_oneshot_seen_repeat15_manifest.tsv"
            )
            with repeat_manifest.open("r", encoding="utf-8", newline="") as handle:
                repeat_rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(repeat_rows) == 8
            assert all(
                row["output_root"].startswith("outputs/e4_a_oneshot_seen_repeat15/")
                for row in repeat_rows
            )
        finally:
            e4_config_gen.CONFIG_ROOT = original_root
            e4_config_gen.BASE_FILE = original_base
            e4_config_gen.MODE_SPECS = original_mode_specs


def test_runner_supports_e4_modes_and_method_specific_sidecars():
    paths = runner.resolve_mode_paths("e4_a_oneshot_seen_repeat15")
    assert paths["manifest_relpath"] == (
        "e4_a_oneshot_seen_repeat15/round23_e4_a_oneshot_seen_repeat15_manifest.tsv"
    )
    assert runner.sidecar_suffix_for_method("round23_absk_oneshot") == "_absolute_k_controller_runtime.json"
    assert runner.sidecar_suffix_for_method("round23_keepk0") == "_keep_k0_runtime.json"
