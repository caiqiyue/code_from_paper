from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import build_e9_federated_partitions as partition_builder  # noqa: E402
import generate_round23_e9_federated_experiment_configs as e9_config_gen  # noqa: E402
import round23_federated_utils as fed_utils  # noqa: E402


def _make_train_json(path: Path, *, count: int) -> None:
    texts: list[str] = []
    families = (
        ("alpha", 18),
        ("beta", 55),
        ("gamma", 110),
        ("delta", 220),
    )
    for index in range(count):
        token, base_len = families[index % len(families)]
        length = base_len + (index % 7)
        texts.append(" ".join([token] * length))
    path.write_text(json.dumps(texts, ensure_ascii=True), encoding="utf-8")


def test_allocate_client_prompt_budget_matches_e9_fairness_examples():
    assert fed_utils.allocate_client_prompt_budget(total_prompt_budget=32, num_clients=4) == [8, 8, 8, 8]
    assert fed_utils.allocate_client_prompt_budget(total_prompt_budget=32, num_clients=8) == [4] * 8


def test_build_partitions_supports_all_three_e9_split_modes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_path = root / "jobs_train.json"
        _make_train_json(train_path, count=96)

        uniform = partition_builder.build_partition_artifact(
            dataset_name="jobs",
            train_path=train_path,
            num_clients=4,
            split_mode="uniform",
            seed=42,
            output_dir=root / "uniform",
        )
        noniid = partition_builder.build_partition_artifact(
            dataset_name="jobs",
            train_path=train_path,
            num_clients=4,
            split_mode="noniid",
            seed=42,
            output_dir=root / "noniid",
        )
        imbalance = partition_builder.build_partition_artifact(
            dataset_name="jobs",
            train_path=train_path,
            num_clients=8,
            split_mode="imbalance_noniid",
            seed=42,
            output_dir=root / "imbalance",
        )

        uniform_counts = [client["train_count"] for client in uniform["clients"]]
        noniid_clusters = {client["dominant_cluster"] for client in noniid["clients"]}
        imbalance_counts = [client["train_count"] for client in imbalance["clients"]]

        assert max(uniform_counts) - min(uniform_counts) <= 1
        assert len(noniid_clusters) >= 2
        assert sum(imbalance_counts) == 96
        assert max(imbalance_counts) > min(imbalance_counts)
        assert (root / "noniid" / "partition_manifest.json").exists()
        assert (root / "imbalance" / "client_007_train.json").exists()


def test_generate_e9_configs_create_270_rows_with_relative_paths_and_prompt_budgets():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        original_root = e9_config_gen.CONFIG_ROOT
        original_base = e9_config_gen.BASE_FILE
        try:
            e9_config_gen.CONFIG_ROOT = root / "configs"
            e9_config_gen.BASE_FILE = e9_config_gen.CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
            e9_config_gen.create_base_and_data_stubs()
            for mode in e9_config_gen.DEFAULT_FORMAL_MODES:
                e9_config_gen.create_mode_configs(mode)

            manifest_paths = {
                mode: e9_config_gen.CONFIG_ROOT / spec["subdir"] / spec["manifest_name"]
                for mode, spec in e9_config_gen.MODE_SPECS.items()
                if mode in e9_config_gen.DEFAULT_FORMAL_MODES
            }
            expected_methods = {"pretext", "round19", "round23"}
            for mode, manifest_path in manifest_paths.items():
                with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                assert len(rows) == 90
                assert {row["method"] for row in rows} == {"e9_pretext", "e9_round19", "e9_round23"}
                assert all(not Path(row["config_path"]).is_absolute() for row in rows)
                assert all(row["output_root"].startswith(f"outputs/{mode}/") for row in rows)
                budgets = {tuple(json.loads(row["per_client_prompt_budget"])) for row in rows}
                if "e9_f8_" in mode:
                    assert budgets == {(4, 4, 4, 4, 4, 4, 4, 4)}
                else:
                    assert budgets == {(8, 8, 8, 8)}
                assert all(row["partition_manifest"].startswith("paper-new-round23/artifacts/e9_partitions/") for row in rows)
        finally:
            e9_config_gen.CONFIG_ROOT = original_root
            e9_config_gen.BASE_FILE = original_base


def test_e9_sequential_script_uses_retry_loop_and_all_three_modes():
    script_path = Path(__file__).parent / "run_round23_e9_federated_all6_repeat5_270_sequential.sh"
    text = script_path.read_text(encoding="utf-8")
    assert "--max-attempts 3" in text
    assert "--retry-delay-seconds 10" in text
    assert "--retry-all-failures" in text
    assert "--target-gpu-index" in text
    assert "e9_f4_uniform_all6_repeat5" in text
    assert "e9_f4_noniid_all6_repeat5" in text
    assert "e9_f8_imbalance_noniid_all6_repeat5" in text
