from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
import yaml

import round23_dynamic_experiment_runner as runner

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "paper-new-round19").resolve()))

from run_e9_federated_common import (
    E9_CLIENT_VLLM_MIN_FREE_GB,
    FederatedSettings,
    build_client_partitions,
    build_federated_sidecar,
    ensure_partition_manifest,
    load_federated_settings,
    resolve_partition_clients,
)
from thesis_platform.core.schemas import Sample


def test_runner_supports_e9_modes_and_federated_sidecars():
    uniform_paths = runner.resolve_mode_paths("e9_f4_uniform_all6_once")
    assert uniform_paths["manifest_relpath"] == (
        "e9_f4_uniform_all6_once/round23_e9_f4_uniform_all6_once_manifest.tsv"
    )
    assert uniform_paths["dataset_split"] == "all6"
    assert runner.resolve_mode_paths("e9_f4_noniid_all6_once")["dataset_split"] == "all6"
    assert runner.resolve_mode_paths("e9_f8_imbalance_noniid_all6_once")["dataset_split"] == "all6"
    assert runner.sidecar_suffix_for_method("e9_round23") == "_federated_runtime.json"
    assert runner.sidecar_suffix_for_method("e9_round19") == "_federated_runtime.json"
    assert runner.sidecar_suffix_for_method("e9_pretext") == "_federated_runtime.json"


def test_e9_client_vllm_threshold_is_2gb():
    assert E9_CLIENT_VLLM_MIN_FREE_GB == 2.0


def test_e9_round19_does_not_require_controller_bundle():
    spec = runner.ExperimentSpec(
        experiment_id="r23_e9_test",
        dataset_name="jobs",
        meta_seed=42,
        config_path=Path("config.yaml"),
        output_root="outputs/e9_f4_uniform_all6_repeat5/jobs/seed42",
        method="e9_round19",
    )
    assert runner.resolve_model_dir_for_spec(None, spec) is None


def test_e9_summary_row_reads_federated_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = runner.ExperimentSpec(
            experiment_id="r23_e9_test",
            dataset_name="jobs",
            meta_seed=42,
            config_path=root / "r23_e9_test.yaml",
            output_root="outputs/e9_f4_uniform_all6_repeat5/jobs/seed42",
            method="e9_pretext",
        )
        sidecar_root = root / runner.normalize_output_root(spec.output_root)
        sidecar_root.mkdir(parents=True, exist_ok=True)
        (sidecar_root / "r23_e9_test_federated_runtime.json").write_text(
            json.dumps(
                {
                    "federated_setting": "f4_uniform",
                    "num_clients": 4,
                    "split_mode": "uniform",
                    "imbalance_mode": "none",
                    "client_success_count": 4,
                    "client_failure_count": 0,
                    "aggregated_synthetic_count": 32,
                    "aggregated_synthetic_count_deduped": 29,
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
                mode="e9_f4_uniform_all6_repeat5",
                dataset_split="all6",
                status="success",
                attempt=1,
                duration_seconds=0.1,
            )
        finally:
            runner.ROUND23_ROOT = original_root

        assert row["method_display_name"] == "PrE-Text"
        assert row["federated_setting"] == "f4_uniform"
        assert row["num_clients"] == 4
        assert row["split_mode"] == "uniform"
        assert row["aggregated_synthetic_count"] == 32
        assert row["aggregated_synthetic_count_deduped"] == 29
        assert row["best_top1"] == 0.11


def test_imbalance_noniid_setting_triggers_long_tail_partitions():
    samples = [
        Sample(
            sample_id=f"s{idx}",
            client_id="source",
            round_id="r0",
            source="jobs",
            dataset_name="jobs",
            task_type="classification",
            text=("alpha " * (20 + idx)).strip(),
        )
        for idx in range(96)
    ]
    settings = FederatedSettings(
        federated_setting="e9_f8_imbalance_noniid_all6_repeat5",
        num_clients=8,
        split_mode="imbalance_noniid",
        imbalance_mode="fixed_tail_v1",
        partition_manifest="unused.json",
        total_prompt_budget=32,
        validation_ratio=0.0,
        per_client_prompt_budget=(4, 4, 4, 4, 4, 4, 4, 4),
    )
    partitions = build_client_partitions(samples, settings=settings, seed=42)
    counts = [len(bucket) for bucket in partitions]
    assert len(counts) == 8
    assert sum(counts) == 96
    assert max(counts) > min(counts)


def test_load_federated_settings_reads_partition_manifest_and_existing_manifest_is_consumed():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_path = root / "artifacts" / "partition_manifest.json"
        client_train_path = root / "artifacts" / "client_000_train.json"
        client_train_path.parent.mkdir(parents=True, exist_ok=True)
        client_train_path.write_text(json.dumps(["sample text"], ensure_ascii=False), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "clients": [
                        {
                            "client_id": "client_000",
                            "train_path": str(client_train_path),
                            "train_count": 1,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "meta": {"seed": 42},
                    "data": {"dataset_name": "jobs", "train_path": "unused.json"},
                    "e9_federated": {
                        "federated_setting": "e9_f4_uniform_all6_repeat5",
                        "num_clients": 1,
                        "split_mode": "uniform",
                        "imbalance_mode": "none",
                        "partition_manifest": str(manifest_path),
                        "total_prompt_budget": 32,
                        "per_client_prompt_budget": [32],
                    },
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        _, settings = load_federated_settings(config_path)
        resolved_manifest_path, manifest = ensure_partition_manifest(config_path, settings=settings, seed=42)
        clients = resolve_partition_clients(manifest)

        assert settings.partition_manifest == str(manifest_path)
        assert resolved_manifest_path == manifest_path.resolve()
        assert clients[0]["client_id"] == "client_000"
        assert Path(clients[0]["train_path"]) == client_train_path.resolve()


def test_build_federated_sidecar_promotes_round23_prediction_lists():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        eval_dir = root / "aggregated" / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = eval_dir / "corpus.json"
        canonical_path.write_text(json.dumps(["a", "b"], ensure_ascii=False), encoding="utf-8")
        sidecar_path = build_federated_sidecar(
            output_root=root,
            experiment_id="r23_e9_round23_jobs_seed42",
            method="e9_round23",
            settings=FederatedSettings(
                federated_setting="e9_f4_uniform_all6_repeat5",
                num_clients=2,
                split_mode="uniform",
                imbalance_mode="none",
                partition_manifest="dummy.json",
                total_prompt_budget=32,
                validation_ratio=0.0,
                per_client_prompt_budget=(16, 16),
            ),
            partition_manifest_path=root / "partition_manifest.json",
            client_rows=[
                {
                    "client_id": "client_000",
                    "status": "success",
                    "predicted_delta_k": -1,
                    "predicted_target_budget": 19,
                },
                {
                    "client_id": "client_001",
                    "status": "success",
                    "predicted_delta_k": 1,
                    "predicted_target_budget": 21,
                },
            ],
            aggregated_texts=["x", "y", "z"],
            eval_summary={"canonical_synthetic_corpus_path": str(canonical_path)},
            reference_budget=20,
            controller_bundle="bundle",
            model_metadata={},
        )
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["predicted_delta_k"] == "[-1, 1]"
        assert payload["predicted_target_budget"] == "[19, 21]"
        assert payload["client_success_count"] == 2
        assert payload["aggregated_synthetic_count_deduped"] == 2
