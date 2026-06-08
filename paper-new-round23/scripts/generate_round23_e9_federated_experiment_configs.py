#!/usr/bin/env python3
"""Generate E9 federated experiment configs and manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import generate_round23_experiment_configs as base_config_gen
from round23_federated_utils import (
    DEFAULT_TOTAL_PROMPT_BUDGET,
    E9_ALL6_DATASETS,
    E9_REPEAT1_SEEDS,
    E9_REPEAT5_SEEDS,
    allocate_client_prompt_budget,
    default_partition_manifest_relpath,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "single_node_tuning_round23_dynamic"
BASE_FILE = CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
CONFIG_MANIFEST_ROOT = Path("configs") / "experiments" / "single_node_tuning_round23_dynamic"
DEFAULT_CONTROLLER_SCOPE = "all6"
DEFAULT_FORMAL_MODES = [
    "e9_f4_uniform_all6_once",
    "e9_f4_noniid_all6_once",
    "e9_f8_imbalance_noniid_all6_once",
]

METHOD_SPECS = (
    {"method": "e9_pretext", "logical_method": "pretext", "controller_bundle": ""},
    {"method": "e9_round19", "logical_method": "round19", "controller_bundle": ""},
    {"method": "e9_round23", "logical_method": "round23", "controller_bundle": base_config_gen.DEFAULT_CONTROLLER_BUNDLE},
)

MODE_SPECS = {
    "e9_f4_uniform_all6_once": {
        "subdir": "e9_f4_uniform_all6_once",
        "manifest_name": "round23_e9_f4_uniform_all6_once_manifest.tsv",
        "datasets": list(E9_ALL6_DATASETS),
        "seeds": list(E9_REPEAT1_SEEDS),
        "experiment_prefix": "r23_e9_f4u_once",
        "source_env": "e9_f4_uniform_all6_once",
        "output_root_prefix": "outputs/e9_f4_uniform_all6_once",
        "num_clients": 4,
        "split_mode": "uniform",
        "imbalance_mode": "none",
    },
    "e9_f4_noniid_all6_once": {
        "subdir": "e9_f4_noniid_all6_once",
        "manifest_name": "round23_e9_f4_noniid_all6_once_manifest.tsv",
        "datasets": list(E9_ALL6_DATASETS),
        "seeds": list(E9_REPEAT1_SEEDS),
        "experiment_prefix": "r23_e9_f4n_once",
        "source_env": "e9_f4_noniid_all6_once",
        "output_root_prefix": "outputs/e9_f4_noniid_all6_once",
        "num_clients": 4,
        "split_mode": "noniid",
        "imbalance_mode": "none",
    },
    "e9_f8_imbalance_noniid_all6_once": {
        "subdir": "e9_f8_imbalance_noniid_all6_once",
        "manifest_name": "round23_e9_f8_imbalance_noniid_all6_once_manifest.tsv",
        "datasets": list(E9_ALL6_DATASETS),
        "seeds": list(E9_REPEAT1_SEEDS),
        "experiment_prefix": "r23_e9_f8in_once",
        "source_env": "e9_f8_imbalance_noniid_all6_once",
        "output_root_prefix": "outputs/e9_f8_imbalance_noniid_all6_once",
        "num_clients": 8,
        "split_mode": "imbalance_noniid",
        "imbalance_mode": "fixed_tail_v1",
    },
    "e9_f4_uniform_all6_repeat5": {
        "subdir": "e9_f4_uniform_all6_repeat5",
        "manifest_name": "round23_e9_f4_uniform_all6_repeat5_manifest.tsv",
        "datasets": list(E9_ALL6_DATASETS),
        "seeds": list(E9_REPEAT5_SEEDS),
        "experiment_prefix": "r23_e9_f4u_r5",
        "source_env": "e9_f4_uniform_all6_repeat5",
        "output_root_prefix": "outputs/e9_f4_uniform_all6_repeat5",
        "num_clients": 4,
        "split_mode": "uniform",
        "imbalance_mode": "none",
    },
    "e9_f4_noniid_all6_repeat5": {
        "subdir": "e9_f4_noniid_all6_repeat5",
        "manifest_name": "round23_e9_f4_noniid_all6_repeat5_manifest.tsv",
        "datasets": list(E9_ALL6_DATASETS),
        "seeds": list(E9_REPEAT5_SEEDS),
        "experiment_prefix": "r23_e9_f4n_r5",
        "source_env": "e9_f4_noniid_all6_repeat5",
        "output_root_prefix": "outputs/e9_f4_noniid_all6_repeat5",
        "num_clients": 4,
        "split_mode": "noniid",
        "imbalance_mode": "none",
    },
    "e9_f8_imbalance_noniid_all6_repeat5": {
        "subdir": "e9_f8_imbalance_noniid_all6_repeat5",
        "manifest_name": "round23_e9_f8_imbalance_noniid_all6_repeat5_manifest.tsv",
        "datasets": list(E9_ALL6_DATASETS),
        "seeds": list(E9_REPEAT5_SEEDS),
        "experiment_prefix": "r23_e9_f8in_r5",
        "source_env": "e9_f8_imbalance_noniid_all6_repeat5",
        "output_root_prefix": "outputs/e9_f8_imbalance_noniid_all6_repeat5",
        "num_clients": 8,
        "split_mode": "imbalance_noniid",
        "imbalance_mode": "fixed_tail_v1",
    },
}


def create_base_and_data_stubs() -> None:
    original_root = base_config_gen.CONFIG_ROOT
    original_base = base_config_gen.BASE_FILE
    try:
        base_config_gen.CONFIG_ROOT = CONFIG_ROOT
        base_config_gen.BASE_FILE = BASE_FILE
        base_config_gen.create_base_and_data_stubs()
    finally:
        base_config_gen.CONFIG_ROOT = original_root
        base_config_gen.BASE_FILE = original_base


def _method_runtime_block(method: str) -> list[str]:
    if method == "round23":
        return [
            "selector:",
            "  seed_budget_rule:",
            "    enabled: false",
            "",
            "round23_controller:",
            "  enabled: true",
        ]
    if method == "round19":
        return [
            "selector:",
            "  seed_budget_rule:",
            "    enabled: true",
            "",
            "round23_controller:",
            "  enabled: false",
        ]
    return [
        "selector:",
        "  seed_budget_rule:",
        "    enabled: false",
        "",
        "round23_controller:",
        "  enabled: false",
    ]


def _build_e9_config_yaml(
    *,
    dataset: str,
    experiment_id: str,
    seed: int,
    output_root: str,
    source_env: str,
    method: str,
    federated_setting: str,
    num_clients: int,
    split_mode: str,
    imbalance_mode: str,
    partition_manifest: str,
    total_prompt_budget: int,
    per_client_prompt_budget: list[int],
) -> str:
    base_text = base_config_gen._build_config_yaml(
        dataset=dataset,
        experiment_id=experiment_id,
        seed=seed,
        output_root=output_root,
        source_env=source_env,
    ).rstrip()
    extra_lines = [
        "",
        * _method_runtime_block(method),
        "  source_env: " + source_env,
        "  context_family: natural",
        "  reference_budget: 20",
        "",
        "e9_federated:",
        f"  federated_setting: {federated_setting}",
        f"  method: {method}",
        f"  num_clients: {num_clients}",
        f"  split_mode: {split_mode}",
        f"  imbalance_mode: {imbalance_mode}",
        f"  partition_manifest: {partition_manifest}",
        f"  total_prompt_budget: {total_prompt_budget}",
        f"  per_client_prompt_budget: {json.dumps(per_client_prompt_budget)}",
    ]
    return base_text + "\n" + "\n".join(extra_lines) + "\n"


def create_mode_configs(mode: str) -> None:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported mode: {mode}")
    spec = MODE_SPECS[mode]
    rows: list[dict[str, str | int]] = []
    target_dir = CONFIG_ROOT / str(spec["subdir"])
    num_clients = int(spec["num_clients"])
    per_client_prompt_budget = allocate_client_prompt_budget(
        total_prompt_budget=DEFAULT_TOTAL_PROMPT_BUDGET,
        num_clients=num_clients,
    )

    for method_spec in METHOD_SPECS:
        method = str(method_spec["method"])
        logical_method = str(method_spec["logical_method"])
        controller_bundle = str(method_spec["controller_bundle"])
        for dataset in list(spec["datasets"]):
            for seed in list(spec["seeds"]):
                experiment_id = f"{spec['experiment_prefix']}_{method}_{dataset}_seed{seed}"
                config_path = target_dir / f"{experiment_id}.yaml"
                output_root = f"{spec['output_root_prefix']}/{method}/{dataset}/seed{seed}"
                partition_manifest = default_partition_manifest_relpath(
                    federated_setting=mode,
                    dataset_name=dataset,
                    seed=int(seed),
                )
                base_config_gen._write_text(
                    config_path,
                    _build_e9_config_yaml(
                        dataset=dataset,
                        experiment_id=experiment_id,
                        seed=int(seed),
                        output_root=output_root,
                        source_env=str(spec["source_env"]),
                        method=logical_method,
                        federated_setting=mode,
                        num_clients=num_clients,
                        split_mode=str(spec["split_mode"]),
                        imbalance_mode=str(spec["imbalance_mode"]),
                        partition_manifest=partition_manifest,
                        total_prompt_budget=DEFAULT_TOTAL_PROMPT_BUDGET,
                        per_client_prompt_budget=list(per_client_prompt_budget),
                    ),
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": dataset,
                        "seed": int(seed),
                        "config_path": (CONFIG_MANIFEST_ROOT / str(spec["subdir"]) / config_path.name).as_posix(),
                        "output_root": output_root,
                        "method": method,
                        "controller_scope": DEFAULT_CONTROLLER_SCOPE,
                        "controller_bundle": controller_bundle,
                        "federated_setting": mode,
                        "num_clients": num_clients,
                        "split_mode": str(spec["split_mode"]),
                        "imbalance_mode": str(spec["imbalance_mode"]),
                        "partition_manifest": partition_manifest,
                        "total_prompt_budget": DEFAULT_TOTAL_PROMPT_BUDGET,
                        "per_client_prompt_budget": json.dumps(per_client_prompt_budget),
                    }
                )

    manifest_path = target_dir / str(spec["manifest_name"])
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "dataset",
                "seed",
                "config_path",
                "output_root",
                "method",
                "controller_scope",
                "controller_bundle",
                "federated_setting",
                "num_clients",
                "split_mode",
                "imbalance_mode",
                "partition_manifest",
                "total_prompt_budget",
                "per_client_prompt_budget",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate round23 E9 federated experiment configs and manifests")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODE_SPECS.keys()),
        help="Mode to generate. May be passed multiple times; default generates all formal E9 modes.",
    )
    args = parser.parse_args()
    modes = args.mode or list(DEFAULT_FORMAL_MODES)
    create_base_and_data_stubs()
    for mode in modes:
        create_mode_configs(mode)
    print("Generated round23 E9 federated configs under:", CONFIG_ROOT)
    print("Generated modes:", ",".join(modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
