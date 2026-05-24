#!/usr/bin/env python3
"""Generate E4 round23 experiment configs and manifests."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_round23_experiment_configs import (
    CONFIG_MANIFEST_ROOT,
    CONFIG_ROOT,
    DEFAULT_CONTROLLER_BUNDLE,
    SEEDS_PILOT,
    SEEDS_REPEAT15,
    SEEDS_SMOKE,
    SEEN_DATASETS,
    UNSEEN_DATASETS,
    _build_config_yaml,
    _write_text,
    create_base_and_data_stubs,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
DEFAULT_CONTROLLER_SCOPE = "all6"
DEFAULT_ONE_SHOT_BUNDLE = "round23_absk_1200_all6_top1_delta_m0005_extratrees_no_dataset"
E4_ALL6_DATASETS = list(SEEN_DATASETS) + list(UNSEEN_DATASETS)
DEFAULT_FORMAL_MODES = [
    "e4_a_oneshot_all6_smoke",
    "e4_a_oneshot_all6_repeat15",
    "e4_b_keepk0_all6_smoke",
    "e4_b_keepk0_all6_repeat15",
    "e4_c_three_round_stress_all6_smoke",
    "e4_c_three_round_stress_all6_repeat15",
]

MODE_SPECS = {
    "e4_a_oneshot_all6_smoke": {
        "subdir": "e4_a_oneshot_all6_smoke",
        "manifest_name": "round23_e4_a_oneshot_all6_smoke_manifest.tsv",
        "datasets": E4_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e4_a_smoke",
        "source_env": "round23_e4_a_oneshot_all6_smoke",
        "output_root_prefix": "outputs/e4_a_oneshot_all6_smoke",
        "method": "round23_absk_oneshot",
        "controller_bundle": DEFAULT_ONE_SHOT_BUNDLE,
    },
    "e4_a_oneshot_all6_repeat15": {
        "subdir": "e4_a_oneshot_all6_repeat15",
        "manifest_name": "round23_e4_a_oneshot_all6_repeat15_manifest.tsv",
        "datasets": E4_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e4_a_r15",
        "source_env": "round23_e4_a_oneshot_all6_repeat15",
        "output_root_prefix": "outputs/e4_a_oneshot_all6_repeat15",
        "method": "round23_absk_oneshot",
        "controller_bundle": DEFAULT_ONE_SHOT_BUNDLE,
    },
    "e4_a_oneshot_seen_smoke": {
        "subdir": "e4_a_oneshot_seen_smoke",
        "manifest_name": "round23_e4_a_oneshot_seen_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e4_a_smoke",
        "source_env": "round23_e4_a_oneshot_seen_smoke",
        "output_root_prefix": "outputs/e4_a_oneshot_seen_smoke",
        "method": "round23_absk_oneshot",
        "controller_bundle": DEFAULT_ONE_SHOT_BUNDLE,
    },
    "e4_a_oneshot_seen_repeat15": {
        "subdir": "e4_a_oneshot_seen_repeat15",
        "manifest_name": "round23_e4_a_oneshot_seen_repeat15_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e4_a_r15",
        "source_env": "round23_e4_a_oneshot_seen_repeat15",
        "output_root_prefix": "outputs/e4_a_oneshot_seen_repeat15",
        "method": "round23_absk_oneshot",
        "controller_bundle": DEFAULT_ONE_SHOT_BUNDLE,
    },
    "e4_b_keepk0_all6_smoke": {
        "subdir": "e4_b_keepk0_all6_smoke",
        "manifest_name": "round23_e4_b_keepk0_all6_smoke_manifest.tsv",
        "datasets": E4_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e4_b_smoke",
        "source_env": "round23_e4_b_keepk0_all6_smoke",
        "output_root_prefix": "outputs/e4_b_keepk0_all6_smoke",
        "method": "round23_keepk0",
        "controller_bundle": "",
    },
    "e4_b_keepk0_all6_repeat15": {
        "subdir": "e4_b_keepk0_all6_repeat15",
        "manifest_name": "round23_e4_b_keepk0_all6_repeat15_manifest.tsv",
        "datasets": E4_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e4_b_r15",
        "source_env": "round23_e4_b_keepk0_all6_repeat15",
        "output_root_prefix": "outputs/e4_b_keepk0_all6_repeat15",
        "method": "round23_keepk0",
        "controller_bundle": "",
    },
    "e4_b_keepk0_seen_smoke": {
        "subdir": "e4_b_keepk0_seen_smoke",
        "manifest_name": "round23_e4_b_keepk0_seen_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e4_b_smoke",
        "source_env": "round23_e4_b_keepk0_seen_smoke",
        "output_root_prefix": "outputs/e4_b_keepk0_seen_smoke",
        "method": "round23_keepk0",
        "controller_bundle": "",
    },
    "e4_b_keepk0_seen_repeat15": {
        "subdir": "e4_b_keepk0_seen_repeat15",
        "manifest_name": "round23_e4_b_keepk0_seen_repeat15_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e4_b_r15",
        "source_env": "round23_e4_b_keepk0_seen_repeat15",
        "output_root_prefix": "outputs/e4_b_keepk0_seen_repeat15",
        "method": "round23_keepk0",
        "controller_bundle": "",
    },
    "e4_c_three_round_stress_all6_smoke": {
        "subdir": "e4_c_three_round_stress_all6_smoke",
        "manifest_name": "round23_e4_c_three_round_stress_all6_smoke_manifest.tsv",
        "datasets": E4_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e4_c_smoke",
        "source_env": "round23_e4_c_three_round_stress_all6_smoke",
        "output_root_prefix": "outputs/e4_c_three_round_stress_all6_smoke",
        "method": "round23_3round_stress",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "note": "non-formal heuristic three-round stress",
    },
    "e4_c_three_round_stress_all6_repeat15": {
        "subdir": "e4_c_three_round_stress_all6_repeat15",
        "manifest_name": "round23_e4_c_three_round_stress_all6_repeat15_manifest.tsv",
        "datasets": E4_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e4_c_r15",
        "source_env": "round23_e4_c_three_round_stress_all6_repeat15",
        "output_root_prefix": "outputs/e4_c_three_round_stress_all6_repeat15",
        "method": "round23_3round_stress",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "note": "non-formal heuristic three-round stress",
    },
    "e4_c_three_round_stress_smoke": {
        "subdir": "e4_c_three_round_stress_smoke",
        "manifest_name": "round23_e4_c_three_round_stress_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e4_c_smoke",
        "source_env": "round23_e4_c_three_round_stress_smoke",
        "output_root_prefix": "outputs/e4_c_three_round_stress_smoke",
        "method": "round23_3round_stress",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "note": "non-formal heuristic three-round stress",
    },
    "e4_c_three_round_stress_pilot": {
        "subdir": "e4_c_three_round_stress_pilot",
        "manifest_name": "round23_e4_c_three_round_stress_pilot_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_PILOT,
        "experiment_prefix": "r23_e4_c_pilot",
        "source_env": "round23_e4_c_three_round_stress_pilot",
        "output_root_prefix": "outputs/e4_c_three_round_stress_pilot",
        "method": "round23_3round_stress",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "note": "non-formal heuristic three-round stress",
    },
}


def create_mode_configs(mode: str) -> None:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported mode: {mode}")
    spec = MODE_SPECS[mode]
    rows: list[dict[str, str | int]] = []
    target_dir = CONFIG_ROOT / str(spec["subdir"])
    for dataset in list(spec["datasets"]):
        for seed in list(spec["seeds"]):
            experiment_id = f"{spec['experiment_prefix']}_{dataset}_seed{seed}"
            config_path = target_dir / f"{experiment_id}.yaml"
            output_root = f"{spec['output_root_prefix']}/{dataset}/seed{seed}"
            _write_text(
                config_path,
                _build_config_yaml(
                    dataset=dataset,
                    experiment_id=experiment_id,
                    seed=int(seed),
                    output_root=output_root,
                    source_env=str(spec["source_env"]),
                ),
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "seed": int(seed),
                    "config_path": (CONFIG_MANIFEST_ROOT / str(spec["subdir"]) / config_path.name).as_posix(),
                    "output_root": output_root,
                    "method": str(spec["method"]),
                    "controller_scope": DEFAULT_CONTROLLER_SCOPE,
                    "controller_bundle": str(spec["controller_bundle"]),
                    "note": str(spec.get("note", "")),
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
                "note",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate round23 E4 experiment configs and manifests")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODE_SPECS.keys()),
        help="Mode to generate. May be passed multiple times; default generates all modes.",
    )
    args = parser.parse_args()
    modes = args.mode or list(DEFAULT_FORMAL_MODES)
    create_base_and_data_stubs()
    for mode in modes:
        create_mode_configs(mode)
    print("Generated round23 E4 configs under:", CONFIG_ROOT)
    print("Generated modes:", ",".join(modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
