#!/usr/bin/env python3
"""Generate E5 anchor-boundary round23 experiment configs and manifests."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_round23_experiment_configs import (
    CONFIG_MANIFEST_ROOT,
    CONFIG_ROOT,
    DEFAULT_CONTROLLER_BUNDLE,
    SEEDS_SMOKE,
    SEEN_DATASETS,
    SEEDS_REPEAT10,
    UNSEEN_DATASETS,
    _build_config_yaml,
    _write_text,
    create_base_and_data_stubs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_SCOPE = "all6"
E5_ALL6_DATASETS = list(SEEN_DATASETS) + list(UNSEEN_DATASETS)
SEEDS_REPEAT5 = list(SEEDS_REPEAT10[:5])
DEFAULT_FORMAL_MODES = [
    "e5_anchor_k19_keepk0_all6_smoke",
    "e5_anchor_k19_round23_all6_smoke",
    "e5_anchor_k21_keepk0_all6_smoke",
    "e5_anchor_k21_round23_all6_smoke",
    "e5_anchor_k19_keepk0_all6_repeat5",
    "e5_anchor_k19_round23_all6_repeat5",
    "e5_anchor_k21_keepk0_all6_repeat5",
    "e5_anchor_k21_round23_all6_repeat5",
]

MODE_SPECS = {
    "e5_anchor_k19_keepk0_all6_smoke": {
        "subdir": "e5_anchor_k19_keepk0_all6_smoke",
        "manifest_name": "round23_e5_anchor_k19_keepk0_all6_smoke_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e5_k19_keepk0_smoke",
        "source_env": "round23_e5_anchor_k19_keepk0_all6_smoke",
        "output_root_prefix": "outputs/e5_anchor_k19_keepk0_all6_smoke",
        "method": "round23_keepk0",
        "controller_bundle": "",
        "reference_budget": 19,
    },
    "e5_anchor_k19_round23_all6_smoke": {
        "subdir": "e5_anchor_k19_round23_all6_smoke",
        "manifest_name": "round23_e5_anchor_k19_round23_all6_smoke_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e5_k19_round23_smoke",
        "source_env": "round23_e5_anchor_k19_round23_all6_smoke",
        "output_root_prefix": "outputs/e5_anchor_k19_round23_all6_smoke",
        "method": "round23",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "reference_budget": 19,
    },
    "e5_anchor_k21_keepk0_all6_smoke": {
        "subdir": "e5_anchor_k21_keepk0_all6_smoke",
        "manifest_name": "round23_e5_anchor_k21_keepk0_all6_smoke_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e5_k21_keepk0_smoke",
        "source_env": "round23_e5_anchor_k21_keepk0_all6_smoke",
        "output_root_prefix": "outputs/e5_anchor_k21_keepk0_all6_smoke",
        "method": "round23_keepk0",
        "controller_bundle": "",
        "reference_budget": 21,
    },
    "e5_anchor_k21_round23_all6_smoke": {
        "subdir": "e5_anchor_k21_round23_all6_smoke",
        "manifest_name": "round23_e5_anchor_k21_round23_all6_smoke_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e5_k21_round23_smoke",
        "source_env": "round23_e5_anchor_k21_round23_all6_smoke",
        "output_root_prefix": "outputs/e5_anchor_k21_round23_all6_smoke",
        "method": "round23",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "reference_budget": 21,
    },
    "e5_anchor_k19_keepk0_all6_repeat5": {
        "subdir": "e5_anchor_k19_keepk0_all6_repeat5",
        "manifest_name": "round23_e5_anchor_k19_keepk0_all6_repeat5_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT5,
        "experiment_prefix": "r23_e5_k19_keepk0_r5",
        "source_env": "round23_e5_anchor_k19_keepk0_all6_repeat5",
        "output_root_prefix": "outputs/e5_anchor_k19_keepk0_all6_repeat5",
        "method": "round23_keepk0",
        "controller_bundle": "",
        "reference_budget": 19,
    },
    "e5_anchor_k19_round23_all6_repeat5": {
        "subdir": "e5_anchor_k19_round23_all6_repeat5",
        "manifest_name": "round23_e5_anchor_k19_round23_all6_repeat5_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT5,
        "experiment_prefix": "r23_e5_k19_round23_r5",
        "source_env": "round23_e5_anchor_k19_round23_all6_repeat5",
        "output_root_prefix": "outputs/e5_anchor_k19_round23_all6_repeat5",
        "method": "round23",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "reference_budget": 19,
    },
    "e5_anchor_k21_keepk0_all6_repeat5": {
        "subdir": "e5_anchor_k21_keepk0_all6_repeat5",
        "manifest_name": "round23_e5_anchor_k21_keepk0_all6_repeat5_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT5,
        "experiment_prefix": "r23_e5_k21_keepk0_r5",
        "source_env": "round23_e5_anchor_k21_keepk0_all6_repeat5",
        "output_root_prefix": "outputs/e5_anchor_k21_keepk0_all6_repeat5",
        "method": "round23_keepk0",
        "controller_bundle": "",
        "reference_budget": 21,
    },
    "e5_anchor_k21_round23_all6_repeat5": {
        "subdir": "e5_anchor_k21_round23_all6_repeat5",
        "manifest_name": "round23_e5_anchor_k21_round23_all6_repeat5_manifest.tsv",
        "datasets": E5_ALL6_DATASETS,
        "seeds": SEEDS_REPEAT5,
        "experiment_prefix": "r23_e5_k21_round23_r5",
        "source_env": "round23_e5_anchor_k21_round23_all6_repeat5",
        "output_root_prefix": "outputs/e5_anchor_k21_round23_all6_repeat5",
        "method": "round23",
        "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
        "reference_budget": 21,
    },
}


def _build_e5_config_yaml(
    *,
    dataset: str,
    experiment_id: str,
    seed: int,
    output_root: str,
    source_env: str,
    reference_budget: int,
) -> str:
    text = _build_config_yaml(
        dataset=dataset,
        experiment_id=experiment_id,
        seed=seed,
        output_root=output_root,
        source_env=source_env,
    )
    return text.replace("seed_top_k: 20", f"seed_top_k: {reference_budget}").replace(
        "reference_budget: 20", f"reference_budget: {reference_budget}"
    )


def create_mode_configs(mode: str) -> None:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported mode: {mode}")
    spec = MODE_SPECS[mode]
    rows: list[dict[str, str | int]] = []
    target_dir = CONFIG_ROOT / str(spec["subdir"])
    reference_budget = int(spec["reference_budget"])
    for dataset in list(spec["datasets"]):
        for seed in list(spec["seeds"]):
            experiment_id = f"{spec['experiment_prefix']}_{dataset}_seed{seed}"
            config_path = target_dir / f"{experiment_id}.yaml"
            output_root = f"{spec['output_root_prefix']}/{dataset}/seed{seed}"
            _write_text(
                config_path,
                _build_e5_config_yaml(
                    dataset=dataset,
                    experiment_id=experiment_id,
                    seed=int(seed),
                    output_root=output_root,
                    source_env=str(spec["source_env"]),
                    reference_budget=reference_budget,
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
                    "reference_budget": reference_budget,
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
                "reference_budget",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate round23 E5 experiment configs and manifests")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODE_SPECS.keys()),
        help="Mode to generate. May be passed multiple times; default generates all E5 modes.",
    )
    args = parser.parse_args()
    modes = args.mode or list(DEFAULT_FORMAL_MODES)
    create_base_and_data_stubs()
    for mode in modes:
        create_mode_configs(mode)
    print("Generated round23 E5 configs under:", CONFIG_ROOT)
    print("Generated modes:", ",".join(modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
