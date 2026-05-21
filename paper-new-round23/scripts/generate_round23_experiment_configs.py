#!/usr/bin/env python3
"""Generate formal round23 experiment configs and manifests."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "single_node_tuning_round23_dynamic"
BASE_FILE = CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
CONFIG_MANIFEST_ROOT = Path("configs") / "experiments" / "single_node_tuning_round23_dynamic"
DEFAULT_CONTROLLER_SCOPE = "all6"
DEFAULT_CONTROLLER_BUNDLE = "round23_controller_1200_all6_top1_delta_m0005_extratrees_broad_no_dataset"
SEEDS_SMOKE = [42]
SEEDS_PILOT = [42, 123, 456]
SEEDS_REPEAT10 = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200]
SEEDS_REPEAT15 = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200, 300, 400, 500, 600, 700]
SEEDS_REPEAT30 = [
    42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1111, 1212, 1313,
    1414, 1515, 1616, 1717, 1818, 1919, 2020, 2222, 2468, 3141,
]
SEEDS_REPEAT40 = SEEDS_REPEAT30 + [
    4242, 5151, 6262, 7373, 8484, 9090, 10010, 11111, 12121, 13131,
]
SEEDS = list(SEEDS_REPEAT40)
SEEN_DATASETS = ["jobs", "congressional", "forums", "microblog"]
UNSEEN_DATASETS = ["imdb", "openreview"]
EXTRA_UNSEEN_DATASETS = ["bioarxiv", "rotten_tomatoes", "twitter_emotion_binary"]
ALL_DATASETS = SEEN_DATASETS + UNSEEN_DATASETS + EXTRA_UNSEEN_DATASETS

MODE_SPECS = {
    "real_smoke": {
        "subdir": "real_smoke",
        "manifest_name": "round23_real_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": [42],
        "experiment_prefix": "r23_real",
        "source_env": "round23_real_smoke",
        "output_root_prefix": "outputs/real_smoke",
    },
    "quick_compare": {
        "subdir": "quick_compare_repeat30",
        "manifest_name": "round23_quick_compare_repeat30_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT30,
        "experiment_prefix": "r23_qc",
        "source_env": "round23_quick_compare_repeat30",
        "output_root_prefix": "outputs/quick_compare_repeat30",
    },
    "unseen_dataset_final_eval": {
        "subdir": "unseen_dataset_final_eval_repeat40",
        "manifest_name": "round23_unseen_dataset_final_eval_repeat40_manifest.tsv",
        "datasets": UNSEEN_DATASETS,
        "seeds": SEEDS_REPEAT40,
        "experiment_prefix": "r23_ud",
        "source_env": "round23_unseen_dataset_final_eval_repeat40",
        "output_root_prefix": "outputs/unseen_dataset_final_eval_repeat40",
    },
    "thesis_main_seen_pilot": {
        "subdir": "thesis_main_seen_pilot",
        "manifest_name": "round23_thesis_main_seen_pilot_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_PILOT,
        "experiment_prefix": "r23_e1_pilot",
        "source_env": "thesis_main_seen_pilot",
        "output_root_prefix": "outputs/thesis_main_seen_pilot",
    },
    "thesis_main_seen_smoke": {
        "subdir": "thesis_main_seen_smoke",
        "manifest_name": "round23_thesis_main_seen_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e1_smoke",
        "source_env": "thesis_main_seen_smoke",
        "output_root_prefix": "outputs/thesis_main_seen_smoke",
    },
    "thesis_main_seen_repeat10": {
        "subdir": "thesis_main_seen_repeat10",
        "manifest_name": "round23_thesis_main_seen_repeat10_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT10,
        "experiment_prefix": "r23_e1_r10",
        "source_env": "thesis_main_seen_repeat10",
        "output_root_prefix": "outputs/thesis_main_seen_repeat10",
    },
    "thesis_main_seen_repeat15": {
        "subdir": "thesis_main_seen_repeat15",
        "manifest_name": "round23_thesis_main_seen_repeat15_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e1_r15",
        "source_env": "thesis_main_seen_repeat15",
        "output_root_prefix": "outputs/thesis_main_seen_repeat15",
    },
    "thesis_main_seen_repeat30": {
        "subdir": "thesis_main_seen_repeat30",
        "manifest_name": "round23_thesis_main_seen_repeat30_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT30,
        "experiment_prefix": "r23_e1_r30",
        "source_env": "thesis_main_seen_repeat30",
        "output_root_prefix": "outputs/thesis_main_seen_repeat30",
    },
    "thesis_e2_extra_unseen_smoke": {
        "subdir": "thesis_e2_extra_unseen_smoke",
        "manifest_name": "round23_thesis_e2_extra_unseen_smoke_manifest.tsv",
        "datasets": EXTRA_UNSEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r23_e2_smoke",
        "source_env": "thesis_e2_extra_unseen_smoke",
        "output_root_prefix": "outputs/thesis_e2_extra_unseen_smoke",
    },
    "thesis_e2_extra_unseen_repeat15": {
        "subdir": "thesis_e2_extra_unseen_repeat15",
        "manifest_name": "round23_thesis_e2_extra_unseen_repeat15_manifest.tsv",
        "datasets": EXTRA_UNSEEN_DATASETS,
        "seeds": SEEDS_REPEAT15,
        "experiment_prefix": "r23_e2_r15",
        "source_env": "thesis_e2_extra_unseen_repeat15",
        "output_root_prefix": "outputs/thesis_e2_extra_unseen_repeat15",
    },
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def create_base_and_data_stubs() -> None:
    _write_text(
        BASE_FILE,
        "\n".join(
            [
                "inherits:",
                "  - ../../../../paper-new-round19/configs/experiments/single_node_tuning_round19/_base_selector_tuning_round19.yaml",
                "",
                "meta:",
                "  stage: round23_dynamic_runtime",
                "  seed: 42",
                "",
                "paths:",
                "  output_root: outputs/default",
                "",
                "selector:",
                "  seed_top_k: 20",
                "  seed_budget_rule:",
                "    enabled: false",
                "    mode: hierarchical_shape_routing",
                "",
                "round23_controller:",
                "  enabled: true",
                "  reference_budget: 20",
                "  action_space: [-2, -1, 0, 1, 2]",
                "  source_env: round23_dynamic_runtime",
                "  context_family: natural",
            ]
        ) + "\n",
    )
    for dataset in ALL_DATASETS:
        _write_text(
            CONFIG_ROOT / f"_data_{dataset}.yaml",
            f"inherits:\n  - ../../../../paper-new-round19/configs/experiments/single_node_tuning_round19/_data_{dataset}.yaml\n",
        )


def _build_config_yaml(
    *,
    dataset: str,
    experiment_id: str,
    seed: int,
    output_root: str,
    source_env: str,
) -> str:
    return (
        "\n".join(
            [
                "inherits:",
                "  - ../_base_selector_tuning_round23_dynamic.yaml",
                f"  - ../_data_{dataset}.yaml",
                "",
                "meta:",
                f"  experiment_id: {experiment_id}",
                f"  seed: {seed}",
                f"  dataset_name: {dataset}",
                "",
                "paths:",
                f"  output_root: {output_root}",
                "",
                "selector:",
                "  seed_top_k: 20",
                "",
                "round23_controller:",
                f"  source_env: {source_env}",
                "  context_family: natural",
                "  reference_budget: 20",
            ]
        )
        + "\n"
    )


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
                    "method": "round23",
                    "controller_scope": DEFAULT_CONTROLLER_SCOPE,
                    "controller_bundle": DEFAULT_CONTROLLER_BUNDLE,
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
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def create_real_smoke() -> None:
    create_mode_configs("real_smoke")


def create_quick_compare() -> None:
    create_mode_configs("quick_compare")


def create_unseen_dataset_final_eval() -> None:
    create_mode_configs("unseen_dataset_final_eval")


def create_thesis_main_seen_pilot() -> None:
    create_mode_configs("thesis_main_seen_pilot")


def create_thesis_main_seen_smoke() -> None:
    create_mode_configs("thesis_main_seen_smoke")


def create_thesis_main_seen_repeat10() -> None:
    create_mode_configs("thesis_main_seen_repeat10")


def create_thesis_main_seen_repeat15() -> None:
    create_mode_configs("thesis_main_seen_repeat15")


def create_thesis_main_seen_repeat30() -> None:
    create_mode_configs("thesis_main_seen_repeat30")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate round23 experiment configs and manifests")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODE_SPECS.keys()),
        help="Mode to generate. May be passed multiple times; default generates all modes.",
    )
    args = parser.parse_args()
    modes = args.mode or list(MODE_SPECS.keys())
    create_base_and_data_stubs()
    for mode in modes:
        create_mode_configs(mode)
    print("Generated round23 configs under:", CONFIG_ROOT)
    print("Generated modes:", ",".join(modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
