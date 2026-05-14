#!/usr/bin/env python3
"""Generate round23 real_smoke and quick_compare_repeat30 experiment configs."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "single_node_tuning_round23_dynamic"
BASE_FILE = CONFIG_ROOT / "_base_selector_tuning_round23_dynamic.yaml"
SEEDS = [
    42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1111, 1212, 1313,
    1414, 1515, 1616, 1717, 1818, 1919, 2020, 2222, 2468, 3141,
]
DATASETS = ["jobs", "congressional", "forums", "microblog"]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    for dataset in DATASETS:
        _write_text(
            CONFIG_ROOT / f"_data_{dataset}.yaml",
            f"inherits:\n  - ../../../../paper-new-round19/configs/experiments/single_node_tuning_round19/_data_{dataset}.yaml\n",
        )


def create_real_smoke() -> None:
    rows: list[dict[str, str | int]] = []
    target_dir = CONFIG_ROOT / "real_smoke"
    for dataset in DATASETS:
        experiment_id = f"r23_real_{dataset}_seed42"
        config_path = target_dir / f"{experiment_id}.yaml"
        output_root = f"outputs/real_smoke/{dataset}/seed42"
        _write_text(
            config_path,
            "\n".join(
                [
                    "inherits:",
                    "  - ../_base_selector_tuning_round23_dynamic.yaml",
                    f"  - ../_data_{dataset}.yaml",
                    "",
                    "meta:",
                    f"  experiment_id: {experiment_id}",
                    "  seed: 42",
                    "",
                    "paths:",
                    f"  output_root: {output_root}",
                    "",
                    "selector:",
                    "  seed_top_k: 20",
                    "",
                    "round23_controller:",
                    "  source_env: round23_real_smoke",
                    "  context_family: natural",
                    "  reference_budget: 20",
                ]
            ) + "\n",
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "seed": 42,
                "config_path": str(config_path.resolve()),
                "output_root": output_root,
            }
        )
    manifest_path = target_dir / "round23_real_smoke_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "dataset", "seed", "config_path", "output_root"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def create_quick_compare() -> None:
    rows: list[dict[str, str | int]] = []
    target_dir = CONFIG_ROOT / "quick_compare_repeat30"
    for dataset in DATASETS:
        for seed in SEEDS:
            experiment_id = f"r23_qc_{dataset}_seed{seed}"
            config_path = target_dir / f"{experiment_id}.yaml"
            output_root = f"outputs/quick_compare_repeat30/{dataset}/seed{seed}"
            _write_text(
                config_path,
                "\n".join(
                    [
                        "inherits:",
                        "  - ../_base_selector_tuning_round23_dynamic.yaml",
                        f"  - ../_data_{dataset}.yaml",
                        "",
                        "meta:",
                        f"  experiment_id: {experiment_id}",
                        f"  seed: {seed}",
                        "",
                        "paths:",
                        f"  output_root: {output_root}",
                        "",
                        "selector:",
                        "  seed_top_k: 20",
                        "",
                        "round23_controller:",
                        "  source_env: round23_quick_compare_repeat30",
                        "  context_family: natural",
                        "  reference_budget: 20",
                    ]
                ) + "\n",
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "seed": seed,
                    "config_path": str(config_path.resolve()),
                    "output_root": output_root,
                }
            )
    manifest_path = target_dir / "round23_quick_compare_repeat30_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "dataset", "seed", "config_path", "output_root"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    create_base_and_data_stubs()
    create_real_smoke()
    create_quick_compare()
    print("Generated round23 configs under:", CONFIG_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
