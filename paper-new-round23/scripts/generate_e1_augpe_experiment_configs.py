#!/usr/bin/env python3
"""Generate E1 Aug-PE baseline experiment configs and manifests.

Each emitted config inherits the round23 dynamic base (which supplies all data
paths and eval parameters), then adds an ``augpe_baseline`` section so that
``run_e1_augpe_baseline.py`` can read epsilon / delta / seed_top_k without
needing CLI overrides per experiment.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROUND23_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROUND23_ROOT / "configs" / "experiments" / "single_node_tuning_round23_dynamic"
CONFIG_MANIFEST_ROOT = Path("configs") / "experiments" / "single_node_tuning_round23_dynamic"

SEEDS_SMOKE = [42]
SEEDS_REPEAT30 = [
    42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1111, 1212, 1313,
    1414, 1515, 1616, 1717, 1818, 1919, 2020, 2222, 2468, 3141,
]
SEEN_DATASETS = ["jobs", "congressional", "forums", "microblog"]

# Align ε/δ with PrE-Text paper's reported values (ε=1.29, δ=3e-6).
DEFAULT_EPSILON = 1.29
DEFAULT_DELTA = 3e-6

MODE_SPECS = {
    "e1_augpe_seen_smoke": {
        "subdir": "e1_augpe_seen_smoke",
        "manifest_name": "round23_e1_augpe_seen_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "e1_augpe_smoke",
        "source_env": "e1_augpe_seen_smoke",
        "output_root_prefix": "outputs/e1_augpe_seen_smoke",
    },
    "e1_augpe_seen_repeat30": {
        "subdir": "e1_augpe_seen_repeat30",
        "manifest_name": "round23_e1_augpe_seen_repeat30_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT30,
        "experiment_prefix": "e1_augpe",
        "source_env": "e1_augpe_seen_repeat30",
        "output_root_prefix": "outputs/e1_augpe_seen_repeat30",
    },
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


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
                "pipeline:",
                "  stage1_mode: c4_only",
                "  stage2_mode: pretext_bootstrap",
                "  run_eval: true",
                "",
                "selector:",
                "  seed_top_k: 20",
                "  seed_budget_rule:",
                "    enabled: false",
                "",
                "round23_controller:",
                "  enabled: false",
                f"  source_env: {source_env}",
                "",
                "augpe_baseline:",
                "  enabled: true",
                f"  epsilon: {DEFAULT_EPSILON}",
                f"  delta: {DEFAULT_DELTA}",
                "  seed_top_k: 20",
                f"  source_env: {source_env}",
            ]
        )
        + "\n"
    )


def create_mode_configs(mode: str) -> None:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported mode: {mode}")
    spec = MODE_SPECS[mode]
    target_dir = CONFIG_ROOT / str(spec["subdir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
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
                    "method": "e1_augpe",
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
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate E1 Aug-PE experiment configs and manifests")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODE_SPECS.keys()),
        help="Mode to generate. May be passed multiple times; default generates all modes.",
    )
    args = parser.parse_args()
    modes = args.mode or list(MODE_SPECS.keys())
    for mode in modes:
        create_mode_configs(mode)
    print("Generated Aug-PE configs under:", CONFIG_ROOT)
    print("Generated modes:", ",".join(modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
