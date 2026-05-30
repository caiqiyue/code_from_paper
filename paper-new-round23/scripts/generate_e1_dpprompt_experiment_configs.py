#!/usr/bin/env python3
"""Generate E1 DP-Prompt experiment configs and manifests.

Each emitted config inherits the dp-prompt pretext-style base, overrides the
experiment id, seed, and output_root, and points at the appropriate per-dataset
config under ``dp-prompt/configs/datasets/``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROUND23_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROUND23_ROOT.parent
CONFIG_ROOT = ROUND23_ROOT / "configs" / "experiments" / "single_node_tuning_round23_dynamic"
CONFIG_MANIFEST_ROOT = Path("configs") / "experiments" / "single_node_tuning_round23_dynamic"
DPPROMPT_ROOT = REPO_ROOT / "dp-prompt"
DPPROMPT_BASE_CONFIG = DPPROMPT_ROOT / "configs" / "base" / "e1_dpprompt_e1_base.yaml"
DPPROMPT_DATASETS_DIR = DPPROMPT_ROOT / "configs" / "datasets"

SEEDS_SMOKE = [42]
SEEDS_REPEAT30 = [
    42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1111, 1212, 1313,
    1414, 1515, 1616, 1717, 1818, 1919, 2020, 2222, 2468, 3141,
]
SEEN_DATASETS = ["jobs", "congressional", "forums", "microblog"]

MODE_SPECS = {
    "e1_dpprompt_seen_smoke": {
        "subdir": "e1_dpprompt_seen_smoke",
        "manifest_name": "round23_e1_dpprompt_seen_smoke_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "e1_dpprompt_smoke",
        "output_root_prefix": "outputs/e1_dpprompt_seen_smoke",
    },
    "e1_dpprompt_seen_repeat30": {
        "subdir": "e1_dpprompt_seen_repeat30",
        "manifest_name": "round23_e1_dpprompt_seen_repeat30_manifest.tsv",
        "datasets": SEEN_DATASETS,
        "seeds": SEEDS_REPEAT30,
        "experiment_prefix": "e1_dpprompt",
        "output_root_prefix": "outputs/e1_dpprompt_seen_repeat30",
    },
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _relpath_to(target: Path, anchor_dir: Path) -> str:
    """Compute a portable POSIX relative path from anchor_dir to target."""
    rel = Path(__import__("os").path.relpath(str(target), str(anchor_dir)))
    return rel.as_posix()


def _build_config_yaml(
    *,
    dataset: str,
    experiment_id: str,
    seed: int,
    output_root: str,
    config_dir: Path,
) -> str:
    base_rel = _relpath_to(DPPROMPT_BASE_CONFIG, config_dir)
    data_rel = _relpath_to(DPPROMPT_DATASETS_DIR / f"pretext_{dataset}.yaml", config_dir)
    return (
        "\n".join(
            [
                "inherits:",
                f"  - {base_rel}",
                f"  - {data_rel}",
                "",
                "experiment:",
                f"  id: {experiment_id}",
                "  pipeline_mode: pretext_style",
                "",
                "runtime:",
                f"  seed: {seed}",
                f"  output_root: {output_root}",
                "",
                "data:",
                f"  dataset_name: {dataset}",
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
                    config_dir=target_dir,
                ),
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "seed": int(seed),
                    "config_path": (CONFIG_MANIFEST_ROOT / str(spec["subdir"]) / config_path.name).as_posix(),
                    "output_root": output_root,
                    "method": "e1_dpprompt",
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
    parser = argparse.ArgumentParser(description="Generate E1 DP-Prompt experiment configs and manifests")
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
    print("Generated DP-Prompt configs under:", CONFIG_ROOT)
    print("Generated modes:", ",".join(modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
