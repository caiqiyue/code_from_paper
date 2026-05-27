#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "single_node_tuning_e6_round14_budget_sweep"
CONFIG_MANIFEST_ROOT = Path("paper-new-round-14/configs/experiments/single_node_tuning_e6_round14_budget_sweep")

BUDGETS = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]
SEEDS_SMOKE = [42]
SEEDS_REPEAT2 = [42, 123]
DATASET_SPECS = {
    "jobs": {
        "train_path": "thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        "eval_path": "thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json",
    },
    "congressional": {
        "train_path": "thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json",
        "eval_path": "thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json",
    },
    "forums": {
        "train_path": "thesis_platform/datasets/pretext_forums/formatted/forums_train.json",
        "eval_path": "thesis_platform/datasets/pretext_forums/formatted/forums_eval.json",
    },
    "microblog": {
        "train_path": "thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json",
        "eval_path": "thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json",
    },
    "imdb": {
        "train_path": "thesis_platform/datasets/pretext_imdb/formatted/imdb_train.json",
        "eval_path": "thesis_platform/datasets/pretext_imdb/formatted/imdb_eval.json",
    },
    "openreview": {
        "train_path": "thesis_platform/datasets/pretext_openreview/formatted/openreview_train.json",
        "eval_path": "thesis_platform/datasets/pretext_openreview/formatted/openreview_eval.json",
    },
}
INITIALIZATION_PATH = "thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json"

MODE_SPECS = {
    "round14_lineage_e6_smoke90": {
        "subdir": "smoke90",
        "manifest_name": "round14_lineage_e6_smoke90_manifest.tsv",
        "seeds": SEEDS_SMOKE,
        "experiment_prefix": "r14_e6_smoke",
    },
    "round14_lineage_e6_repeat2_180": {
        "subdir": "repeat2_180",
        "manifest_name": "round14_lineage_e6_repeat2_180_manifest.tsv",
        "seeds": SEEDS_REPEAT2,
        "experiment_prefix": "r14_e6_r2",
    },
}
DEFAULT_MODES = list(MODE_SPECS.keys())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_base_config() -> Path:
    path = CONFIG_ROOT / "_base_round14_lineage_fixed_budget_replay.yaml"
    text = """inherits:
  - ../single_node_tuning_round2/_e6_combo_safe2.yaml

meta:
  stage: single_node_tuning_e6_round14_budget_sweep
  seed: 42

paths:
  output_root: paper-new-round-14/outputs/round14_lineage_e6_budget_sweep/default

llm:
  generator:
    gpu_memory_utilization: 0.55
    startup_required_free_gb: 26

generator:
  candidate_count: 96
  generated_per_round: 24
  max_rounds: 8

bootstrap:
  gpu_memory_utilization: 0.55
  startup_required_free_gb: 26
"""
    _write_text(path, text)
    return path


def _build_leaf_yaml(*, experiment_id: str, dataset: str, budget: int, seed: int, output_root: str) -> str:
    dataset_spec = DATASET_SPECS[dataset]
    return f"""inherits:
  - ../_base_round14_lineage_fixed_budget_replay.yaml

meta:
  experiment_id: {experiment_id}
  seed: {seed}

paths:
  output_root: {output_root}

data:
  dataset_name: {dataset}
  train_path: {dataset_spec["train_path"]}
  eval_path: {dataset_spec["eval_path"]}
  initialization_path: {INITIALIZATION_PATH}

selector:
  seed_top_k: {budget}
"""


def create_mode_configs(mode: str) -> None:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported mode: {mode}")
    create_base_config()
    spec = MODE_SPECS[mode]
    target_dir = CONFIG_ROOT / str(spec["subdir"])
    rows: list[dict[str, str | int]] = []
    for dataset in DATASET_SPECS:
        for budget in BUDGETS:
            for seed in spec["seeds"]:
                experiment_id = f"{spec['experiment_prefix']}_{dataset}_k{budget}_seed{seed}"
                config_path = target_dir / f"{experiment_id}.yaml"
                output_root = (
                    f"paper-new-round-14/outputs/round14_lineage_e6_budget_sweep/"
                    f"{spec['subdir']}/{dataset}/k{budget}/seed{seed}"
                )
                _write_text(
                    config_path,
                    _build_leaf_yaml(
                        experiment_id=experiment_id,
                        dataset=dataset,
                        budget=budget,
                        seed=int(seed),
                        output_root=output_root,
                    ),
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": dataset,
                        "budget": budget,
                        "seed": int(seed),
                        "config_path": (CONFIG_MANIFEST_ROOT / spec["subdir"] / config_path.name).as_posix(),
                        "output_root": output_root,
                    }
                )

    manifest_path = target_dir / str(spec["manifest_name"])
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment_id", "dataset", "budget", "seed", "config_path", "output_root"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate round14-lineage E6 budget sweep configs.")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODE_SPECS.keys()),
        help="Mode to generate. Defaults to all E6 modes.",
    )
    args = parser.parse_args()
    for mode in args.mode or DEFAULT_MODES:
        create_mode_configs(mode)
    print(f"Generated E6 round14-lineage configs under: {CONFIG_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
