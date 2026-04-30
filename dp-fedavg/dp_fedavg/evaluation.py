from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from thesis_platform.core.config import ExperimentConfig

from .paths import resolve_repo_root


def build_thesis_eval_config(dp_config: dict[str, Any], *, output_dir: Path) -> ExperimentConfig:
    repo_root = resolve_repo_root()
    evaluation_cfg = dict(dp_config.get("evaluation", {}))
    data_cfg = dict(dp_config.get("data", {}))
    runtime_cfg = dict(dp_config.get("runtime", {}))
    raw = {
        "meta": {
            "experiment_id": str(dp_config.get("meta", {}).get("experiment_id", "dp_fedavg")),
            "seed": int(runtime_cfg.get("seed", 42)),
        },
        "paths": {
            "repo_root": str(repo_root),
            "output_root": str(output_dir),
        },
        "data": {
            "dataset_name": str(data_cfg["dataset_name"]),
            "train_path": str(data_cfg["train_path"]),
            "eval_path": str(data_cfg["eval_path"]),
            "initialization_path": str(
                data_cfg.get(
                    "initialization_path",
                    "thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json",
                )
            ),
            "max_samples_per_client": int(data_cfg.get("max_samples_per_client", 8)),
            "initialization_min_words": int(data_cfg.get("initialization_min_words", 20)),
            "train_limit": data_cfg.get("train_limit"),
            "eval_limit": data_cfg.get("eval_limit"),
            "initialization_limit": data_cfg.get("initialization_limit"),
        },
        "runtime": {
            "device": str(runtime_cfg.get("device", "cuda")),
        },
        "downstream_eval": {
            "enabled": bool(evaluation_cfg.get("enabled", True)),
            "kind": "pretext_small_eval",
            "run_large_eval": False,
            "run_small_eval": True,
            "small_eval_mode": str(evaluation_cfg.get("small_eval_mode", "gpt2")),
            "distilgpt2_path": str(evaluation_cfg.get("distilgpt2_path", "thesis_platform/open_model/distilgpt2")),
            "small_epochs": int(evaluation_cfg.get("small_epochs", 1)),
            "small_batch_size": int(evaluation_cfg.get("small_batch_size", 1)),
            "small_eval_batch_size": int(evaluation_cfg.get("small_eval_batch_size", 1)),
            "small_grad_accum_steps": int(evaluation_cfg.get("small_grad_accum_steps", 1)),
            "small_cutoff_len": int(evaluation_cfg.get("small_cutoff_len", 64)),
            "small_learning_rate": float(evaluation_cfg.get("small_learning_rate", 0.0002)),
            "small_num_proc": int(evaluation_cfg.get("small_num_proc", 1)),
        },
    }
    return ExperimentConfig(path=Path(output_dir / "dp_fedavg_eval.yaml"), raw=raw)


def run_downstream_eval(*, synthetic_texts: list[str], dp_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    repo_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager

    thesis_config = build_thesis_eval_config(dp_config, output_dir=output_dir)
    manager = DownstreamEvalManager(
        thesis_config,
        experiment_id=str(thesis_config.meta.get("experiment_id", "dp_fedavg")),
        output_dir=output_dir,
    )
    return manager.run(synthetic_texts)
