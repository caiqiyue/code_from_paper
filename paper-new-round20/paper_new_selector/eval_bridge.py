from __future__ import annotations

from pathlib import Path
from typing import Any

from .thesis_bridge import load_yaml_config, resolve_output_root, resolve_repo_root


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    return int(value)


def prepare_eval_runtime(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    eval_cfg = dict(config.get("eval", {}))
    if not eval_cfg:
        raise ValueError("eval section must be configured.")
    if not bool(eval_cfg.get("enabled", False)):
        return {"enabled": False, "mode": str(eval_cfg.get("mode", "disabled"))}
    mode = str(eval_cfg.get("mode", "")).strip().lower()
    if mode != "pretext_small":
        raise ValueError(f"eval.mode must be 'pretext_small', got '{mode or '<empty>'}'.")
    return {
        "enabled": True,
        "mode": mode,
        "small_eval_mode": str(eval_cfg.get("small_eval_mode", "gpt2")),
        "output_root": str(resolve_output_root(config_path)),
    }


def _build_thesis_eval_config(config_path: str | Path):
    selector_cfg = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    if str(repo_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo_root))
    from thesis_platform.core.config import ExperimentConfig

    eval_cfg = dict(selector_cfg["eval"])
    raw = {
        "meta": {
            "experiment_id": str(selector_cfg.get("meta", {}).get("experiment_id", "paper_new_selector_test")),
            "seed": int(selector_cfg.get("meta", {}).get("seed", 42)),
        },
        "paths": {
            "repo_root": str(repo_root),
            "output_root": str(selector_cfg.get("paths", {}).get("output_root", "paper-new/outputs/selector_test")),
        },
        "data": {
            "dataset_name": str(selector_cfg["data"]["dataset_name"]),
            "train_path": str(selector_cfg["data"]["train_path"]),
            "eval_path": str(selector_cfg["data"]["eval_path"]),
            "initialization_path": str(selector_cfg["data"]["initialization_path"]),
            "max_samples_per_client": int(eval_cfg.get("max_samples_per_client", 8)),
            "initialization_min_words": int(eval_cfg.get("initialization_min_words", 20)),
            "train_limit": _optional_int(selector_cfg["data"].get("train_limit")),
            "eval_limit": _optional_int(selector_cfg["data"].get("eval_limit")),
            "initialization_limit": _optional_int(selector_cfg["data"].get("initialization_limit")),
        },
        "runtime": {
            "device": str(eval_cfg.get("device", "cuda")),
        },
        "downstream_eval": {
            "enabled": True,
            "kind": "pretext_small_eval",
            "run_large_eval": False,
            "run_small_eval": True,
            "small_eval_mode": str(eval_cfg.get("small_eval_mode", "gpt2")),
            "distilgpt2_path": "thesis_platform/open_model/distilgpt2",
            "small_epochs": int(eval_cfg.get("small_epochs", 1)),
            "small_batch_size": int(eval_cfg.get("small_batch_size", 1)),
            "small_eval_batch_size": int(eval_cfg.get("small_eval_batch_size", 1)),
            "small_grad_accum_steps": int(eval_cfg.get("small_grad_accum_steps", 1)),
            "small_cutoff_len": int(eval_cfg.get("small_cutoff_len", 64)),
            "small_learning_rate": float(eval_cfg.get("small_learning_rate", 0.0002)),
            "small_num_proc": int(eval_cfg.get("small_num_proc", 1)),
        },
    }
    return ExperimentConfig(path=Path(config_path).resolve(), raw=raw)


def run_eval(*, synthetic_texts: list[str], config_path: str | Path, output_dir: Path | None = None) -> dict[str, Any]:
    runtime = prepare_eval_runtime(config_path)
    if not runtime["enabled"]:
        return runtime

    repo_root = resolve_repo_root()
    if str(repo_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo_root))
    from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager

    thesis_config = _build_thesis_eval_config(config_path)
    eval_output_dir = output_dir or (resolve_output_root(config_path) / "eval")
    manager = DownstreamEvalManager(
        thesis_config,
        experiment_id=str(thesis_config.meta.get("experiment_id", "paper_new_selector_test")),
        output_dir=eval_output_dir,
    )
    return manager.run(synthetic_texts)
