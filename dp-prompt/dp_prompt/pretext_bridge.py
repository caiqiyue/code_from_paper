from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_thesis_platform_importable(repo_root: Path | None = None) -> None:
    root = repo_root or resolve_repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def build_thesis_eval_config(
    *,
    experiment_id: str,
    dataset_name: str,
    train_path: str,
    eval_path: str,
    initialization_path: str,
    output_root: str,
    repo_root: Path | None = None,
    distilgpt2_path: str = "thesis_platform/open_model/distilgpt2",
    llama2_7b_path: str = "thesis_platform/open_model/llama_2_7b_hf",
    linux_small_eval_mode: str = "gpt2",
) -> Any:
    root = repo_root or resolve_repo_root()
    ensure_thesis_platform_importable(root)
    from thesis_platform.core.config import ExperimentConfig

    raw = {
        "meta": {
            "experiment_id": experiment_id,
            "seed": 42,
        },
        "paths": {
            "repo_root": str(root),
            "output_root": output_root,
        },
        "data": {
            "dataset_name": dataset_name,
            "train_path": train_path,
            "eval_path": eval_path,
            "initialization_path": initialization_path,
            "max_samples_per_client": 16,
            "initialization_min_words": 20,
            "sample_format": "raw_text",
            "task_type": "instruction_tuning",
        },
        "downstream_eval": {
            "enabled": True,
            "run_large_eval": False,
            "run_small_eval": True,
            "small_eval_mode": linux_small_eval_mode,
            "windows_small_eval_mode": "gpt2",
            "linux_small_eval_mode": linux_small_eval_mode,
            "llama2_7b_path": llama2_7b_path,
            "distilgpt2_path": distilgpt2_path,
        },
    }
    return ExperimentConfig(path=root / "dp-prompt" / "synthetic_eval_config.yaml", raw=raw)


def export_and_run_small_eval(
    thesis_config: Any,
    *,
    synthetic_texts: list[str],
    stage2_dir: Path,
    eval_dir: Path,
) -> dict[str, Any]:
    ensure_thesis_platform_importable(resolve_repo_root())
    from thesis_platform.evaluation.downstream_eval import export_synthetic_corpus, run_pretext_small_eval

    export_synthetic_corpus(
        synthetic_texts,
        output_dir=stage2_dir,
        filename="llama7b_text_syn.json",
    )
    return run_pretext_small_eval(thesis_config, stage2_dir=stage2_dir, output_dir=eval_dir)
