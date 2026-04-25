from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for ancestor in current.parents:
        if (
            (ancestor / "paper-new").is_dir()
            and (ancestor / "PrE-Text").is_dir()
            and (ancestor / "thesis_platform").is_dir()
        ):
            return ancestor
    raise FileNotFoundError("Could not locate repo root containing paper-new, PrE-Text, and thesis_platform.")


def ensure_repo_imports() -> Path:
    repo_root = resolve_repo_root()
    search_paths = [
        repo_root,
        repo_root / "paper-new",
        repo_root / "PrE-Text",
        repo_root / "paper-new-2",
    ]
    for path in search_paths:
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    return repo_root


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    ensure_repo_imports()
    from paper_new_selector.thesis_bridge import load_yaml_config as _load_yaml_config

    return _load_yaml_config(config_path)


def resolve_output_root(config_path: str | Path) -> Path:
    ensure_repo_imports()
    from paper_new_selector.thesis_bridge import resolve_output_root as _resolve_output_root

    return _resolve_output_root(config_path)


def build_embedder_from_config(config_path: str | Path):
    ensure_repo_imports()
    from paper_new_selector.thesis_bridge import build_embedder_from_config as _build_embedder_from_config

    return _build_embedder_from_config(config_path)


def prepare_bootstrap_runtime(config_path: str | Path) -> dict[str, Any]:
    ensure_repo_imports()
    from paper_new_selector.pretext_bridge import prepare_bootstrap_runtime as _prepare_bootstrap_runtime

    return _prepare_bootstrap_runtime(config_path)


def run_eval_selected_texts(
    config_path: str | Path,
    *,
    selected_texts: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    ensure_repo_imports()
    from paper_new_selector.eval_bridge import run_eval as _run_eval

    return _run_eval(synthetic_texts=selected_texts, config_path=config_path, output_dir=output_dir)


def build_pretext_stage1_config(config_path: str | Path):
    ensure_repo_imports()
    from pretext_platform.core.config import ExperimentConfig

    cfg = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    data_cfg = dict(cfg["data"])
    data_cfg.setdefault("max_samples_per_client", int(cfg.get("eval", {}).get("max_samples_per_client", 8)))
    data_cfg.setdefault("initialization_min_words", int(cfg.get("eval", {}).get("initialization_min_words", 20)))

    output_root = resolve_output_root(config_path)
    stage1_cfg = dict(cfg["stage1"]) | {"enabled": True}
    if sys.platform.startswith("win"):
        # PrE-Text variation.py uses a local collate_fn that is not pickle-safe on Windows.
        # Keep the server config unchanged while forcing a safe local fallback only on Windows.
        stage1_cfg["num_workers"] = 0

    raw = {
        "meta": {
            "experiment_id": str(cfg["meta"]["experiment_id"]),
            "seed": int(cfg["meta"].get("seed", 42)),
        },
        "paths": {
            "repo_root": ".",
            "output_root": str(output_root),
            "dataset_root": "thesis_platform/datasets",
            "model_root": "thesis_platform/open_model",
        },
        "data": data_cfg,
        "models": {
            "minilm_path": "thesis_platform/open_model/all_minilm_l6_v2",
            "roberta_large_path": "thesis_platform/open_model/roberta_large",
            "llama2_7b_path": "thesis_platform/open_model/llama_2_7b_hf",
            "distilgpt2_path": "thesis_platform/open_model/distilgpt2",
            "c4_checkpoint_path": "",
        },
        "stage1": stage1_cfg,
        "bootstrap": dict(cfg.get("bootstrap", {})) | {"enabled": False},
        "eval_small": {"enabled": False},
        "eval_large": {"enabled": False},
        "runtime": {
            "device": str(cfg.get("bootstrap", {}).get("device", cfg.get("eval", {}).get("device", "cuda"))),
        },
    }
    return ExperimentConfig.from_mapping(raw, base_dir=repo_root, name=Path(config_path).name)


def run_pretext_stage1(config_path: str | Path) -> dict[str, Any]:
    ensure_repo_imports()
    from pretext_platform.algorithms.bootstrap import load_surviving_seed_texts
    from pretext_platform.core.pipeline import run_stage1

    config = build_pretext_stage1_config(config_path)
    summary = run_stage1(config)
    stage1_dir = Path(summary.output_dir)
    seed_texts = load_surviving_seed_texts(stage1_dir, num_rounds=int(config.stage1.get("rounds", 11)))
    return {
        "stage1_dir": stage1_dir,
        "seed_texts": seed_texts,
        "summary": summary,
    }
