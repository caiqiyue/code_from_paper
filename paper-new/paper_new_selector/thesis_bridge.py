from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_resource_root(root: Path) -> bool:
    return (
        (root / "thesis_platform").is_dir()
        and (root / "PrE-Text").is_dir()
        and (root / "thesis_platform" / "datasets").exists()
        and (root / "thesis_platform" / "open_model").exists()
    )


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for ancestor in current.parents:
        if _is_resource_root(ancestor):
            return ancestor
        if ancestor.name == ".worktrees":
            candidate = ancestor.parent
            if _is_resource_root(candidate):
                return candidate
    raise FileNotFoundError("Could not locate the repo root with thesis_platform datasets/open_model resources.")


def resolve_worktree_root() -> Path:
    current = Path(__file__).resolve()
    for ancestor in current.parents:
        if (ancestor / ".git").exists() and (ancestor / "thesis_platform").is_dir():
            return ancestor
    return resolve_repo_root()


def _resolve_relative_to_root(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def resolve_dataset_paths(config_path: str | Path) -> tuple[Path, Path, Path]:
    config = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    data_cfg = config["data"]
    train_path = _resolve_relative_to_root(repo_root, str(data_cfg["train_path"]))
    eval_path = _resolve_relative_to_root(repo_root, str(data_cfg["eval_path"]))
    init_path = _resolve_relative_to_root(repo_root, str(data_cfg["initialization_path"]))
    return train_path, eval_path, init_path


def resolve_models_root(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    return _resolve_relative_to_root(repo_root, str(config["paths"]["models_root"]))


def resolve_output_root(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    output_cfg = str(config.get("paths", {}).get("output_root", "paper-new/outputs/selector_test"))
    return _resolve_relative_to_root(repo_root, output_cfg)


def build_embedder_from_config(config_path: str | Path):
    config = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from thesis_platform.models.embedding import build_embedder

    embedding_cfg = dict(config.get("embedding", {}))
    model_path = str(embedding_cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("embedding.model_path must be configured.")
    return build_embedder(
        model_path,
        repo_root=repo_root,
        allow_fallback=False,
        device=str(embedding_cfg.get("device", "cpu")),
    )


def load_text_samples(config_path: str | Path) -> dict[str, Any]:
    repo_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from thesis_platform.data.loaders import load_samples

    config = load_yaml_config(config_path)
    train_path, eval_path, init_path = resolve_dataset_paths(config_path)
    dataset_name = str(config["data"]["dataset_name"])
    train_samples = load_samples(
        train_path,
        dataset_name=dataset_name,
        source="private_train",
        task_type="raw_text",
        round_id=0,
        client_id="single_node",
        prefix="train",
        limit=int(config["data"]["train_limit"]) if config["data"].get("train_limit") not in (None, "") else None,
    )
    eval_samples = load_samples(
        eval_path,
        dataset_name=dataset_name,
        source="private_eval",
        task_type="raw_text",
        round_id=0,
        client_id="single_node",
        prefix="eval",
        limit=int(config["data"]["eval_limit"]) if config["data"].get("eval_limit") not in (None, "") else None,
    )
    init_samples = load_samples(
        init_path,
        dataset_name=dataset_name,
        source="public_init",
        task_type="raw_text",
        round_id=0,
        client_id="server",
        prefix="init",
        limit=int(config["data"]["initialization_limit"]) if config["data"].get("initialization_limit") not in (None, "") else None,
    )
    return {
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "init_samples": init_samples,
        "dataset_name": dataset_name,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
