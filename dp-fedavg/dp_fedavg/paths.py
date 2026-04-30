from __future__ import annotations

from pathlib import Path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_root() -> Path:
    return resolve_project_root().parent


def resolve_path_from_repo(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (resolve_repo_root() / path).resolve()
