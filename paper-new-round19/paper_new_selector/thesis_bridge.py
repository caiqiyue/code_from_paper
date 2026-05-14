from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - keep config loading working in constrained environments
    yaml = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} must decode to a mapping.")
        return data
    data = json.loads(json.dumps(_load_yaml_without_dependency(text)))
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must decode to a mapping.")
    return data


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    comment_start = -1
    in_single_quote = False
    in_double_quote = False
    for i, ch in enumerate(value):
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif ch == "#" and not in_single_quote and not in_double_quote:
            comment_start = i
            break
    if comment_start >= 0:
        value = value[:comment_start].strip()

    if value == "":
        return ""
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value.replace("'", '"'))
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    try:
        return float(value) if any(ch in value for ch in ".eE") else value
    except ValueError:
        return value


def _load_yaml_without_dependency(text: str) -> dict[str, Any]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        mapping: dict[str, Any] = {}
        sequence: list[Any] | None = None
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if content.startswith("- "):
                if sequence is None:
                    sequence = []
                item = content[2:].strip()
                if item:
                    sequence.append(_parse_scalar(item))
                    index += 1
                else:
                    child, index = parse_block(index + 1, current_indent + 2)
                    sequence.append(child)
                continue

            key, _, raw_value = content.partition(":")
            if not _:
                raise ValueError(f"Invalid YAML line: {content}")
            if raw_value.strip():
                mapping[key.strip()] = _parse_scalar(raw_value)
                index += 1
            else:
                child, index = parse_block(index + 1, current_indent + 2)
                mapping[key.strip()] = child
        return (sequence if sequence is not None else mapping), index

    parsed, _ = parse_block(0, 0)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML object must be a mapping.")
    return parsed


def _load_with_includes(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    includes = data.pop("inherits", []) or []
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (path.parent / include).resolve()
        merged = _deep_merge(merged, _load_with_includes(include_path))
    return _deep_merge(merged, data)


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


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file does not exist: {resolved}")
        return resolved

    candidates: list[Path] = []
    cwd_candidate = (Path.cwd() / path).resolve()
    candidates.append(cwd_candidate)

    worktree_root = resolve_worktree_root()
    candidates.append((worktree_root / path).resolve())

    repo_root = resolve_repo_root()
    candidates.append((repo_root / path).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve config path. Checked: "
        + ", ".join(str(candidate) for candidate in seen)
    )


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    return _load_with_includes(path)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(row, ensure_ascii=False) for row in rows
    )
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
