from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from thesis_platform.core.io_utils import ensure_dir


def package_root() -> Path:
    """Return the `thesis_platform` package root."""

    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the repository root that contains `thesis_platform`."""

    return package_root().parent


def datasets_root() -> Path:
    """Return the shared dataset download directory."""

    return ensure_dir(package_root() / "datasets")


def to_package_relative(path: Path) -> str:
    """Render a path relative to `thesis_platform` when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(package_root().resolve()).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(repo_root().resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()


def optional_package_relative(path: Path | None) -> str | None:
    """Render an optional path relative to the package root."""

    if path is None:
        return None
    return to_package_relative(path)


def utc_timestamp() -> str:
    """Return an RFC3339-style UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def remove_path(path: Path) -> None:
    """Delete a file or directory tree when it exists."""

    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def copy_file(source: Path, target: Path) -> None:
    """Copy one file while creating missing parent directories."""

    ensure_dir(target.parent)
    shutil.copy2(source, target)


def move_path(source: Path, target: Path) -> None:
    """Move a file or directory while creating missing parent directories."""

    ensure_dir(target.parent)
    shutil.move(str(source), str(target))


def _summarize_split_counts(split_counts: dict[str, int]) -> dict[str, Any] | None:
    """Return one normalized split-count payload."""

    normalized = {name: int(count) for name, count in split_counts.items()}
    if not normalized:
        return None
    return {
        "splits": normalized,
        "total": int(sum(normalized.values())),
    }


def _infer_single_dataset_split_name(path: Path) -> str:
    """Return a stable split label for one dataset artifact."""

    if path.suffix == ".hf":
        return path.stem
    if path.name in {"raw", "formatted"}:
        return "dataset"
    return path.stem or path.name


def _looks_like_huggingface_artifact(path: Path) -> bool:
    """Return whether one path resembles a `datasets.save_to_disk` artifact."""

    if not path.exists():
        return False
    if path.is_file():
        return path.suffix in {".arrow", ".parquet"}
    marker_names = {
        "dataset_info.json",
        "state.json",
        "dataset_dict.json",
    }
    if any((path / marker_name).exists() for marker_name in marker_names):
        return True
    return any(child.suffix in {".arrow", ".parquet"} for child in path.iterdir() if child.is_file())


def _inspect_huggingface_artifact(path: Path) -> dict[str, Any] | None:
    """Inspect one Hugging Face `save_to_disk` artifact when possible."""

    if not _looks_like_huggingface_artifact(path):
        return None

    try:
        from datasets import DatasetDict, load_from_disk
    except Exception:
        return None

    try:
        dataset = load_from_disk(str(path))
    except Exception:
        return None

    if isinstance(dataset, DatasetDict):
        return _summarize_split_counts({split_name: len(split) for split_name, split in dataset.items()})
    return _summarize_split_counts({_infer_single_dataset_split_name(path): len(dataset)})


def _count_jsonl_rows(path: Path) -> int:
    """Count JSONL records without loading the entire file into memory."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _count_csv_rows(path: Path) -> int:
    """Count CSV data rows while excluding the header row."""

    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)
    return max(0, row_count - 1)


def _count_known_json_rows(path: Path) -> int | None:
    """Count samples for supported JSON layouts such as BBH raw files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return len(payload["examples"])
    if isinstance(payload, dict):
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if list_values and len(list_values) == len(payload):
            return sum(len(value) for value in list_values)
    if isinstance(payload, list):
        return len(payload)
    return None


def inspect_sample_counts(path: Path | None) -> dict[str, Any] | None:
    """Infer sample counts from a dataset artifact path."""

    if path is None or not path.exists():
        return None

    huggingface_counts = _inspect_huggingface_artifact(path)
    if huggingface_counts is not None:
        return huggingface_counts

    if path.is_file():
        if path.suffix == ".jsonl":
            return _summarize_split_counts({path.stem: _count_jsonl_rows(path)})
        if path.suffix == ".csv":
            return _summarize_split_counts({path.stem: _count_csv_rows(path)})
        if path.suffix == ".json":
            count = _count_known_json_rows(path)
            if count is not None:
                return _summarize_split_counts({path.stem: count})
        return None

    split_counts: dict[str, int] = {}
    for child in sorted(path.iterdir()):
        child_counts = _inspect_huggingface_artifact(child)
        if child_counts is not None:
            split_counts.update(child_counts["splits"])
            continue
        if child.is_file() and child.suffix == ".jsonl":
            split_counts[child.stem] = _count_jsonl_rows(child)
            continue
        if child.is_file() and child.suffix == ".csv":
            split_counts[child.stem] = _count_csv_rows(child)
            continue
        if child.is_file() and child.suffix == ".json":
            count = _count_known_json_rows(child)
            if count is not None:
                split_counts[child.stem] = count

    return _summarize_split_counts(split_counts)
