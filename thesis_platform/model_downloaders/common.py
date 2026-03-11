from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from thesis_platform.core.io_utils import ensure_dir


def package_root() -> Path:
    """Return the `thesis_platform` package root."""

    return Path(__file__).resolve().parents[1]


def models_root() -> Path:
    """Return the shared model download directory."""

    return ensure_dir(package_root() / "open_model")


def to_package_relative(path: Path) -> str:
    """Render a path relative to `thesis_platform` when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(package_root().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
