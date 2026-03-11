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


def compute_path_size_bytes(path: Path | None) -> int:
    """Return the total disk usage for one file or directory tree."""

    if path is None or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_bytes(num_bytes: int) -> str:
    """Render one byte count using binary units."""

    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"
