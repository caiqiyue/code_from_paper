from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

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
