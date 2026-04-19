from __future__ import annotations

from pathlib import Path
from typing import Any

from pretext_platform.core.io_utils import ensure_dir, write_json
from pretext_platform.core.types import StageSummary


def ensure_experiment_dir(output_root: Path, experiment_id: str) -> Path:
    return ensure_dir(output_root / experiment_id)


def round_dir(experiment_dir: Path, round_id: int) -> Path:
    return ensure_dir(experiment_dir / f"round_{round_id:03d}")


def client_dir(experiment_dir: Path, round_id: int, client_id: str) -> Path:
    return ensure_dir(round_dir(experiment_dir, round_id) / client_id)


def server_stage2_dir(experiment_dir: Path, round_id: int) -> Path:
    return ensure_dir(round_dir(experiment_dir, round_id) / "server_stage2")


def write_stage_summary(path: Path, summary: StageSummary) -> None:
    write_json(path, summary)


def write_metrics_summary(experiment_dir: Path, payload: dict[str, Any]) -> None:
    write_json(experiment_dir / "metrics_summary.json", payload)


def write_privacy_ledger(experiment_dir: Path, payload: dict[str, Any]) -> None:
    write_json(experiment_dir / "privacy_ledger.json", payload)
