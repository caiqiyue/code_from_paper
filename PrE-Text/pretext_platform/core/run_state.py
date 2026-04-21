from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.io_utils import write_json


class PretextFailure(RuntimeError):
    """Failure with a stable code that can be persisted for automation."""

    def __init__(
        self,
        failure_code: str,
        message: str,
        *,
        phase: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.phase = phase
        self.details = dict(details or {})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_kind(config: ExperimentConfig) -> str:
    mode = str(config.execution.get("mode", "")).strip().lower()
    return "federated_formal" if mode == "federated_pretext" else "single_node_formal"


def is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "outofmemoryerror" in text or ("cuda" in text and "out of memory" in text)


def build_error_payload(exc: BaseException, *, phase: str) -> dict[str, Any]:
    if isinstance(exc, PretextFailure):
        return {
            "failure_code": exc.failure_code,
            "message": str(exc),
            "phase": exc.phase,
            "error_type": type(exc).__name__,
            "details": dict(exc.details),
        }
    failure_code = "stage2_runtime_gpu_oom" if phase == "stage2" and is_cuda_oom(exc) else "pretext_runtime_error"
    return {
        "failure_code": failure_code,
        "message": str(exc),
        "phase": phase,
        "error_type": type(exc).__name__,
        "details": {},
    }


def write_run_state(
    experiment_dir: Path,
    config: ExperimentConfig,
    *,
    status: str,
    phase: str,
    last_error: dict[str, Any] | None = None,
) -> None:
    write_json(
        experiment_dir / "run_state.json",
        {
            "experiment_id": config.experiment_id(),
            "status": status,
            "phase": phase,
            "updated_at": _now_iso(),
            "last_error": last_error,
            "run_kind": _run_kind(config),
            "artifacts": {
                "metrics_summary_path": str(experiment_dir / "metrics_summary.json"),
                "failure_summary_path": str(experiment_dir / "failure_summary.json"),
            },
        },
    )


def write_failure_artifacts(
    experiment_dir: Path,
    config: ExperimentConfig,
    exc: BaseException,
    *,
    phase: str,
) -> dict[str, Any]:
    error_payload = build_error_payload(exc, phase=phase)
    failure_phase = str(error_payload.get("phase") or phase)
    failure_payload = {
        "experiment_id": config.experiment_id(),
        "status": "failed",
        "phase": failure_phase,
        "failure_code": error_payload["failure_code"],
        "message": error_payload["message"],
        "details": error_payload.get("details", {}),
        "error_type": error_payload.get("error_type"),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "updated_at": _now_iso(),
    }
    metrics_payload = {
        "experiment_id": config.experiment_id(),
        "experiment_dir": str(experiment_dir),
        "status": "failed",
        "phase": failure_phase,
        "failure_code": error_payload["failure_code"],
        "run_state_path": str(experiment_dir / "run_state.json"),
        "failure_summary_path": str(experiment_dir / "failure_summary.json"),
        "last_error": error_payload,
    }
    write_json(experiment_dir / "failure_summary.json", failure_payload)
    write_json(experiment_dir / "metrics_summary.json", metrics_payload)
    write_run_state(experiment_dir, config, status="failed", phase=failure_phase, last_error=error_payload)
    return error_payload
