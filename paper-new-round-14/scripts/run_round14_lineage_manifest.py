#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPT_PATH = ROOT / "scripts" / "run_selector_single_node.py"
SUMMARY_FIELDS = [
    "experiment_id",
    "dataset_name",
    "budget",
    "seed",
    "status",
    "attempt",
    "duration_seconds",
    "config_path",
    "output_root",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
    "error",
]


@dataclass
class ExperimentSpec:
    experiment_id: str
    dataset_name: str
    budget: int
    meta_seed: int
    config_path: Path
    output_root: str


@dataclass
class ExperimentResult:
    returncode: int
    duration_seconds: float
    status: str
    summary: dict[str, Any]
    metrics: dict[str, float]
    stdout: str
    stderr: str


def _query_gpu_free_gb(target_gpu_index: int) -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception:
        return None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
            free_mib = float(parts[1])
        except ValueError:
            continue
        if index == target_gpu_index:
            return free_mib / 1024.0
    return None


def wait_for_gpu_ready(
    *,
    target_gpu_index: int,
    min_free_gb_for_vllm: float,
    gpu_wait_poll_seconds: int,
    gpu_wait_timeout_seconds: int,
) -> None:
    start = time.time()
    while True:
        free_gb = _query_gpu_free_gb(target_gpu_index)
        if free_gb is not None and free_gb >= min_free_gb_for_vllm:
            return
        if (time.time() - start) >= gpu_wait_timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for GPU {target_gpu_index} to reach {min_free_gb_for_vllm:.2f} GiB free memory."
            )
        time.sleep(gpu_wait_poll_seconds)


def _resolve_repo_relative(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_manifest(path: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            specs.append(
                ExperimentSpec(
                    experiment_id=row["experiment_id"],
                    dataset_name=row["dataset"],
                    budget=int(row["budget"]),
                    meta_seed=int(row["seed"]),
                    config_path=_resolve_repo_relative(row["config_path"]),
                    output_root=row["output_root"],
                )
            )
    return specs


def _extract_eval_metrics(summary: dict[str, Any]) -> tuple[str, dict[str, float]]:
    eval_summary = summary.get("eval", {})
    status = str(eval_summary.get("status", "completed")) if isinstance(eval_summary, dict) else "completed"
    metrics_raw = eval_summary.get("metrics", {}) if isinstance(eval_summary, dict) else {}
    metrics: dict[str, float] = {}
    for key in ("best_top1", "best_top3", "best_top5", "best_top10"):
        if key in metrics_raw and metrics_raw[key] not in (None, ""):
            metrics[key] = float(metrics_raw[key])
    return status, metrics


def run_single_experiment(
    spec: ExperimentSpec,
    *,
    python_executable: str,
    timeout_seconds: int,
) -> ExperimentResult:
    command = [
        python_executable,
        str(SCRIPT_PATH),
        "--config",
        str(spec.config_path),
    ]
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    duration = time.time() - start
    summary: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    status = "failed" if completed.returncode != 0 else "success"
    if completed.returncode == 0:
        try:
            summary = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            status = "failed"
            completed = subprocess.CompletedProcess(
                completed.args,
                1,
                stdout=completed.stdout,
                stderr=f"{completed.stderr}\nJSONDecodeError: {exc}",
            )
        else:
            eval_status, metrics = _extract_eval_metrics(summary)
            status = "success" if eval_status in {"completed", "success"} else eval_status
    return ExperimentResult(
        returncode=completed.returncode,
        duration_seconds=duration,
        status=status,
        summary=summary,
        metrics=metrics,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _load_existing_success_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["experiment_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("status") == "success"
        }


def _append_summary_row(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_log(log_dir: Path, experiment_id: str, attempt: int, result: ExperimentResult) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_id}.attempt{attempt}.log"
    log_path.write_text(
        "\n".join(
            [
                f"returncode={result.returncode}",
                f"status={result.status}",
                "--- STDOUT ---",
                result.stdout,
                "--- STDERR ---",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )


def run_manifest(
    *,
    manifest_path: Path,
    summary_path: Path,
    log_dir: Path,
    python_executable: str,
    timeout_seconds: int,
    max_attempts: int,
    retry_delay_seconds: int,
    reset_summary: bool,
    min_free_gb_for_vllm: float,
    gpu_wait_poll_seconds: int,
    gpu_wait_timeout_seconds: int,
    target_gpu_index: int,
) -> int:
    if reset_summary and summary_path.exists():
        summary_path.unlink()
    success_ids = _load_existing_success_ids(summary_path)
    specs = load_manifest(manifest_path)
    final_failures = 0
    for spec in specs:
        if spec.experiment_id in success_ids:
            continue
        last_result: ExperimentResult | None = None
        for attempt in range(1, max_attempts + 1):
            wait_for_gpu_ready(
                target_gpu_index=target_gpu_index,
                min_free_gb_for_vllm=min_free_gb_for_vllm,
                gpu_wait_poll_seconds=gpu_wait_poll_seconds,
                gpu_wait_timeout_seconds=gpu_wait_timeout_seconds,
            )
            result = run_single_experiment(
                spec,
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            _write_log(log_dir, spec.experiment_id, attempt, result)
            row = {
                "experiment_id": spec.experiment_id,
                "dataset_name": spec.dataset_name,
                "budget": spec.budget,
                "seed": spec.meta_seed,
                "status": result.status if result.returncode == 0 else "failed",
                "attempt": attempt,
                "duration_seconds": f"{result.duration_seconds:.3f}",
                "config_path": str(spec.config_path),
                "output_root": spec.output_root,
                "best_top1": result.metrics.get("best_top1", ""),
                "best_top3": result.metrics.get("best_top3", ""),
                "best_top5": result.metrics.get("best_top5", ""),
                "best_top10": result.metrics.get("best_top10", ""),
                "error": "" if result.returncode == 0 else (result.stderr or "non-zero exit"),
            }
            _append_summary_row(summary_path, row)
            last_result = result
            if result.returncode == 0:
                success_ids.add(spec.experiment_id)
                break
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
        else:
            final_failures += 1
            if last_result is not None:
                print(f"FINAL_FAILURE {spec.experiment_id}: {last_result.stderr}", file=sys.stderr)
    return 1 if final_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run round14-lineage E6 manifest sequentially.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=int, default=10)
    parser.add_argument("--reset-summary", action="store_true")
    parser.add_argument("--min-free-gb-for-vllm", type=float, default=2.0)
    parser.add_argument("--gpu-wait-poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-wait-timeout-seconds", type=int, default=7200)
    parser.add_argument("--target-gpu-index", type=int, default=1)
    args = parser.parse_args()
    return run_manifest(
        manifest_path=_resolve_repo_relative(args.manifest_path),
        summary_path=_resolve_repo_relative(args.summary_path),
        log_dir=_resolve_repo_relative(args.log_dir),
        python_executable=args.python_executable,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        reset_summary=args.reset_summary,
        min_free_gb_for_vllm=args.min_free_gb_for_vllm,
        gpu_wait_poll_seconds=args.gpu_wait_poll_seconds,
        gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
        target_gpu_index=args.target_gpu_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
