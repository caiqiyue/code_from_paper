#!/usr/bin/env python3
"""Sequential runner for PrE-Text screening on unseen2 datasets (imdb, openreview).

Reads the manifest TSV, executes each experiment using paper-new-round19's
run_selector_single_node module (which supports expand_private / PrE-Text mode),
retries on failure, and appends results to the summary TSV.

Usage:
    python paper-new-round-14/scripts/run_pretext_screening_unseen2_runner.py \
        [--manifest-path MANIFEST] [--summary-path SUMMARY] [--log-dir LOG_DIR] \
        [--python-executable PYTHON] [--max-attempts N] [--retry-delay-seconds N] \
        [--reset-summary] [--target-gpu-index N] [--min-free-gb-for-vllm N]
"""

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

# Use paper-new-round19's run_selector_single_node which supports expand_private (PrE-Text)
ROUND19_ROOT = REPO_ROOT / "paper-new-round19"
ROUND19_SCRIPT = ROUND19_ROOT / "scripts" / "run_selector_single_node.py"

SUMMARY_FIELDS = [
    "experiment_id",
    "method",
    "method_display_name",
    "dataset",
    "seed",
    "status",
    "attempt",
    "duration_seconds",
    "config_path",
    "output_root",
    "implementation_key",
    "pretext_template_key",
    "mapping_status",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
    "error",
]


@dataclass
class ExperimentSpec:
    experiment_id: str
    method: str
    method_display_name: str
    dataset: str
    seed: int
    config_path: Path
    output_root: str
    implementation_key: str
    pretext_template_key: str
    mapping_status: str


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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
                f"Timed out waiting for GPU {target_gpu_index} to reach "
                f"{min_free_gb_for_vllm:.2f} GiB free memory."
            )
        time.sleep(gpu_wait_poll_seconds)


def load_manifest(path: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            specs.append(
                ExperimentSpec(
                    experiment_id=row["experiment_id"],
                    method=row["method"],
                    method_display_name=row["method_display_name"],
                    dataset=row["dataset"],
                    seed=int(row["seed"]),
                    config_path=_resolve(row["config_path"]),
                    output_root=row["output_root"],
                    implementation_key=row.get("implementation_key", ""),
                    pretext_template_key=row.get("pretext_template_key", ""),
                    mapping_status=row.get("mapping_status", ""),
                )
            )
    return specs


def _load_eval_metrics(output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    for candidate in (
        eval_dir / "downstream_eval_summary.json",
        eval_dir / "summary.json",
    ):
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            metrics = payload.get("metrics") or {}
            return {k: metrics.get(k, "") for k in ("best_top1", "best_top3", "best_top5", "best_top10")}
    return {k: "" for k in ("best_top1", "best_top3", "best_top5", "best_top10")}


def run_single(
    spec: ExperimentSpec,
    *,
    python_executable: str,
    timeout_seconds: int,
) -> tuple[int, float, dict[str, Any]]:
    command = [
        python_executable,
        str(ROUND19_SCRIPT),
        "--config",
        str(spec.config_path),
    ]
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(ROUND19_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    duration = time.time() - start
    return completed.returncode, duration, {"stdout": completed.stdout, "stderr": completed.stderr}


def _load_success_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["experiment_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("status") == "success"
        }


def _append_row(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_log(log_dir: Path, experiment_id: str, attempt: int, io: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_id}.attempt{attempt}.log"
    log_path.write_text(
        "\n".join([
            f"--- STDOUT ---",
            io.get("stdout", ""),
            f"--- STDERR ---",
            io.get("stderr", ""),
        ]),
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

    success_ids = _load_success_ids(summary_path)
    specs = load_manifest(manifest_path)
    final_failures = 0

    for spec in specs:
        if spec.experiment_id in success_ids:
            print(f"SKIP {spec.experiment_id} (already succeeded)", flush=True)
            continue

        output_root = _resolve(spec.output_root)
        last_io: dict[str, Any] = {}
        succeeded = False

        for attempt in range(1, max_attempts + 1):
            wait_for_gpu_ready(
                target_gpu_index=target_gpu_index,
                min_free_gb_for_vllm=min_free_gb_for_vllm,
                gpu_wait_poll_seconds=gpu_wait_poll_seconds,
                gpu_wait_timeout_seconds=gpu_wait_timeout_seconds,
            )

            print(
                f"START {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} "
                f"attempt={attempt}/{max_attempts}",
                flush=True,
            )

            returncode, duration, last_io = run_single(
                spec,
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            _write_log(log_dir, spec.experiment_id, attempt, last_io)

            status = "success" if returncode == 0 else "failed"
            metrics = _load_eval_metrics(output_root) if returncode == 0 else {
                k: "" for k in ("best_top1", "best_top3", "best_top5", "best_top10")
            }

            _append_row(
                summary_path,
                {
                    "experiment_id": spec.experiment_id,
                    "method": spec.method,
                    "method_display_name": spec.method_display_name,
                    "dataset": spec.dataset,
                    "seed": spec.seed,
                    "status": status,
                    "attempt": attempt,
                    "duration_seconds": f"{duration:.3f}",
                    "config_path": str(spec.config_path),
                    "output_root": spec.output_root,
                    "implementation_key": spec.implementation_key,
                    "pretext_template_key": spec.pretext_template_key,
                    "mapping_status": spec.mapping_status,
                    "best_top1": metrics.get("best_top1", ""),
                    "best_top3": metrics.get("best_top3", ""),
                    "best_top5": metrics.get("best_top5", ""),
                    "best_top10": metrics.get("best_top10", ""),
                    "error": "" if returncode == 0 else (last_io.get("stderr") or "non-zero exit")[:400],
                },
            )

            print(
                f"END {spec.experiment_id} status={status} returncode={returncode} "
                f"duration={duration:.1f}s",
                flush=True,
            )

            if returncode == 0:
                succeeded = True
                break

            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)

        if not succeeded:
            final_failures += 1
            print(
                f"FINAL_FAILURE {spec.experiment_id}: {last_io.get('stderr', '')[:200]}",
                file=sys.stderr,
                flush=True,
            )

    return 1 if final_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PrE-Text screening on unseen2 datasets (imdb, openreview) sequentially."
    )
    parser.add_argument(
        "--manifest-path",
        default="paper-new-round-14/configs/experiments/pretext_screening_unseen2/pretext_screening_unseen2_manifest.tsv",
    )
    parser.add_argument(
        "--summary-path",
        default="paper-new-round-14/logs/pretext_screening_15rounds_unseen2_summary.tsv",
    )
    parser.add_argument(
        "--log-dir",
        default="paper-new-round-14/logs/pretext_screening_unseen2_logs",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=int, default=10)
    parser.add_argument("--reset-summary", action="store_true")
    parser.add_argument("--min-free-gb-for-vllm", type=float, default=2.0)
    parser.add_argument("--gpu-wait-poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-wait-timeout-seconds", type=int, default=43200)
    parser.add_argument("--target-gpu-index", type=int, default=1)
    args = parser.parse_args()

    return run_manifest(
        manifest_path=_resolve(args.manifest_path),
        summary_path=_resolve(args.summary_path),
        log_dir=_resolve(args.log_dir),
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
