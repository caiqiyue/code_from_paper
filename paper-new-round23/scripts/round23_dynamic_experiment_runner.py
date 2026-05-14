#!/usr/bin/env python3
"""Sequential runner for round23 dynamic-controller experiments."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROUND23_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROUND23_ROOT / "scripts" / "run_round23_with_dynamic_controller.py"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_name: str
    meta_seed: int
    config_path: Path
    output_root: str


def normalize_output_root(raw_output_root: str) -> Path:
    candidate = Path(str(raw_output_root).replace("\\", "/"))
    parts = list(candidate.parts)
    if parts and parts[0] == ROUND23_ROOT.name:
        candidate = Path(*parts[1:])
    return candidate


def load_manifest(manifest_path: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            specs.append(
                ExperimentSpec(
                    experiment_id=str(row["experiment_id"]),
                    dataset_name=str(row["dataset"]),
                    meta_seed=int(row["seed"]),
                    config_path=Path(str(row["config_path"])).resolve(),
                    output_root=str(row["output_root"]),
                )
            )
    return specs


def initialize_tsv(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def append_tsv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(row)


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_ids(summary_tsv: Path, summary_jsonl: Path) -> set[str]:
    completed: set[str] = set()
    for path, is_tsv in ((summary_tsv, True), (summary_jsonl, False)):
        if not path.exists():
            continue
        if is_tsv:
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    if str(row.get("status", "")).lower() == "success":
                        completed.add(str(row["experiment_id"]))
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if str(row.get("status", "")).lower() == "success":
                        completed.add(str(row["experiment_id"]))
    return completed


def run_single_experiment(
    spec: ExperimentSpec,
    *,
    model_dir: Path,
    timeout_seconds: int,
    log_dir: Path,
) -> tuple[int, str, str, float]:
    started = time.time()
    log_path = log_dir / f"{spec.experiment_id}.log"
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--config",
        str(spec.config_path),
        "--model-dir",
        str(model_dir),
        "--output-root",
        str((ROUND23_ROOT / normalize_output_root(spec.output_root)).resolve()),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    result = subprocess.run(
        command,
        cwd=str(ROUND23_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds + 300,
    )
    log_path.write_text(result.stdout + "\n\nSTDERR:\n" + result.stderr, encoding="utf-8")
    return result.returncode, result.stdout, result.stderr, time.time() - started


def build_summary_row(
    spec: ExperimentSpec,
    *,
    status: str,
    attempt: int,
    duration_seconds: float,
) -> dict[str, Any]:
    normalized_output_root = normalize_output_root(spec.output_root)
    sidecar_path = (
        ROUND23_ROOT / normalized_output_root / f"{spec.experiment_id}_dynamic_controller_runtime.json"
    ).resolve()
    row: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "dataset_name": spec.dataset_name,
        "meta_seed": spec.meta_seed,
        "status": status,
        "attempt": attempt,
        "duration_seconds": round(duration_seconds, 3),
        "predicted_delta_k": "",
        "predicted_target_budget": "",
        "best_top1": "",
        "config_path": str(spec.config_path),
        "output_root": str(normalized_output_root),
        "sidecar_path": str(sidecar_path),
    }
    if sidecar_path.exists():
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        row["predicted_delta_k"] = payload.get("predicted_delta_k", "")
        row["predicted_target_budget"] = payload.get("predicted_target_budget", "")
        eval_summary = (
            payload.get("runtime_artifacts", {}).get("eval_summary")
            or {}
        )
        if "best_top1" in eval_summary:
            row["best_top1"] = eval_summary["best_top1"]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round23 dynamic-controller experiments sequentially")
    parser.add_argument("--mode", choices=["real_smoke", "quick_compare"], required=True)
    parser.add_argument("--model-dir", required=True, help="Path to round23 controller bundle")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments without executing them")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on pending experiments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = (
        ROUND23_ROOT / "configs" / "experiments" / "single_node_tuning_round23_dynamic" /
        ("real_smoke/round23_real_smoke_manifest.tsv" if args.mode == "real_smoke" else "quick_compare_repeat30/round23_quick_compare_repeat30_manifest.tsv")
    )
    specs = load_manifest(manifest)
    logs_root = ROUND23_ROOT / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    per_exp_log_dir = logs_root / f"round23_{args.mode}_logs"
    per_exp_log_dir.mkdir(parents=True, exist_ok=True)
    master_log = logs_root / f"round23_{args.mode}_master.log"
    summary_tsv = logs_root / f"round23_{args.mode}_summary.tsv"
    summary_jsonl = logs_root / f"round23_{args.mode}_summary.jsonl"
    fields = [
        "experiment_id",
        "dataset_name",
        "meta_seed",
        "status",
        "attempt",
        "duration_seconds",
        "predicted_delta_k",
        "predicted_target_budget",
        "best_top1",
        "config_path",
        "output_root",
        "sidecar_path",
    ]
    initialize_tsv(summary_tsv, fields)
    done = completed_ids(summary_tsv, summary_jsonl)
    pending = [spec for spec in specs if spec.experiment_id not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        print(json.dumps(
            {
                "mode": args.mode,
                "pending_count": len(pending),
                "first_experiments": [spec.experiment_id for spec in pending[:10]],
                "model_dir": str(Path(args.model_dir).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    with master_log.open("a", encoding="utf-8") as master:
        for spec in pending:
            success = False
            for attempt in range(1, args.max_attempts + 1):
                master.write(f"{datetime.now().isoformat()} START {spec.experiment_id} attempt={attempt}\n")
                master.flush()
                code, _, stderr, duration = run_single_experiment(
                    spec,
                    model_dir=Path(args.model_dir).resolve(),
                    timeout_seconds=args.timeout_seconds,
                    log_dir=per_exp_log_dir,
                )
                status = "success" if code == 0 else "failed"
                row = build_summary_row(
                    spec,
                    status=status,
                    attempt=attempt,
                    duration_seconds=duration,
                )
                append_tsv_row(summary_tsv, fields, row)
                append_jsonl_row(summary_jsonl, row)
                master.write(
                    f"{datetime.now().isoformat()} END {spec.experiment_id} status={status} duration={duration:.2f}s\n"
                )
                master.flush()
                if code == 0:
                    success = True
                    break
                if attempt < args.max_attempts:
                    master.write(f"{datetime.now().isoformat()} RETRY {spec.experiment_id} stderr={stderr[:400]}\n")
                    master.flush()
                    time.sleep(10)
            if not success:
                master.write(f"{datetime.now().isoformat()} STOP_ON_FAILURE {spec.experiment_id}\n")
                master.flush()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
