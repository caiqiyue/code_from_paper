#!/usr/bin/env python3
"""Sequential runner for formal round23 dynamic-controller experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from round23_runtime_utils import (
    DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE,
    DEFAULT_ROUND23_CONTROLLER_SCOPE,
    extract_eval_metric,
)


ROUND23_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPTS = {
    "round23": ROUND23_ROOT / "scripts" / "run_round23_with_dynamic_controller.py",
    "round23_absk_oneshot": ROUND23_ROOT / "scripts" / "run_round23_with_absolute_k_controller.py",
    "round23_keepk0": ROUND23_ROOT / "scripts" / "run_round23_keep_k0_baseline.py",
    "round23_3round_stress": ROUND23_ROOT / "scripts" / "run_round23_with_three_round_stress.py",
    "e1_c4only": ROUND23_ROOT / "scripts" / "run_e1_c4only_baseline.py",
    "e1_augpe": ROUND23_ROOT / "scripts" / "run_e1_augpe_baseline.py",
    "e1_dpprompt": ROUND23_ROOT / "scripts" / "run_e1_dpprompt_baseline.py",
}
DEFAULT_TARGET_GPU_NAME_TOKEN = "RTX A6000"
DEFAULT_MIN_FREE_GB_FOR_VLLM = 2.0
DEFAULT_GPU_WAIT_POLL_SECONDS = 30
DEFAULT_GPU_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60
RETRYABLE_RESOURCE_FAILURE_MARKERS = (
    "No available memory for the cache blocks",
    "vllm_runtime_gpu_oom",
    "CUDA out of memory",
)
MODE_PATHS = {
    "real_smoke": {
        "manifest_relpath": "real_smoke/round23_real_smoke_manifest.tsv",
        "log_stem": "round23_real_smoke",
        "dataset_split": "seen",
    },
    "quick_compare": {
        "manifest_relpath": "quick_compare_repeat30/round23_quick_compare_repeat30_manifest.tsv",
        "log_stem": "round23_quick_compare_repeat30",
        "dataset_split": "seen",
    },
    "unseen_dataset_final_eval": {
        "manifest_relpath": (
            "unseen_dataset_final_eval_repeat40/"
            "round23_unseen_dataset_final_eval_repeat40_manifest.tsv"
        ),
        "log_stem": "round23_unseen_dataset_final_eval_repeat40",
        "dataset_split": "unseen",
    },
    "thesis_main_seen_pilot": {
        "manifest_relpath": "thesis_main_seen_pilot/round23_thesis_main_seen_pilot_manifest.tsv",
        "log_stem": "round23_thesis_main_seen_pilot",
        "dataset_split": "seen",
    },
    "thesis_main_seen_smoke": {
        "manifest_relpath": "thesis_main_seen_smoke/round23_thesis_main_seen_smoke_manifest.tsv",
        "log_stem": "round23_thesis_main_seen_smoke",
        "dataset_split": "seen",
    },
    "thesis_main_seen_repeat10": {
        "manifest_relpath": "thesis_main_seen_repeat10/round23_thesis_main_seen_repeat10_manifest.tsv",
        "log_stem": "round23_thesis_main_seen_repeat10",
        "dataset_split": "seen",
    },
    "thesis_main_seen_repeat15": {
        "manifest_relpath": "thesis_main_seen_repeat15/round23_thesis_main_seen_repeat15_manifest.tsv",
        "log_stem": "round23_thesis_main_seen_repeat15",
        "dataset_split": "seen",
    },
    "thesis_main_controller_dev_extra_repeat15": {
        "manifest_relpath": (
            "thesis_main_controller_dev_extra_repeat15/"
            "round23_thesis_main_controller_dev_extra_repeat15_manifest.tsv"
        ),
        "log_stem": "round23_thesis_main_controller_dev_extra_repeat15",
        "dataset_split": "controller_dev_extra",
    },
    "thesis_main_seen_repeat30": {
        "manifest_relpath": "thesis_main_seen_repeat30/round23_thesis_main_seen_repeat30_manifest.tsv",
        "log_stem": "round23_thesis_main_seen_repeat30",
        "dataset_split": "seen",
    },
    "thesis_e2_extra_unseen_smoke": {
        "manifest_relpath": (
            "thesis_e2_extra_unseen_smoke/"
            "round23_thesis_e2_extra_unseen_smoke_manifest.tsv"
        ),
        "log_stem": "round23_thesis_e2_extra_unseen_smoke",
        "dataset_split": "extra_unseen",
    },
    "thesis_e2_extra_unseen_repeat15": {
        "manifest_relpath": (
            "thesis_e2_extra_unseen_repeat15/"
            "round23_thesis_e2_extra_unseen_repeat15_manifest.tsv"
        ),
        "log_stem": "round23_thesis_e2_extra_unseen_repeat15",
        "dataset_split": "extra_unseen",
    },
    "e4_a_oneshot_seen_smoke": {
        "manifest_relpath": "e4_a_oneshot_seen_smoke/round23_e4_a_oneshot_seen_smoke_manifest.tsv",
        "log_stem": "round23_e4_a_oneshot_seen_smoke",
        "dataset_split": "seen",
    },
    "e4_a_oneshot_all6_smoke": {
        "manifest_relpath": "e4_a_oneshot_all6_smoke/round23_e4_a_oneshot_all6_smoke_manifest.tsv",
        "log_stem": "round23_e4_a_oneshot_all6_smoke",
        "dataset_split": "all6",
    },
    "e4_a_oneshot_seen_repeat15": {
        "manifest_relpath": "e4_a_oneshot_seen_repeat15/round23_e4_a_oneshot_seen_repeat15_manifest.tsv",
        "log_stem": "round23_e4_a_oneshot_seen_repeat15",
        "dataset_split": "seen",
    },
    "e4_a_oneshot_all6_repeat15": {
        "manifest_relpath": "e4_a_oneshot_all6_repeat15/round23_e4_a_oneshot_all6_repeat15_manifest.tsv",
        "log_stem": "round23_e4_a_oneshot_all6_repeat15",
        "dataset_split": "all6",
    },
    "e4_b_keepk0_seen_smoke": {
        "manifest_relpath": "e4_b_keepk0_seen_smoke/round23_e4_b_keepk0_seen_smoke_manifest.tsv",
        "log_stem": "round23_e4_b_keepk0_seen_smoke",
        "dataset_split": "seen",
    },
    "e4_b_keepk0_all6_smoke": {
        "manifest_relpath": "e4_b_keepk0_all6_smoke/round23_e4_b_keepk0_all6_smoke_manifest.tsv",
        "log_stem": "round23_e4_b_keepk0_all6_smoke",
        "dataset_split": "all6",
    },
    "e4_b_keepk0_seen_repeat15": {
        "manifest_relpath": "e4_b_keepk0_seen_repeat15/round23_e4_b_keepk0_seen_repeat15_manifest.tsv",
        "log_stem": "round23_e4_b_keepk0_seen_repeat15",
        "dataset_split": "seen",
    },
    "e4_b_keepk0_all6_repeat15": {
        "manifest_relpath": "e4_b_keepk0_all6_repeat15/round23_e4_b_keepk0_all6_repeat15_manifest.tsv",
        "log_stem": "round23_e4_b_keepk0_all6_repeat15",
        "dataset_split": "all6",
    },
    "e4_c_three_round_stress_smoke": {
        "manifest_relpath": "e4_c_three_round_stress_smoke/round23_e4_c_three_round_stress_smoke_manifest.tsv",
        "log_stem": "round23_e4_c_three_round_stress_smoke",
        "dataset_split": "seen",
    },
    "e4_c_three_round_stress_all6_smoke": {
        "manifest_relpath": "e4_c_three_round_stress_all6_smoke/round23_e4_c_three_round_stress_all6_smoke_manifest.tsv",
        "log_stem": "round23_e4_c_three_round_stress_all6_smoke",
        "dataset_split": "all6",
    },
    "e4_c_three_round_stress_pilot": {
        "manifest_relpath": "e4_c_three_round_stress_pilot/round23_e4_c_three_round_stress_pilot_manifest.tsv",
        "log_stem": "round23_e4_c_three_round_stress_pilot",
        "dataset_split": "seen",
    },
    "e4_c_three_round_stress_all6_repeat15": {
        "manifest_relpath": "e4_c_three_round_stress_all6_repeat15/round23_e4_c_three_round_stress_all6_repeat15_manifest.tsv",
        "log_stem": "round23_e4_c_three_round_stress_all6_repeat15",
        "dataset_split": "all6",
    },
    "e5_anchor_k19_keepk0_all6_smoke": {
        "manifest_relpath": "e5_anchor_k19_keepk0_all6_smoke/round23_e5_anchor_k19_keepk0_all6_smoke_manifest.tsv",
        "log_stem": "round23_e5_anchor_k19_keepk0_all6_smoke",
        "dataset_split": "all6",
    },
    "e5_anchor_k19_round23_all6_smoke": {
        "manifest_relpath": "e5_anchor_k19_round23_all6_smoke/round23_e5_anchor_k19_round23_all6_smoke_manifest.tsv",
        "log_stem": "round23_e5_anchor_k19_round23_all6_smoke",
        "dataset_split": "all6",
    },
    "e5_anchor_k21_keepk0_all6_smoke": {
        "manifest_relpath": "e5_anchor_k21_keepk0_all6_smoke/round23_e5_anchor_k21_keepk0_all6_smoke_manifest.tsv",
        "log_stem": "round23_e5_anchor_k21_keepk0_all6_smoke",
        "dataset_split": "all6",
    },
    "e5_anchor_k21_round23_all6_smoke": {
        "manifest_relpath": "e5_anchor_k21_round23_all6_smoke/round23_e5_anchor_k21_round23_all6_smoke_manifest.tsv",
        "log_stem": "round23_e5_anchor_k21_round23_all6_smoke",
        "dataset_split": "all6",
    },
    "e5_anchor_k19_keepk0_all6_repeat5": {
        "manifest_relpath": "e5_anchor_k19_keepk0_all6_repeat5/round23_e5_anchor_k19_keepk0_all6_repeat5_manifest.tsv",
        "log_stem": "round23_e5_anchor_k19_keepk0_all6_repeat5",
        "dataset_split": "all6",
    },
    "e5_anchor_k19_round23_all6_repeat5": {
        "manifest_relpath": "e5_anchor_k19_round23_all6_repeat5/round23_e5_anchor_k19_round23_all6_repeat5_manifest.tsv",
        "log_stem": "round23_e5_anchor_k19_round23_all6_repeat5",
        "dataset_split": "all6",
    },
    "e5_anchor_k21_keepk0_all6_repeat5": {
        "manifest_relpath": "e5_anchor_k21_keepk0_all6_repeat5/round23_e5_anchor_k21_keepk0_all6_repeat5_manifest.tsv",
        "log_stem": "round23_e5_anchor_k21_keepk0_all6_repeat5",
        "dataset_split": "all6",
    },
    "e5_anchor_k21_round23_all6_repeat5": {
        "manifest_relpath": "e5_anchor_k21_round23_all6_repeat5/round23_e5_anchor_k21_round23_all6_repeat5_manifest.tsv",
        "log_stem": "round23_e5_anchor_k21_round23_all6_repeat5",
        "dataset_split": "all6",
    },
    "e7_no_dataset_seen_repeat10": {
        "manifest_relpath": "e7_no_dataset_seen_repeat10/round23_e7_no_dataset_seen_repeat10_manifest.tsv",
        "log_stem": "round23_e7_no_dataset_seen_repeat10",
        "dataset_split": "seen",
    },
    "e7_no_dataset_unseen_repeat10": {
        "manifest_relpath": "e7_no_dataset_unseen_repeat10/round23_e7_no_dataset_unseen_repeat10_manifest.tsv",
        "log_stem": "round23_e7_no_dataset_unseen_repeat10",
        "dataset_split": "unseen",
    },
    "e7_no_coverage_seen_repeat5": {
        "manifest_relpath": "e7_no_coverage_seen_repeat5/round23_e7_no_coverage_seen_repeat5_manifest.tsv",
        "log_stem": "round23_e7_no_coverage_seen_repeat5",
        "dataset_split": "seen",
    },
    "e7_no_redundancy_seen_repeat5": {
        "manifest_relpath": "e7_no_redundancy_seen_repeat5/round23_e7_no_redundancy_seen_repeat5_manifest.tsv",
        "log_stem": "round23_e7_no_redundancy_seen_repeat5",
        "dataset_split": "seen",
    },
    "e1_c4only_seen_smoke": {
        "manifest_relpath": "e1_c4only_seen_smoke/round23_e1_c4only_seen_smoke_manifest.tsv",
        "log_stem": "round23_e1_c4only_seen_smoke",
        "dataset_split": "seen",
    },
    "e1_c4only_seen_repeat30": {
        "manifest_relpath": "e1_c4only_seen_repeat30/round23_e1_c4only_seen_repeat30_manifest.tsv",
        "log_stem": "round23_e1_c4only_seen_repeat30",
        "dataset_split": "seen",
    },
    "e1_augpe_seen_smoke": {
        "manifest_relpath": "e1_augpe_seen_smoke/round23_e1_augpe_seen_smoke_manifest.tsv",
        "log_stem": "round23_e1_augpe_seen_smoke",
        "dataset_split": "seen",
    },
    "e1_augpe_seen_repeat30": {
        "manifest_relpath": "e1_augpe_seen_repeat30/round23_e1_augpe_seen_repeat30_manifest.tsv",
        "log_stem": "round23_e1_augpe_seen_repeat30",
        "dataset_split": "seen",
    },
    "e1_dpprompt_seen_smoke": {
        "manifest_relpath": "e1_dpprompt_seen_smoke/round23_e1_dpprompt_seen_smoke_manifest.tsv",
        "log_stem": "round23_e1_dpprompt_seen_smoke",
        "dataset_split": "seen",
    },
    "e1_dpprompt_seen_repeat30": {
        "manifest_relpath": "e1_dpprompt_seen_repeat30/round23_e1_dpprompt_seen_repeat30_manifest.tsv",
        "log_stem": "round23_e1_dpprompt_seen_repeat30",
        "dataset_split": "seen",
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_name: str
    meta_seed: int
    config_path: Path
    output_root: str
    method: str = "round23"
    controller_scope: str = DEFAULT_ROUND23_CONTROLLER_SCOPE
    controller_bundle: str = ""
    reference_budget: int = 20


def normalize_output_root(raw_output_root: str) -> Path:
    candidate = Path(str(raw_output_root).replace("\\", "/"))
    parts = list(candidate.parts)
    if parts and parts[0] == ROUND23_ROOT.name:
        candidate = Path(*parts[1:])
    return candidate


def resolve_mode_paths(mode: str) -> dict[str, str]:
    if mode not in MODE_PATHS:
        raise ValueError(f"Unsupported mode: {mode}")
    return dict(MODE_PATHS[mode])


def resolve_model_dir(model_dir: str | Path | None) -> Path:
    resolved = Path(model_dir).resolve() if model_dir else DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            "Round23 controller bundle not found. Provide --model-dir or create the default all6 bundle at: "
            f"{resolved}"
        )
    return resolved


def resolve_config_path(raw_config_path: str | Path) -> Path:
    path = Path(str(raw_config_path))
    if path.is_absolute():
        return path.resolve()
    return (ROUND23_ROOT / path).resolve()


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
                    config_path=resolve_config_path(str(row["config_path"])),
                    output_root=str(row["output_root"]),
                    method=str(row.get("method", "round23")),
                    controller_scope=str(row.get("controller_scope", DEFAULT_ROUND23_CONTROLLER_SCOPE)),
                    controller_bundle=str(row.get("controller_bundle", "")),
                    reference_budget=int(row.get("reference_budget", 20) or 20),
                )
            )
    return specs


def resolve_run_script(method: str) -> Path:
    if method not in RUN_SCRIPTS:
        raise ValueError(f"Unsupported method for runner: {method}")
    return RUN_SCRIPTS[method]


def sidecar_suffix_for_method(method: str) -> str:
    if method == "round23":
        return "_dynamic_controller_runtime.json"
    if method == "round23_absk_oneshot":
        return "_absolute_k_controller_runtime.json"
    if method == "round23_keepk0":
        return "_keep_k0_runtime.json"
    if method == "round23_3round_stress":
        return "_three_round_stress_runtime.json"
    if method == "e1_c4only":
        return "_c4only_runtime.json"
    if method == "e1_augpe":
        return "_augpe_runtime.json"
    if method == "e1_dpprompt":
        return "_dpprompt_runtime.json"
    return "_dynamic_controller_runtime.json"


def resolve_model_dir_for_spec(cli_model_dir: str | Path | None, spec: ExperimentSpec) -> Path | None:
    if spec.method in ("round23_keepk0", "e1_c4only", "e1_augpe", "e1_dpprompt"):
        return None
    if cli_model_dir:
        return Path(cli_model_dir).resolve()
    if spec.controller_bundle:
        resolved = (ROUND23_ROOT / "artifacts" / "controller_bundle" / spec.controller_bundle).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Controller bundle not found for {spec.experiment_id}: {resolved}")
        return resolved
    return DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE.resolve()


def parse_nvidia_smi_memory_report(
    report_text: str,
    *,
    target_name_token: str,
    preferred_index: str | None = None,
) -> tuple[str, float]:
    parsed_rows: list[tuple[str, str, float]] = []
    for raw_line in report_text.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        index, name, free_mib = parts
        if target_name_token in name:
            parsed_rows.append((index, name, float(free_mib) / 1024.0))
    if preferred_index is not None:
        for index, _name, free_gb in parsed_rows:
            if index == str(preferred_index):
                return index, free_gb
    if parsed_rows:
        index, _name, free_gb = parsed_rows[0]
        return index, free_gb
    raise RuntimeError(f"Could not find GPU with token {target_name_token!r} in nvidia-smi output.")


def query_target_gpu_free_gb(*, target_name_token: str, preferred_index: str | None = None) -> tuple[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_nvidia_smi_memory_report(
        completed.stdout,
        target_name_token=target_name_token,
        preferred_index=preferred_index,
    )


def wait_for_vllm_capacity(
    log_handle,
    *,
    target_name_token: str,
    preferred_index: str | None,
    minimum_free_gb: float,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        gpu_index, free_gb = query_target_gpu_free_gb(
            target_name_token=target_name_token,
            preferred_index=preferred_index,
        )
        if free_gb >= minimum_free_gb:
            log_handle.write(
                f"{datetime.now().isoformat()} GPU_READY target={target_name_token} "
                f"index={gpu_index} free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}\n"
            )
            log_handle.flush()
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"A6000 free memory stayed below {minimum_free_gb:.2f} GiB for {timeout_seconds:.0f} seconds."
            )
        log_handle.write(
            f"{datetime.now().isoformat()} GPU_WAIT target={target_name_token} "
            f"index={gpu_index} free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}\n"
        )
        log_handle.flush()
        time.sleep(poll_seconds)


def is_retryable_resource_failure(text: str) -> bool:
    return any(marker in text for marker in RETRYABLE_RESOURCE_FAILURE_MARKERS)


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
    model_dir: Path | None,
    timeout_seconds: int,
    log_dir: Path,
) -> tuple[int, str, str, float]:
    started = time.time()
    log_path = log_dir / f"{spec.experiment_id}.log"
    run_script = resolve_run_script(spec.method)
    command = [
        sys.executable,
        str(run_script),
        "--config",
        str(spec.config_path),
        "--output-root",
        str((ROUND23_ROOT / normalize_output_root(spec.output_root)).resolve()),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if model_dir and spec.method not in ("round23_keepk0", "e1_c4only", "e1_augpe", "e1_dpprompt"):
        command.extend(["--model-dir", str(model_dir)])
    command.extend(["--reference-budget", str(spec.reference_budget)])
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
    mode: str,
    dataset_split: str,
    status: str,
    attempt: int,
    duration_seconds: float,
    error_excerpt: str = "",
) -> dict[str, Any]:
    normalized_output_root = normalize_output_root(spec.output_root)
    sidecar_path = (
        ROUND23_ROOT / normalized_output_root / f"{spec.experiment_id}{sidecar_suffix_for_method(spec.method)}"
    ).resolve()
    row: dict[str, Any] = {
        "mode": mode,
        "dataset_split": dataset_split,
        "experiment_id": spec.experiment_id,
        "method": spec.method,
        "method_display_name": spec.method,
        "dataset_name": spec.dataset_name,
        "meta_seed": spec.meta_seed,
        "status": status,
        "attempt": attempt,
        "duration_seconds": round(duration_seconds, 3),
        "predicted_delta_k": "",
        "predicted_target_budget": "",
        "controller_scope": "",
        "bundle_version": "",
        "learner_family": "",
        "model_family": "",
        "feature_version": "",
        "target_mode": "",
        "target_field": "",
        "reference_budget": "",
        "best_top1": "",
        "best_top3": "",
        "best_top5": "",
        "best_top10": "",
        "config_path": str(spec.config_path),
        "output_root": str(normalized_output_root),
        "sidecar_path": str(sidecar_path),
        "error_excerpt": error_excerpt,
    }
    if sidecar_path.exists():
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        row["predicted_delta_k"] = payload.get("predicted_delta_k", "")
        row["predicted_target_budget"] = payload.get("predicted_target_budget", "")
        for key in (
            "controller_scope",
            "bundle_version",
            "learner_family",
            "model_family",
            "feature_version",
            "target_mode",
            "target_field",
            "reference_budget",
        ):
            row[key] = payload.get(key, "")
        eval_summary = payload.get("runtime_artifacts", {}).get("eval_summary") or {}
        for metric in ("best_top1", "best_top3", "best_top5", "best_top10"):
            row[metric] = extract_eval_metric(eval_summary, metric)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round23 dynamic-controller experiments sequentially")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PATHS.keys()),
        required=True,
    )
    parser.add_argument("--model-dir", default=None, help="Path to round23 controller bundle")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument(
        "--retry-all-failures",
        action="store_true",
        help="Retry every failed experiment until --max-attempts, not only retryable resource failures.",
    )
    parser.add_argument("--target-gpu-name-token", default=DEFAULT_TARGET_GPU_NAME_TOKEN)
    parser.add_argument(
        "--target-gpu-index",
        default=None,
        help="Optional physical GPU index to match in nvidia-smi when waiting for vLLM capacity.",
    )
    parser.add_argument("--min-free-gb-for-vllm", type=float, default=DEFAULT_MIN_FREE_GB_FOR_VLLM)
    parser.add_argument("--gpu-wait-poll-seconds", type=float, default=DEFAULT_GPU_WAIT_POLL_SECONDS)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=DEFAULT_GPU_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--reset-summary", action="store_true", help="Remove existing mode summary/log rows before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments without executing them")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on pending experiments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode_paths = resolve_mode_paths(args.mode)
    manifest = (
        ROUND23_ROOT
        / "configs"
        / "experiments"
        / "single_node_tuning_round23_dynamic"
        / mode_paths["manifest_relpath"]
    )
    specs = load_manifest(manifest)
    logs_root = ROUND23_ROOT / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    per_exp_log_dir = logs_root / f"{mode_paths['log_stem']}_logs"
    per_exp_log_dir.mkdir(parents=True, exist_ok=True)
    master_log = logs_root / f"{mode_paths['log_stem']}_master.log"
    summary_tsv = logs_root / f"{mode_paths['log_stem']}_summary.tsv"
    summary_jsonl = logs_root / f"{mode_paths['log_stem']}_summary.jsonl"
    if args.reset_summary and not args.dry_run:
        summary_tsv.unlink(missing_ok=True)
        summary_jsonl.unlink(missing_ok=True)
        master_log.unlink(missing_ok=True)
    fields = [
        "mode",
        "dataset_split",
        "experiment_id",
        "method",
        "method_display_name",
        "dataset_name",
        "meta_seed",
        "status",
        "attempt",
        "duration_seconds",
        "predicted_delta_k",
        "predicted_target_budget",
        "controller_scope",
        "bundle_version",
        "learner_family",
        "model_family",
        "feature_version",
        "target_mode",
        "target_field",
        "reference_budget",
        "best_top1",
        "best_top3",
        "best_top5",
        "best_top10",
        "config_path",
        "output_root",
        "sidecar_path",
        "error_excerpt",
    ]
    initialize_tsv(summary_tsv, fields)
    done = completed_ids(summary_tsv, summary_jsonl)
    pending = [spec for spec in specs if spec.experiment_id not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "dataset_split": mode_paths["dataset_split"],
                    "pending_count": len(pending),
                    "manifest": str(manifest.resolve()),
                    "summary_tsv": str(summary_tsv.resolve()),
                    "first_experiments": [spec.experiment_id for spec in pending[:10]],
                    "model_dir": str(resolve_model_dir_for_spec(args.model_dir, pending[0])) if pending else "",
                    "controller_scope": DEFAULT_ROUND23_CONTROLLER_SCOPE,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    failed_experiment_ids: list[str] = []
    with master_log.open("a", encoding="utf-8") as master:
        for spec in pending:
            success = False
            for attempt in range(1, args.max_attempts + 1):
                master.write(f"{datetime.now().isoformat()} START {spec.experiment_id} attempt={attempt}\n")
                master.flush()
                wait_for_vllm_capacity(
                    master,
                    target_name_token=args.target_gpu_name_token,
                    preferred_index=getattr(args, "target_gpu_index", None),
                    minimum_free_gb=args.min_free_gb_for_vllm,
                    poll_seconds=args.gpu_wait_poll_seconds,
                    timeout_seconds=args.gpu_wait_timeout_seconds,
                )
                model_dir = resolve_model_dir_for_spec(args.model_dir, spec)
                code, _, stderr, duration = run_single_experiment(
                    spec,
                    model_dir=model_dir,
                    timeout_seconds=args.timeout_seconds,
                    log_dir=per_exp_log_dir,
                )
                status = "success" if code == 0 else "failed"
                row = build_summary_row(
                    spec,
                    mode=args.mode,
                    dataset_split=mode_paths["dataset_split"],
                    status=status,
                    attempt=attempt,
                    duration_seconds=duration,
                    error_excerpt=stderr[:400].replace("\n", "\\n"),
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
                    log_text = ""
                    log_path = per_exp_log_dir / f"{spec.experiment_id}.log"
                    if log_path.exists():
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                    if (
                        not args.retry_all_failures
                        and not is_retryable_resource_failure(stderr + "\n" + log_text)
                    ):
                        break
                    master.write(
                        f"{datetime.now().isoformat()} RETRY {spec.experiment_id} stderr={stderr[:400]}\n"
                    )
                    master.flush()
                    if args.retry_delay_seconds > 0:
                        time.sleep(args.retry_delay_seconds)
            if not success:
                failed_experiment_ids.append(spec.experiment_id)
                master.write(
                    f"{datetime.now().isoformat()} FINAL_FAILURE {spec.experiment_id} "
                    f"attempts={args.max_attempts}\n"
                )
                master.flush()
        if failed_experiment_ids:
            master.write(
                f"{datetime.now().isoformat()} FAILED_EXPERIMENTS "
                f"count={len(failed_experiment_ids)} ids={','.join(failed_experiment_ids)}\n"
            )
            master.flush()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
