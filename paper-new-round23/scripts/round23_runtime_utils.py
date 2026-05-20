#!/usr/bin/env python3
"""Shared runtime helpers for round23 dynamic budget controller."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PAPER_NEW_ROUND23_ROOT = Path(__file__).resolve().parents[1]
PAPER_NEW_ROUND19_ROOT = (PAPER_NEW_ROUND23_ROOT / "../paper-new-round19").resolve()
DEFAULT_ROUND23_CONTROLLER_SCOPE = "all6"
DEFAULT_ROUND23_ALL6_CONTROLLER_BUNDLE = (
    PAPER_NEW_ROUND23_ROOT
    / "artifacts"
    / "controller_bundle"
    / "round23_controller_1200_all6_top1_delta_m0005_extratrees_broad_no_dataset"
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_yaml_with_inherits(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    inherits = data.pop("inherits", []) or []
    merged: dict[str, Any] = {}
    for parent in inherits:
        parent_path = (Path(path).parent / str(parent)).resolve()
        merged = deep_merge(merged, load_yaml_with_inherits(parent_path))
    return deep_merge(merged, data)


def build_round19_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(PAPER_NEW_ROUND19_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if env.get("CUDA_VISIBLE_DEVICES") and not env.get("CUDA_DEVICE_ORDER"):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def run_round19_selector_subprocess(
    *,
    config_path: str | Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "paper_new_selector.run_selector_single_node", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(PAPER_NEW_ROUND19_ROOT),
        env=build_round19_subprocess_env(),
        timeout=timeout_seconds,
    )


def collect_runtime_artifacts(output_root: str | Path) -> dict[str, Any]:
    runtime_root = Path(output_root).resolve()
    stage1_summary = runtime_root / "stage1_summary.json"
    budget_calibration = runtime_root / "stage1_budget_calibration.json"
    eval_dir = runtime_root / "eval"
    eval_summary_candidates = [
        eval_dir / "downstream_eval_summary.json",
        eval_dir / "summary.json",
    ]
    eval_summary = next((path for path in eval_summary_candidates if path.exists()), None)

    if not stage1_summary.exists():
        raise FileNotFoundError(f"Expected runtime stage1 summary at {stage1_summary}")
    if eval_summary is None:
        raise FileNotFoundError(
            "Expected runtime eval summary under "
            f"{eval_dir}; checked: {', '.join(str(path) for path in eval_summary_candidates)}"
        )

    artifacts: dict[str, Any] = {
        "runtime_output_root": str(runtime_root),
        "stage1_summary_path": str(stage1_summary),
        "eval_summary_path": str(eval_summary),
    }
    if budget_calibration.exists():
        artifacts["budget_calibration_path"] = str(budget_calibration)

    try:
        import json

        eval_payload = json.loads(eval_summary.read_text(encoding="utf-8"))
    except Exception:
        eval_payload = None
    if isinstance(eval_payload, dict):
        artifacts["eval_summary"] = eval_payload
    return artifacts


def extract_eval_metric(eval_summary: dict[str, Any], metric_name: str) -> Any:
    if metric_name in eval_summary:
        return eval_summary.get(metric_name)
    metrics = eval_summary.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(metric_name, "")
    return ""


def extract_controller_metadata(
    model_metadata: dict[str, Any] | None,
    *,
    controller_scope: str = DEFAULT_ROUND23_CONTROLLER_SCOPE,
) -> dict[str, Any]:
    metadata = dict(model_metadata or {})
    return {
        "controller_scope": controller_scope,
        "bundle_version": metadata.get(
            "bundle_version",
            metadata.get("controller_version", ""),
        ),
        "learner_family": metadata.get(
            "learner_family",
            metadata.get("model_family", ""),
        ),
        "model_family": metadata.get(
            "model_family",
            metadata.get("learner_family", ""),
        ),
        "feature_version": metadata.get("feature_version", ""),
        "target_mode": metadata.get("target_mode", ""),
        "target_field": metadata.get("target_field", ""),
        "reference_budget": metadata.get("reference_budget", ""),
    }
