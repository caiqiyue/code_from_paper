#!/usr/bin/env python3
"""Run a single E1 C4-only baseline experiment.

C4-only uses public C4 text as the synthetic corpus with no DP noise (eps=inf).
We reuse round19's existing ``stage1_mode: c4_only`` path which already implements
"random sample from C4 initialization pool, skip bootstrap, run downstream eval".
This script generates an override YAML, invokes the round19 selector subprocess,
then writes a runner-compatible sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from round23_runtime_utils import (
    collect_runtime_artifacts,
    deep_merge,
    load_yaml_with_inherits,
    run_round19_selector_subprocess,
)


ROUND23_ROOT = Path(__file__).resolve().parents[1]


def generate_override_config(
    original_config_path: str | Path,
    output_root: str | Path,
) -> tuple[Path, str, str]:
    original_cfg = load_yaml_with_inherits(original_config_path)
    experiment_id = str(original_cfg.get("meta", {}).get("experiment_id", Path(original_config_path).stem))
    dataset_name = str(original_cfg.get("data", {}).get("dataset_name", "unknown"))
    override = {
        "meta": {
            "c4only_baseline_runtime": {
                "enabled": True,
                "epsilon": "inf",
                "privacy_note": "no DP; random sample from public C4 pool",
            }
        },
        "paths": {
            "output_root": str(Path(output_root).resolve()),
        },
        "pipeline": {
            "stage1_mode": "c4_only",
            "stage2_mode": "pretext_bootstrap",
            "run_eval": True,
        },
        "selector": {
            "seed_budget_rule": {
                "enabled": False,
                "mode": "hierarchical_shape_routing",
            },
        },
        "round23_controller": {
            "enabled": False,
        },
        "eval": {
            "enabled": True,
            "mode": "pretext_small",
        },
    }
    merged = deep_merge(original_cfg, override)
    override_path = Path(output_root) / f"{experiment_id}_c4only_override.yaml"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.dump(merged, handle)
    return override_path, experiment_id, dataset_name


def write_runtime_sidecar(
    output_root: str | Path,
    experiment_id: str,
    dataset_name: str,
    override_config_path: Path,
    runtime_artifacts: dict[str, Any],
) -> Path:
    sidecar = {
        "budget_policy_type": "c4only",
        "dataset_name": dataset_name,
        "experiment_id": experiment_id,
        "epsilon": "inf",
        "privacy_note": "no DP; random sample from public C4 pool, then LLaMA few-shot expansion via round19 bootstrap",
        "override_config_path": str(override_config_path),
        "runtime_artifacts": runtime_artifacts,
    }
    sidecar_path = Path(output_root) / f"{experiment_id}_c4only_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E1 C4-only baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reference-budget",
        type=int,
        default=20,
        help="Accepted for runner compatibility; not used by the C4-only baseline.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    override_path, experiment_id, dataset_name = generate_override_config(
        original_config_path=args.config,
        output_root=output_root,
    )
    print(
        f"[e1_c4only] experiment_id={experiment_id} dataset={dataset_name} "
        f"output_root={output_root}"
    )
    print(f"[e1_c4only] Invoking round19 selector subprocess with {override_path}")
    result = run_round19_selector_subprocess(
        config_path=override_path,
        timeout_seconds=args.timeout_seconds,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(f"[ERROR] round19 runtime failed:\n{result.stderr}", file=sys.stderr)
        return int(result.returncode)
    print(result.stdout, end="")
    runtime_artifacts = collect_runtime_artifacts(args.output_root)
    sidecar_path = write_runtime_sidecar(
        output_root=output_root,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        override_config_path=override_path,
        runtime_artifacts=runtime_artifacts,
    )
    print(f"[e1_c4only] Done. Sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
