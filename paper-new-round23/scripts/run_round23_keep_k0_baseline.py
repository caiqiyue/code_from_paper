#!/usr/bin/env python3
"""Run a fixed keep-k0=20 round23 baseline without adaptive controller."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from round23_runtime_utils import collect_runtime_artifacts, deep_merge, load_yaml_with_inherits, run_round19_selector_subprocess


def generate_override_config(
    original_config_path: str | Path,
    output_root: str | Path,
    reference_budget: int = 20,
) -> tuple[Path, str]:
    original_cfg = load_yaml_with_inherits(original_config_path)
    experiment_id = str(original_cfg.get("meta", {}).get("experiment_id", Path(original_config_path).stem))
    override = {
        "meta": {
            "keep_k0_runtime": {
                "enabled": True,
                "reference_budget": int(reference_budget),
                "predicted_delta_k": 0,
                "predicted_target_budget": int(reference_budget),
            }
        },
        "paths": {
            "output_root": str(Path(output_root).resolve()),
        },
        "selector": {
            "seed_top_k": int(reference_budget),
            "seed_budget_rule": {
                "enabled": False,
                "mode": "hierarchical_shape_routing",
            },
        },
        "round23_controller": {
            "enabled": False,
        },
    }
    merged = deep_merge(original_cfg, override)
    override_path = Path(output_root) / f"{experiment_id}_keep_k0_override.yaml"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.dump(merged, handle)
    return override_path, experiment_id


def write_runtime_sidecar(
    output_root: str | Path,
    experiment_id: str,
    override_config_path: Path,
    runtime_artifacts: dict[str, Any],
    reference_budget: int,
) -> Path:
    sidecar = {
        "budget_policy_type": "keep_k0",
        "reference_budget": int(reference_budget),
        "predicted_delta_k": 0,
        "predicted_target_budget": int(reference_budget),
        "override_config_path": str(override_config_path),
        "runtime_artifacts": runtime_artifacts,
    }
    sidecar_path = Path(output_root) / f"{experiment_id}_keep_k0_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run keep-k0=20 baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-budget", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    override_path, experiment_id = generate_override_config(
        original_config_path=args.config,
        output_root=args.output_root,
        reference_budget=args.reference_budget,
    )
    result = run_round19_selector_subprocess(config_path=override_path, timeout_seconds=args.timeout_seconds)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(f"[ERROR] round19 runtime failed:\n{result.stderr}", file=sys.stderr)
        return int(result.returncode)
    print(result.stdout, end="")
    runtime_artifacts = collect_runtime_artifacts(args.output_root)
    sidecar_path = write_runtime_sidecar(
        args.output_root,
        experiment_id,
        override_path,
        runtime_artifacts,
        args.reference_budget,
    )
    print(f"[round23-keepk0] Done. Sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
