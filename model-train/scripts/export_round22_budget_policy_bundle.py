#!/usr/bin/env python3
"""Export a round22 learned-budget-policy bundle for runtime use."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PAPER_NEW_ROUND22 = Path(__file__).resolve().parents[2] / "paper-new-round22"
BUNDLE_DIR_DEFAULT = PAPER_NEW_ROUND22 / "artifacts/learned_budget_policy/round22_lgbm_v1"

DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
BUDGETS = [18, 19, 20, 21, 22]

FEATURE_NAMES = [
    "shape_score",
    "private_mean_length",
    "private_p75_length",
    "private_length_iqr",
    "support_mean_at_k20",
    "coverage_mean_at_k20",
    "coverage_p25_at_k20",
    "genericity_mean_at_k20",
]


def get_git_info() -> tuple[str, str]:
    """Return (git_commit, git_branch) for the model-train repo."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
        ).strip()
    except Exception:
        git_commit = ""
        git_branch = ""
    return git_commit, git_branch


def load_train_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping at {path}")
    return payload


def export_feature_schema(include_dataset_onehot: bool, output_path: Path) -> None:
    """Write feature_schema.json to the bundle directory."""
    schema = {
        "version": "1.0",
        "feature_names": FEATURE_NAMES,
        "include_dataset_onehot": include_dataset_onehot,
        "onehot_order": DATASET_ORDER if include_dataset_onehot else [],
        "total_features": 12 if include_dataset_onehot else 8,
    }
    output_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    print(f"[export] feature_schema.json -> {output_path}")


def _load_training_seeds(training_seeds_json: Path | None) -> list[int]:
    if training_seeds_json is None:
        return []
    payload = json.loads(training_seeds_json.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(value) for value in payload]
    if isinstance(payload, dict) and "training_seeds" in payload:
        return [int(value) for value in payload["training_seeds"]]
    raise TypeError("training seeds JSON must be a list or {'training_seeds': [...]} mapping")


def export_metadata(
    *,
    bundle_dir: Path,
    train_config_path: Path,
    model_dir: Path,
    training_data_version: str,
    training_seeds: list[int],
    lightgbm_params_override: dict[str, Any] | None,
) -> None:
    """Write metadata.json to the bundle directory."""
    train_cfg = load_train_config(train_config_path)
    lgbm_params = (
        dict(lightgbm_params_override)
        if lightgbm_params_override is not None
        else dict(train_cfg.get("model", {}).get("lightgbm_params", {}))
    )
    git_commit, git_branch = get_git_info()
    metadata = {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "training_data_version": str(training_data_version),
        "reward_lambda": 0.002,
        "lightgbm_params": lgbm_params,
        "training_seeds": list(training_seeds),
        "model_train_git_commit": git_commit,
        "model_train_git_branch": git_branch,
        "source_model_dir": str(model_dir.resolve()),
        "bundle_dir": str(bundle_dir.resolve()),
    }
    output_path = bundle_dir / "metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[export] metadata.json -> {output_path}")


def export_model_files(model_dir: Path, bundle_dir: Path) -> None:
    """Copy model_k*.txt files to the bundle directory."""
    for k in BUDGETS:
        src = model_dir / f"model_k{k}.txt"
        dst = bundle_dir / f"model_k{k}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Model not found: {src}")
        shutil.copy2(src, dst)
        print(f"[export] model_k{k}.txt copied")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export round22 budget policy bundle")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained model directory (model-train/artifacts/models/...)",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to train_round22_bandit.yaml",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR_DEFAULT,
        help="Output bundle directory",
    )
    parser.add_argument(
        "--training-data-version",
        type=str,
        default="round22_bandit_full500_v2",
        help="Free-form training data version string recorded into metadata.json",
    )
    parser.add_argument(
        "--training-seeds-json",
        type=Path,
        default=None,
        help="Optional JSON file containing training seeds or {'training_seeds': [...]}",
    )
    parser.add_argument(
        "--lightgbm-params-json",
        type=Path,
        default=None,
        help="Optional JSON file whose object overrides metadata.lightgbm_params",
    )
    args = parser.parse_args()

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = load_train_config(args.train_config)
    include_dataset_onehot = bool(train_cfg.get("model", {}).get("include_dataset_one_hot", True))
    training_seeds = _load_training_seeds(args.training_seeds_json)
    lightgbm_params_override = None
    if args.lightgbm_params_json is not None:
        payload = json.loads(args.lightgbm_params_json.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise TypeError("--lightgbm-params-json must contain a JSON object")
        lightgbm_params_override = payload

    export_feature_schema(
        include_dataset_onehot=include_dataset_onehot,
        output_path=args.bundle_dir / "feature_schema.json",
    )
    export_metadata(
        bundle_dir=args.bundle_dir,
        train_config_path=args.train_config,
        model_dir=args.model_dir,
        training_data_version=args.training_data_version,
        training_seeds=training_seeds,
        lightgbm_params_override=lightgbm_params_override,
    )
    export_model_files(args.model_dir, args.bundle_dir)
    print("[done] Bundle exported successfully")


if __name__ == "__main__":
    main()
