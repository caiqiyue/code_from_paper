from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent

if str(MODEL_TRAIN_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MODEL_TRAIN_ROOT))

from common import BUDGETS, CANONICAL_ROUND23_ROOT, REFERENCE_BUDGET, dump_json
from round23_feature_sets import get_feature_spec


def _load_train_metrics(trained_model_dir: str | Path) -> dict[str, Any]:
    metrics_path = Path(trained_model_dir) / "train_metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def export_round23_absolute_k_bundle(
    *,
    trained_model_dir: str | Path,
    output_dir: str | Path,
    bundle_version: str,
    model_family: str,
    training_data_version: str,
    feature_version: str | None = None,
    feature_names: list[str] | None = None,
    include_dataset_onehot: bool | None = None,
    onehot_order: list[str] | None = None,
    target_field: str | None = None,
    target_mode: str | None = None,
    tie_margin: float | None = None,
    controller_scope: str | None = None,
) -> dict[str, Any]:
    model_root = Path(trained_model_dir)
    bundle_root = Path(output_dir)
    bundle_root.mkdir(parents=True, exist_ok=True)
    train_metrics = _load_train_metrics(model_root)
    if feature_version is None:
        feature_version = train_metrics.get("feature_version")
    if feature_names is None:
        feature_names = train_metrics.get("feature_fields")
    if include_dataset_onehot is None:
        include_dataset_onehot = train_metrics.get("include_dataset_one_hot")
    if onehot_order is None:
        onehot_order = train_metrics.get("onehot_order")
    if target_field is None:
        target_field = train_metrics.get("target_field")
    if target_mode is None:
        target_mode = train_metrics.get("target_mode")
    if tie_margin is None:
        tie_margin = train_metrics.get("tie_margin")

    if feature_version and (feature_names is None or include_dataset_onehot is None):
        feature_spec = get_feature_spec(feature_version)
        feature_names = list(feature_names or feature_spec.feature_fields)
        include_dataset_onehot = feature_spec.include_dataset_onehot if include_dataset_onehot is None else bool(include_dataset_onehot)
        onehot_order = list(onehot_order or feature_spec.onehot_order)
    if feature_names is None:
        raise ValueError("feature_names must be provided directly or via train_metrics.json")
    if include_dataset_onehot is None:
        raise ValueError("include_dataset_onehot must be provided directly or via train_metrics.json")
    onehot_order = list(onehot_order or [])

    exported_models: dict[str, str] = {}
    for budget in BUDGETS:
        matches = list(model_root.glob(f"model_k{budget}.*"))
        if not matches:
            raise FileNotFoundError(f"Missing model file for budget={budget} under {model_root}")
        source = matches[0]
        target = bundle_root / source.name
        shutil.copy2(source, target)
        exported_models[str(budget)] = str(target)

    feature_schema = {
        "version": "round23_absolute_k_feature_schema_v1",
        "feature_version": feature_version or "custom",
        "feature_names": list(feature_names),
        "state_features": list(feature_names),
        "include_dataset_onehot": bool(include_dataset_onehot),
        "onehot_order": list(onehot_order if include_dataset_onehot else []),
        "total_features": len(feature_names) + (len(onehot_order) if include_dataset_onehot else 0),
        "reference_budget": REFERENCE_BUDGET,
        "budget_space": BUDGETS,
        "policy_parameterization": "absolute_k",
    }
    metadata = {
        "controller_version": bundle_version,
        "bundle_version": bundle_version,
        "learner_family": str(model_family),
        "model_family": str(model_family),
        "training_data_version": str(training_data_version),
        "reference_budget": REFERENCE_BUDGET,
        "budget_space": BUDGETS,
        "policy_parameterization": "absolute_k",
        "feature_version": feature_schema["feature_version"],
        "feature_names": list(feature_names),
        "expected_feature_order": list(feature_names),
        "include_dataset_onehot": bool(include_dataset_onehot),
        "onehot_order": feature_schema["onehot_order"],
        "total_features": feature_schema["total_features"],
        "target_field": target_field or "reward_round23_controller",
        "target_mode": target_mode or "",
        "tie_margin": tie_margin,
        "controller_scope": controller_scope or str(train_metrics.get("controller_scope", "")),
    }
    dump_json(bundle_root / "feature_schema.json", feature_schema)
    dump_json(bundle_root / "metadata.json", metadata)
    report = {
        "bundle_version": bundle_version,
        "bundle_root": str(bundle_root),
        "model_family": model_family,
        "training_data_version": training_data_version,
        "policy_parameterization": "absolute_k",
        "exported_models": exported_models,
    }
    dump_json(bundle_root / "bundle_export_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export round23 absolute-k controller bundle.")
    parser.add_argument("--trained-model-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(CANONICAL_ROUND23_ROOT / "artifacts" / "controller_bundle" / "round23_absk_v1"),
    )
    parser.add_argument("--bundle-version", default="round23_absk_v1")
    parser.add_argument("--model-family", default="lightgbm")
    parser.add_argument("--training-data-version", default="round19_round23_collection_repeat40")
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--target-field", default=None)
    parser.add_argument("--target-mode", default=None)
    parser.add_argument("--tie-margin", type=float, default=None)
    parser.add_argument("--controller-scope", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export_round23_absolute_k_bundle(
        trained_model_dir=args.trained_model_dir,
        output_dir=args.output_dir,
        bundle_version=args.bundle_version,
        model_family=args.model_family,
        training_data_version=args.training_data_version,
        feature_version=args.feature_version,
        target_field=args.target_field,
        target_mode=args.target_mode,
        tie_margin=args.tie_margin,
        controller_scope=args.controller_scope,
    )
    print(f"EXPORTED bundle={report['bundle_version']} output_dir={report['bundle_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
