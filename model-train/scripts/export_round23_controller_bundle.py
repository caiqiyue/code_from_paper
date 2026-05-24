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

from common import CANONICAL_ROUND23_ROOT, DELTA_ACTIONS, REFERENCE_BUDGET, dump_json
from round23_feature_sets import get_feature_spec


def _load_train_metrics(trained_model_dir: str | Path) -> dict[str, Any]:
    metrics_path = Path(trained_model_dir) / "train_metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def export_round23_controller_bundle(
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
    reward_formula: str | None = None,
    target_field: str | None = None,
    target_mode: str | None = None,
    tie_margin: float | None = None,
    label_audit_report: str | None = None,
    controller_scope: str | None = None,
    included_datasets: list[str] | None = None,
    excluded_datasets: list[str] | None = None,
    record_count: int | None = None,
    training_data_hash: str | None = None,
    selection_protocol: str | None = None,
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
    for delta_k in DELTA_ACTIONS:
        action_name = f"neg{abs(delta_k)}" if delta_k < 0 else ("0" if delta_k == 0 else f"pos{delta_k}")
        matches = list(model_root.glob(f"model_dk_{action_name}.*"))
        if not matches:
            raise FileNotFoundError(f"Missing model file for delta_k={delta_k} under {model_root}")
        source = matches[0]
        target = bundle_root / source.name
        shutil.copy2(source, target)
        exported_models[str(delta_k)] = str(target)

    feature_schema = {
        "version": "round23_controller_feature_schema_v1",
        "feature_version": feature_version or "custom",
        "feature_names": list(feature_names),
        "state_features": list(feature_names),
        "include_dataset_onehot": bool(include_dataset_onehot),
        "onehot_order": list(onehot_order if include_dataset_onehot else []),
        "total_features": len(feature_names) + (len(onehot_order) if include_dataset_onehot else 0),
        "reference_budget": REFERENCE_BUDGET,
        "delta_actions": DELTA_ACTIONS,
    }
    metadata = {
        "controller_version": bundle_version,
        "bundle_version": bundle_version,
        "learner_family": str(model_family),
        "model_family": str(model_family),
        "training_data_version": str(training_data_version),
        "reference_budget": REFERENCE_BUDGET,
        "action_space": DELTA_ACTIONS,
        "delta_actions": DELTA_ACTIONS,
        "feature_version": feature_schema["feature_version"],
        "feature_names": list(feature_names),
        "expected_feature_order": list(feature_names),
        "include_dataset_onehot": bool(include_dataset_onehot),
        "onehot_order": feature_schema["onehot_order"],
        "total_features": feature_schema["total_features"],
        "reward_formula": reward_formula
        or "1.0*(best_top1_k1-best_top1_k0)+0.25*(coverage_p25_k1-coverage_p25_k0)-0.20*max(0,support_mean_k0-support_mean_k1)-0.02*abs(delta_k)",
        "target_field": target_field or "reward_round23_controller",
        "target_mode": target_mode or "",
        "tie_margin": tie_margin,
        "label_audit_report": label_audit_report or "",
        "controller_scope": controller_scope or str(train_metrics.get("controller_scope", "")),
        "included_datasets": list(included_datasets or train_metrics.get("included_datasets", [])),
        "excluded_datasets": list(excluded_datasets or train_metrics.get("excluded_datasets", [])),
        "record_count": record_count if record_count is not None else train_metrics.get("train_row_count"),
        "training_data_hash": training_data_hash or str(train_metrics.get("training_data_hash", "")),
        "selection_protocol": selection_protocol or str(train_metrics.get("selection_protocol", "")),
        "export_destination_convention": "paper-new-round23/artifacts/controller_bundle/<bundle_version>/",
    }

    dump_json(bundle_root / "feature_schema.json", feature_schema)
    dump_json(bundle_root / "metadata.json", metadata)
    report = {
        "bundle_version": bundle_version,
        "bundle_root": str(bundle_root),
        "model_family": model_family,
        "training_data_version": training_data_version,
        "feature_version": feature_schema["feature_version"],
        "controller_scope": metadata["controller_scope"],
        "included_datasets": metadata["included_datasets"],
        "excluded_datasets": metadata["excluded_datasets"],
        "record_count": metadata["record_count"],
        "training_data_hash": metadata["training_data_hash"],
        "selection_protocol": metadata["selection_protocol"],
        "exported_models": exported_models,
    }
    dump_json(bundle_root / "bundle_export_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export round23 controller bundle.")
    parser.add_argument("--trained-model-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(CANONICAL_ROUND23_ROOT / "artifacts" / "controller_bundle" / "round23_controller_v1"),
    )
    parser.add_argument("--bundle-version", default="round23_controller_v1")
    parser.add_argument("--model-family", default="lightgbm")
    parser.add_argument("--training-data-version", default="round19_round23_collection_repeat40")
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--target-field", default=None)
    parser.add_argument("--target-mode", default=None)
    parser.add_argument("--tie-margin", type=float, default=None)
    parser.add_argument("--label-audit-report", default=None)
    parser.add_argument("--controller-scope", default=None)
    parser.add_argument("--included-datasets", default=None, help="Comma-separated dataset names used for training.")
    parser.add_argument("--excluded-datasets", default=None, help="Comma-separated dataset names excluded from training.")
    parser.add_argument("--record-count", type=int, default=None)
    parser.add_argument("--training-data-hash", default=None)
    parser.add_argument("--selection-protocol", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export_round23_controller_bundle(
        trained_model_dir=args.trained_model_dir,
        output_dir=args.output_dir,
        bundle_version=args.bundle_version,
        model_family=args.model_family,
        training_data_version=args.training_data_version,
        feature_version=args.feature_version,
        target_field=args.target_field,
        target_mode=args.target_mode,
        tie_margin=args.tie_margin,
        label_audit_report=args.label_audit_report,
        controller_scope=args.controller_scope,
        included_datasets=(
            [value.strip() for value in str(args.included_datasets).split(",") if value.strip()]
            if args.included_datasets
            else None
        ),
        excluded_datasets=(
            [value.strip() for value in str(args.excluded_datasets).split(",") if value.strip()]
            if args.excluded_datasets
            else None
        ),
        record_count=args.record_count,
        training_data_hash=args.training_data_hash,
        selection_protocol=args.selection_protocol,
    )
    print(f"EXPORTED bundle={report['bundle_version']} output_dir={report['bundle_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
