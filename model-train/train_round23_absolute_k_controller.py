from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from common import (
    BUDGETS,
    DEFAULT_ROUND23_DATASET_DIR,
    DEFAULT_ROUND23_MODEL_DIR,
    DEFAULT_ROUND23_SPLIT_DIR,
    dump_json,
    load_yaml,
    read_jsonl,
)
from features import extract_feature_matrix_with_fields
from round23_controller_models import build_regressor, model_file_extension, require_model_family, save_regressor
from round23_feature_sets import resolve_feature_spec


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "train_round23_absolute_k_controller.yaml"


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _filter_rows_by_context_ids(rows: list[dict[str, Any]], context_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["context_id"]) in context_ids]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.inf


def _resolve_candidate_param_sets(*, family: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = require_model_family(family)
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))
    base_params = dict(model_cfg.get("family_params", {}).get(normalized, {}))
    raw_candidates = list(training_cfg.get("param_candidates", {}).get(normalized, []))
    if not raw_candidates:
        return [dict(base_params, _candidate_name=f"{normalized}_default")]
    resolved: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_candidates):
        merged = dict(base_params)
        merged.update({key: value for key, value in candidate.items() if key != "name"})
        merged["_candidate_name"] = str(candidate.get("name", f"{normalized}_{index}"))
        resolved.append(merged)
    return resolved


def train_absolute_k_models(
    *,
    controller_samples_path: str | Path,
    final_test_path: str | Path,
    unseen_test_path: str | Path | None,
    cv_folds_path: str | Path,
    config_path: str | Path,
    model_output_dir: str | Path,
    model_family: str,
    feature_version: str | None = None,
    target_field: str = "reward_round23_controller",
) -> dict[str, Any]:
    normalized_family = require_model_family(model_family)
    config = load_yaml(config_path)
    model_cfg = dict(config.get("model", {}))
    feature_spec = resolve_feature_spec(
        feature_version=feature_version or model_cfg.get("feature_version"),
        feature_fields=list(model_cfg.get("feature_fields", [])) or None,
        include_dataset_onehot=model_cfg.get("include_dataset_one_hot"),
        onehot_order=list(model_cfg.get("onehot_order", [])) or None,
    )
    absolute_budgets = [int(value) for value in model_cfg.get("absolute_budgets", BUDGETS)]

    rows = read_jsonl(controller_samples_path)
    if rows and target_field not in rows[0]:
        raise KeyError(f"Requested target_field={target_field!r} is not present in controller samples")
    folds_payload = _load_payload(cv_folds_path)
    final_test_payload = _load_payload(final_test_path)
    final_test_ids = set(str(value) for value in final_test_payload["final_test_context_ids"])
    unseen_test_ids: set[str] = set()
    if unseen_test_path is not None:
        unseen_test_payload = _load_payload(unseen_test_path)
        unseen_test_ids = set(str(value) for value in unseen_test_payload["unseen_test_context_ids"])
    excluded_ids = final_test_ids | unseen_test_ids
    dev_rows = [row for row in rows if str(row["context_id"]) not in excluded_ids]

    candidate_param_sets = _resolve_candidate_param_sets(family=normalized_family, config=config)
    cv_results: dict[str, Any] = {}
    selected_params_by_budget: dict[str, Any] = {}

    for budget in absolute_budgets:
        action_rows = [row for row in dev_rows if int(row["target_budget"]) == int(budget)]
        candidate_results: list[dict[str, Any]] = []
        best_candidate: dict[str, Any] | None = None
        best_candidate_mean_mse = math.inf
        for candidate in candidate_param_sets:
            candidate_name = str(candidate.get("_candidate_name", "unnamed"))
            fit_params = {key: value for key, value in candidate.items() if not str(key).startswith("_")}
            fold_scores: list[dict[str, Any]] = []
            for fold in folds_payload["folds"]:
                val_ids = set(str(value) for value in fold["validation_context_ids"])
                train_ids = set(str(value) for value in fold["training_context_ids"])
                train_rows = _filter_rows_by_context_ids(action_rows, train_ids)
                val_rows = _filter_rows_by_context_ids(action_rows, val_ids)
                feature_names, train_x = extract_feature_matrix_with_fields(
                    train_rows,
                    state_fields=feature_spec.feature_fields,
                    include_dataset_one_hot=feature_spec.include_dataset_onehot,
                    dataset_order=feature_spec.onehot_order,
                )
                _, val_x = extract_feature_matrix_with_fields(
                    val_rows,
                    state_fields=feature_spec.feature_fields,
                    include_dataset_one_hot=feature_spec.include_dataset_onehot,
                    dataset_order=feature_spec.onehot_order,
                )
                train_y = [float(row[target_field]) for row in train_rows]
                val_y = [float(row[target_field]) for row in val_rows]
                reg = build_regressor(family=normalized_family, params=fit_params)
                reg.fit(train_x, train_y)
                preds = reg.predict(val_x) if val_x else []
                mse = (
                    sum((float(pred) - gold) ** 2 for pred, gold in zip(preds, val_y)) / len(val_y)
                    if val_y
                    else 0.0
                )
                fold_scores.append(
                    {
                        "fold_index": int(fold["fold_index"]),
                        "train_count": len(train_rows),
                        "val_count": len(val_rows),
                        "mse": mse,
                        "feature_count": len(feature_names),
                    }
                )
            mean_mse = _mean([float(fold["mse"]) for fold in fold_scores])
            candidate_results.append(
                {
                    "candidate_name": candidate_name,
                    "params": fit_params,
                    "fold_scores": fold_scores,
                    "mean_mse": mean_mse,
                }
            )
            if mean_mse < best_candidate_mean_mse:
                best_candidate = {
                    "candidate_name": candidate_name,
                    "params": fit_params,
                    "mean_mse": mean_mse,
                }
                best_candidate_mean_mse = mean_mse
        if best_candidate is None:
            raise RuntimeError(f"No valid candidate parameters produced for target_budget={budget}")
        cv_results[str(budget)] = candidate_results
        selected_params_by_budget[str(budget)] = best_candidate

    output_root = Path(model_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trained_models: dict[str, Any] = {}
    feature_importance_payload: dict[str, Any] = {}

    for budget in absolute_budgets:
        action_rows = [row for row in dev_rows if int(row["target_budget"]) == int(budget)]
        feature_names, train_x = extract_feature_matrix_with_fields(
            action_rows,
            state_fields=feature_spec.feature_fields,
            include_dataset_one_hot=feature_spec.include_dataset_onehot,
            dataset_order=feature_spec.onehot_order,
        )
        train_y = [float(row[target_field]) for row in action_rows]
        best_params = dict(selected_params_by_budget[str(budget)]["params"])
        reg = build_regressor(family=normalized_family, params=best_params)
        reg.fit(train_x, train_y)
        extension = model_file_extension(normalized_family)
        model_path = output_root / f"model_k{budget}{extension}"
        save_regressor(family=normalized_family, model=reg, path=model_path)
        trained_models[str(budget)] = str(model_path)

        importance_payload = {"feature_names": feature_names}
        if normalized_family == "lightgbm":
            booster = reg.booster_
            importance_payload["gain_importance"] = booster.feature_importance(importance_type="gain").tolist()
            importance_payload["split_importance"] = booster.feature_importance(importance_type="split").tolist()
        elif normalized_family in {"randomforest", "extratrees", "xgboost"} and hasattr(reg, "feature_importances_"):
            importance_payload["feature_importances"] = [float(value) for value in reg.feature_importances_.tolist()]
        feature_importance_payload[str(budget)] = importance_payload

    report = {
        "controller_samples_path": str(Path(controller_samples_path).resolve()),
        "final_test_path": str(Path(final_test_path).resolve()),
        "cv_folds_path": str(Path(cv_folds_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "train_context_count": len({str(row["context_id"]) for row in dev_rows}),
        "train_row_count": len(dev_rows),
        "final_test_context_count": len(final_test_ids),
        "unseen_test_context_count": len(unseen_test_ids),
        "excluded_context_count": len(excluded_ids),
        "feature_version": feature_spec.feature_version,
        "include_dataset_one_hot": feature_spec.include_dataset_onehot,
        "onehot_order": feature_spec.onehot_order,
        "feature_fields": feature_spec.feature_fields,
        "total_features": feature_spec.total_features,
        "absolute_budgets": absolute_budgets,
        "model_family": normalized_family,
        "target_field": target_field,
        "target_mode": str(rows[0].get("label_target_mode", "")) if rows else "",
        "tie_margin": rows[0].get("tie_margin") if rows else None,
        "training_data_version": str(rows[0].get("training_data_version", "")) if rows else "",
        "policy_parameterization": "absolute_k",
        "cv_results": cv_results,
        "selected_params_by_budget": selected_params_by_budget,
        "trained_models": trained_models,
        "feature_importance": feature_importance_payload,
    }
    dump_json(output_root / "train_metrics.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train round23 one-shot absolute-k models.")
    parser.add_argument("--controller-samples", default=str(DEFAULT_ROUND23_DATASET_DIR / "round23_controller_samples.jsonl"))
    parser.add_argument("--final-test", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_final_test_contexts.json"))
    parser.add_argument("--unseen-test", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_unseen_test_contexts.json"))
    parser.add_argument("--cv-folds", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_cv_folds.json"))
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--model-output-dir", default=str(DEFAULT_ROUND23_MODEL_DIR))
    parser.add_argument("--model-family", default="lightgbm")
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--target-field", default="reward_round23_controller")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_absolute_k_models(
        controller_samples_path=args.controller_samples,
        final_test_path=args.final_test,
        unseen_test_path=args.unseen_test,
        cv_folds_path=args.cv_folds,
        config_path=args.config,
        model_output_dir=args.model_output_dir,
        model_family=args.model_family,
        feature_version=args.feature_version,
        target_field=args.target_field,
    )
    print(
        f"TRAINED round23-absk family={report['model_family']} feature_version={report['feature_version']} "
        f"budgets={len(report['absolute_budgets'])} train_rows={report['train_row_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
