from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from common import DEFAULT_DATASET_DIR, DEFAULT_MODEL_DIR, DEFAULT_SPLIT_DIR, MODEL_TRAIN_ROOT, dump_json, load_yaml, read_jsonl
from features import DEFAULT_INCLUDE_DATASET_ONE_HOT, extract_feature_matrix


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "lightgbm is not installed in the current environment. "
            "Install LightGBM before running train_round22_bandit.py."
        ) from exc
    return lgb


def _load_folds(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_final_test(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _filter_rows_by_context_ids(rows: list[dict[str, Any]], context_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["context_id"]) in context_ids]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.inf


def _resolve_candidate_param_sets(
    *,
    base_params: dict[str, Any],
    train_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = list(train_cfg.get("param_candidates", []))
    if not candidates:
        return [dict(base_params)]
    resolved: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        merged = dict(base_params)
        merged.update({k: v for k, v in candidate.items() if k != "name"})
        merged["_candidate_name"] = str(candidate.get("name", f"candidate_{index}"))
        resolved.append(merged)
    return resolved


def train_models(
    *,
    action_samples_path: str | Path,
    final_test_path: str | Path,
    cv_folds_path: str | Path,
    config_path: str | Path,
    model_output_dir: str | Path,
) -> dict[str, Any]:
    lgb = _require_lightgbm()
    config = load_yaml(config_path)
    rows = read_jsonl(action_samples_path)
    folds_payload = _load_folds(cv_folds_path)
    final_test_payload = _load_final_test(final_test_path)

    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("training", {}))
    include_dataset_one_hot = bool(
        model_cfg.get("include_dataset_one_hot", DEFAULT_INCLUDE_DATASET_ONE_HOT)
    )
    budgets = list(model_cfg.get("budgets", [18, 19, 20, 21, 22]))
    base_lgb_params = dict(model_cfg.get("lightgbm_params", {}))
    candidate_param_sets = _resolve_candidate_param_sets(
        base_params=base_lgb_params,
        train_cfg=train_cfg,
    )

    final_test_ids = set(str(value) for value in final_test_payload["final_test_context_ids"])
    dev_rows = [row for row in rows if str(row["context_id"]) not in final_test_ids]

    cv_results: dict[str, Any] = {}
    selected_params_by_budget: dict[str, Any] = {}
    for budget in budgets:
        budget_rows = [row for row in dev_rows if int(row["action_budget"]) == int(budget)]
        candidate_results: list[dict[str, Any]] = []
        best_candidate: dict[str, Any] | None = None
        best_candidate_mean_mse = math.inf
        for candidate in candidate_param_sets:
            candidate_name = str(candidate.get("_candidate_name", "unnamed"))
            fit_params = {k: v for k, v in candidate.items() if not str(k).startswith("_")}
            fold_scores: list[dict[str, Any]] = []
            for fold in folds_payload["folds"]:
                val_ids = set(str(value) for value in fold["validation_context_ids"])
                train_ids = set(str(value) for value in fold["training_context_ids"])
                train_rows = _filter_rows_by_context_ids(budget_rows, train_ids)
                val_rows = _filter_rows_by_context_ids(budget_rows, val_ids)
                feature_names, train_x = extract_feature_matrix(
                    train_rows,
                    include_dataset_one_hot=include_dataset_one_hot,
                )
                _, val_x = extract_feature_matrix(
                    val_rows,
                    include_dataset_one_hot=include_dataset_one_hot,
                )
                train_y = [float(row["reward"]) for row in train_rows]
                val_y = [float(row["reward"]) for row in val_rows]
                reg = lgb.LGBMRegressor(**fit_params)
                reg.fit(train_x, train_y)
                preds = reg.predict(val_x) if val_x else []
                mse = (
                    sum((pred - gold) ** 2 for pred, gold in zip(preds, val_y)) / len(val_y)
                    if val_y
                    else 0.0
                )
                fold_scores.append(
                    {
                        "fold_index": int(fold["fold_index"]),
                        "train_count": len(train_rows),
                        "val_count": len(val_rows),
                        "mse": mse,
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
            raise RuntimeError(f"No valid candidate parameters produced for budget={budget}")
        cv_results[str(budget)] = candidate_results
        selected_params_by_budget[str(budget)] = best_candidate

    output_root = Path(model_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trained_models: dict[str, Any] = {}
    feature_importance_payload: dict[str, Any] = {}

    for budget in budgets:
        budget_rows = [row for row in dev_rows if int(row["action_budget"]) == int(budget)]
        feature_names, train_x = extract_feature_matrix(
            budget_rows,
            include_dataset_one_hot=include_dataset_one_hot,
        )
        train_y = [float(row["reward"]) for row in budget_rows]
        best_params = dict(selected_params_by_budget[str(budget)]["params"])
        reg = lgb.LGBMRegressor(**best_params)
        reg.fit(train_x, train_y)
        booster = reg.booster_
        model_path = output_root / f"model_k{budget}.txt"
        booster.save_model(str(model_path))
        trained_models[str(budget)] = str(model_path)
        feature_importance_payload[str(budget)] = {
            "feature_names": feature_names,
            "gain_importance": booster.feature_importance(importance_type="gain").tolist(),
            "split_importance": booster.feature_importance(importance_type="split").tolist(),
        }

    report = {
        "action_samples_path": str(Path(action_samples_path).resolve()),
        "final_test_path": str(Path(final_test_path).resolve()),
        "cv_folds_path": str(Path(cv_folds_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "train_context_count": len({str(row["context_id"]) for row in dev_rows}),
        "train_row_count": len(dev_rows),
        "final_test_context_count": len(final_test_ids),
        "include_dataset_one_hot": include_dataset_one_hot,
        "budgets": budgets,
        "cv_results": cv_results,
        "selected_params_by_budget": selected_params_by_budget,
        "trained_models": trained_models,
        "feature_importance": feature_importance_payload,
    }
    dump_json(output_root / "train_metrics.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train round22 contextual-bandit LightGBM models.")
    parser.add_argument(
        "--action-samples",
        default=str(DEFAULT_DATASET_DIR / "round22_action_samples.jsonl"),
    )
    parser.add_argument(
        "--final-test",
        default=str(DEFAULT_SPLIT_DIR / "round22_final_test_contexts.json"),
    )
    parser.add_argument(
        "--cv-folds",
        default=str(DEFAULT_SPLIT_DIR / "round22_cv_folds.json"),
    )
    parser.add_argument(
        "--config",
        default=str(MODEL_TRAIN_ROOT / "configs" / "train_round22_bandit.yaml"),
    )
    parser.add_argument("--model-output-dir", default=str(DEFAULT_MODEL_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_models(
        action_samples_path=args.action_samples,
        final_test_path=args.final_test,
        cv_folds_path=args.cv_folds,
        config_path=args.config,
        model_output_dir=args.model_output_dir,
    )
    print(
        f"TRAINED budgets={len(report['budgets'])} train_rows={report['train_row_count']} "
        f"train_contexts={report['train_context_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
