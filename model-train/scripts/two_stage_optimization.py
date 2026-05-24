#!/usr/bin/env python3
"""Two-stage model optimization.

Stage 1: Compare models with same base training params
Stage 2: Tune params on best model to find optimal config
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data/ready/full-500"
MODEL_OUTPUT_DIR = Path("d:/model_train_output/two_stage")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_FEATURES = [
    "shape_score",
    "private_mean_length",
    "private_p75_length",
    "private_length_iqr",
    "support_mean_at_k20",
    "coverage_mean_at_k20",
    "coverage_p25_at_k20",
    "genericity_mean_at_k20",
]
DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
BUDGETS = [18, 19, 20, 21, 22]

# Fixed base params for fair model comparison
BASE_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.04,
    "min_samples_leaf": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def load_action_samples(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_feature_vector(record: dict) -> list[float]:
    ctx_features = [record[f] for f in CONTEXT_FEATURES]
    onehot = [1.0 if d == record["dataset_name"] else 0.0 for d in DATASET_ORDER]
    return ctx_features + onehot


def predict_best_budget(models: dict, feature_vector: list[float]) -> tuple[int, dict]:
    X = np.array([feature_vector], dtype=np.float64)
    rewards = {}
    for k, model in models.items():
        if hasattr(model, 'predict'):
            rewards[k] = float(model.predict(X)[0])
    best_k = max(rewards, key=rewards.get)
    return best_k, rewards


def evaluate_policy(models: dict, samples: list[dict], X_all: np.ndarray) -> dict:
    context_data = {}
    for i, r in enumerate(samples):
        cid = r["context_id"]
        if cid not in context_data:
            context_data[cid] = {"features": X_all[i], "true_rewards": {}}
        context_data[cid]["true_rewards"][r["action_budget"]] = r["reward"]

    correct = 0
    total = 0
    regret_sum = 0.0

    for cid, ctx in context_data.items():
        true_rewards = ctx["true_rewards"]
        oracle_k = max(true_rewards, key=true_rewards.get)
        pred_k, _ = predict_best_budget(models, ctx["features"])

        if pred_k == oracle_k:
            correct += 1
        total += 1
        regret_sum += true_rewards[oracle_k] - true_rewards.get(pred_k, 0)

    return {
        "oracle_accuracy": correct / total * 100 if total > 0 else 0,
        "avg_regret": regret_sum / total if total > 0 else 0,
    }


def train_model_for_budget(model_class, params: dict, X_train_k: np.ndarray, y_train_k: np.ndarray) -> tuple:
    """Train single model for one budget with 5-fold CV."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []
    fold_models = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(X_train_k)):
        model = model_class(**params)
        model.fit(X_train_k[tr_idx], y_train_k[tr_idx])
        preds = model.predict(X_train_k[va_idx])
        mae = float(np.mean(np.abs(preds - y_train_k[va_idx])))
        fold_maes.append(mae)
        fold_models.append(model)

    cv_mae = sum(fold_maes) / len(fold_maes)
    final_model = model_class(**params)
    final_model.fit(X_train_k, y_train_k)

    return cv_mae, final_model, fold_maes


def stage1_compare_models() -> dict:
    """Stage 1: Compare different models with same base params."""
    print("\n" + "=" * 70)
    print("STAGE 1: Compare Models with Same Base Parameters")
    print("=" * 70)

    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    # Define models to compare (all with same base params structure)
    models_config = {
        "XGBoost": {
            "module": "xgboost",
            "class": "XGBRegressor",
            "params": {
                "objective": "reg:absoluteerror",
                "max_depth": 4,
                "learning_rate": 0.04,
                "n_estimators": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": 0,
            }
        },
        "CatBoost": {
            "module": "catboost",
            "class": "CatBoostRegressor",
            "params": {
                "loss_function": "MAE",
                "depth": 4,
                "learning_rate": 0.04,
                "iterations": 80,
                "subsample": 0.8,
                "random_seed": 42,
                "verbose": False,
            }
        },
        "LightGBM": {
            "module": "lightgbm",
            "class": "LGBMRegressor",
            "params": {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "num_leaves": 15,
                "learning_rate": 0.04,
                "n_estimators": 80,
                "min_child_samples": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbose": -1,
                "random_state": 42,
            }
        },
        "GradientBoosting": {
            "module": "sklearn.ensemble",
            "class": "GradientBoostingRegressor",
            "params": {
                "loss": "absolute_error",
                "max_depth": 4,
                "learning_rate": 0.04,
                "n_estimators": 80,
                "min_samples_leaf": 10,
                "subsample": 0.8,
                "random_state": 42,
            }
        },
        "RandomForest": {
            "module": "sklearn.ensemble",
            "class": "RandomForestRegressor",
            "params": {
                "criterion": "absolute_error",
                "max_depth": 8,
                "n_estimators": 80,
                "min_samples_leaf": 5,
                "random_state": 42,
            }
        },
        "ExtraTrees": {
            "module": "sklearn.ensemble",
            "class": "ExtraTreesRegressor",
            "params": {
                "criterion": "absolute_error",
                "max_depth": 8,
                "n_estimators": 80,
                "min_samples_leaf": 5,
                "random_state": 42,
            }
        },
        "HistGradientBoosting": {
            "module": "sklearn.ensemble",
            "class": "HistGradientBoostingRegressor",
            "params": {
                "loss": "absolute_error",
                "max_depth": 4,
                "learning_rate": 0.04,
                "max_iter": 80,
                "min_samples_leaf": 10,
                "random_state": 42,
            }
        },
    }

    results = {}

    for name, config in models_config.items():
        print(f"\n--- Training {name} ---")

        try:
            import importlib
            parts = config["module"].split(".")
            if len(parts) == 1:
                module = importlib.import_module(parts[0])
            else:
                module = importlib.import_module(parts[0])
                for p in parts[1:]:
                    module = getattr(module, p)
            model_class = getattr(module, config["class"])

            models = {}
            cv_scores = {}

            for k in BUDGETS:
                train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
                X_train_k = X_train[train_k_idx]
                y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

                cv_mae, model, fold_maes = train_model_for_budget(model_class, config["params"], X_train_k, y_train_k)
                cv_scores[k] = cv_mae
                models[k] = model

            val_results = evaluate_policy(models, val_samples, X_val)
            test_results = evaluate_policy(models, test_samples, X_test)

            print(f"  Val: {val_results['oracle_accuracy']:.1f}%, Regret: {val_results['avg_regret']:.6f}")
            print(f"  Test: {test_results['oracle_accuracy']:.1f}%, Regret: {test_results['avg_regret']:.6f}")

            results[name] = {
                "cv_scores": cv_scores,
                "val_accuracy": val_results["oracle_accuracy"],
                "val_regret": val_results["avg_regret"],
                "test_accuracy": test_results["oracle_accuracy"],
                "test_regret": test_results["avg_regret"],
                "config": config,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    return results


def stage2_tune_best_model(best_model_name: str, best_config: dict) -> dict:
    """Stage 2: Tune parameters on best model."""
    print("\n" + "=" * 70)
    print(f"STAGE 2: Tune Parameters for {best_model_name}")
    print("=" * 70)

    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    # Parameter grids for tuning
    param_grids = {
        "XGBoost": [
            {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 50, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 80, "reg_alpha": 0.05, "reg_lambda": 0.05},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 60, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "reg_alpha": 0.05, "reg_lambda": 0.05},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 70, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {"max_depth": 2, "learning_rate": 0.04, "n_estimators": 60, "reg_alpha": 0.15, "reg_lambda": 0.15},
        ],
        "CatBoost": [
            {"depth": 3, "learning_rate": 0.03, "iterations": 50, "l2_leaf_reg": 3},
            {"depth": 3, "learning_rate": 0.05, "iterations": 80, "l2_leaf_reg": 5},
            {"depth": 4, "learning_rate": 0.03, "iterations": 60, "l2_leaf_reg": 3},
            {"depth": 4, "learning_rate": 0.05, "iterations": 100, "l2_leaf_reg": 5},
            {"depth": 5, "learning_rate": 0.03, "iterations": 70, "l2_leaf_reg": 3},
            {"depth": 2, "learning_rate": 0.04, "iterations": 60, "l2_leaf_reg": 5},
        ],
        "LightGBM": [
            {"num_leaves": 10, "learning_rate": 0.03, "n_estimators": 50, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_samples": 15},
            {"num_leaves": 10, "learning_rate": 0.05, "n_estimators": 80, "reg_alpha": 0.05, "reg_lambda": 0.05, "min_child_samples": 15},
            {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 60, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_samples": 10},
            {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 100, "reg_alpha": 0.05, "reg_lambda": 0.05, "min_child_samples": 10},
            {"num_leaves": 20, "learning_rate": 0.03, "n_estimators": 70, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_samples": 10},
            {"num_leaves": 8, "learning_rate": 0.04, "n_estimators": 60, "reg_alpha": 0.15, "reg_lambda": 0.15, "min_child_samples": 20},
        ],
        "GradientBoosting": [
            {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 50, "min_samples_leaf": 15},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 80, "min_samples_leaf": 15},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 60, "min_samples_leaf": 10},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "min_samples_leaf": 10},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 70, "min_samples_leaf": 15},
        ],
        "RandomForest": [
            {"max_depth": 6, "n_estimators": 60, "min_samples_leaf": 5},
            {"max_depth": 8, "n_estimators": 80, "min_samples_leaf": 5},
            {"max_depth": 10, "n_estimators": 100, "min_samples_leaf": 3},
            {"max_depth": 6, "n_estimators": 100, "min_samples_leaf": 8},
            {"max_depth": 4, "n_estimators": 60, "min_samples_leaf": 10},
        ],
        "ExtraTrees": [
            {"max_depth": 6, "n_estimators": 60, "min_samples_leaf": 5},
            {"max_depth": 8, "n_estimators": 80, "min_samples_leaf": 5},
            {"max_depth": 10, "n_estimators": 100, "min_samples_leaf": 3},
            {"max_depth": 6, "n_estimators": 100, "min_samples_leaf": 8},
        ],
        "HistGradientBoosting": [
            {"max_depth": 3, "learning_rate": 0.03, "max_iter": 50, "min_samples_leaf": 15},
            {"max_depth": 3, "learning_rate": 0.05, "max_iter": 80, "min_samples_leaf": 15},
            {"max_depth": 4, "learning_rate": 0.03, "max_iter": 60, "min_samples_leaf": 10},
            {"max_depth": 4, "learning_rate": 0.05, "max_iter": 100, "min_samples_leaf": 10},
            {"max_depth": 5, "learning_rate": 0.03, "max_iter": 70, "min_samples_leaf": 15},
        ],
    }

    import importlib
    parts = best_config["module"].split(".")
    if len(parts) == 1:
        module = importlib.import_module(parts[0])
    else:
        module = importlib.import_module(parts[0])
        for p in parts[1:]:
            module = getattr(module, p)
    model_class = getattr(module, best_config["class"])

    param_grid = param_grids.get(best_model_name, [])
    results = {}

    for i, params in enumerate(param_grid):
        param_name = f"config_{i+1}"
        print(f"\n--- {param_name}: {params} ---")

        try:
            models = {}
            cv_scores = {}

            for k in BUDGETS:
                train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
                X_train_k = X_train[train_k_idx]
                y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

                cv_mae, model, _ = train_model_for_budget(model_class, params, X_train_k, y_train_k)
                cv_scores[k] = cv_mae
                models[k] = model

            val_results = evaluate_policy(models, val_samples, X_val)
            test_results = evaluate_policy(models, test_samples, X_test)

            print(f"  Val: {val_results['oracle_accuracy']:.1f}%, Regret: {val_results['avg_regret']:.6f}")
            print(f"  Test: {test_results['oracle_accuracy']:.1f}%, Regret: {test_results['avg_regret']:.6f}")

            results[param_name] = {
                "params": params,
                "cv_scores": cv_scores,
                "val_accuracy": val_results["oracle_accuracy"],
                "val_regret": val_results["avg_regret"],
                "test_accuracy": test_results["oracle_accuracy"],
                "test_regret": test_results["avg_regret"],
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[param_name] = {"error": str(e)}

    return results


def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("TWO-STAGE MODEL OPTIMIZATION")
    print("Stage 1: Find best model architecture")
    print("Stage 2: Tune parameters on best model")
    print("=" * 70)

    # Stage 1: Compare models
    stage1_results = stage1_compare_models()

    # Print Stage 1 summary
    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY: Model Comparison")
    print("=" * 70)
    print(f"{'Model':<25} {'Val Acc':>10} {'Val Regret':>12} {'Test Acc':>10} {'Test Regret':>12}")
    print("-" * 70)

    valid_results = {k: v for k, v in stage1_results.items() if "error" not in v}
    for name, res in sorted(valid_results.items(), key=lambda x: -x[1]["test_accuracy"]):
        print(f"{name:<25} {res['val_accuracy']:>9.1f}% {res['val_regret']:>11.6f} {res['test_accuracy']:>9.1f}% {res['test_regret']:>11.6f}")

    # Find best model
    best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]["test_accuracy"])
    best_config = valid_results[best_model_name]["config"]

    print(f"\n>> Best Model: {best_model_name}")

    # Stage 2: Tune best model
    stage2_results = stage2_tune_best_model(best_model_name, best_config)

    # Print Stage 2 summary
    print("\n" + "=" * 70)
    print(f"STAGE 2 SUMMARY: {best_model_name} Parameter Tuning")
    print("=" * 70)
    print(f"{'Config':<15} {'Val Acc':>10} {'Val Regret':>12} {'Test Acc':>10} {'Test Regret':>12}")
    print("-" * 60)

    valid_s2 = {k: v for k, v in stage2_results.items() if "error" not in v}
    for name, res in sorted(valid_s2.items(), key=lambda x: -x[1]["test_accuracy"]):
        print(f"{name:<15} {res['val_accuracy']:>9.1f}% {res['val_regret']:>11.6f} {res['test_accuracy']:>9.1f}% {res['test_regret']:>11.6f}")

    # Find best params
    best_params_name = max(valid_s2.keys(), key=lambda k: valid_s2[k]["test_accuracy"])
    best_params = valid_s2[best_params_name]

    print(f"\n>> Best Params: {best_params['params']}")
    print(f">> Best Test Accuracy: {best_params['test_accuracy']:.1f}%")
    print(f">> Best Test Regret: {best_params['test_regret']:.6f}")

    # Save results
    all_results = {
        "stage1_model_comparison": stage1_results,
        "stage2_tuning": {
            "best_model": best_model_name,
            "best_config": best_config,
            "results": stage2_results,
            "optimal_params": best_params["params"],
            "optimal_test_accuracy": best_params["test_accuracy"],
            "optimal_test_regret": best_params["test_regret"],
        }
    }

    output_file = MODEL_OUTPUT_DIR / "two_stage_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()