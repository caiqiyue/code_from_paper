#!/usr/bin/env python3
"""Comprehensive model comparison with extended parameter tuning.

Stage 1: Compare 15+ models with same base params
Stage 2: Tune top models with extended param grids
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

# Import all models directly
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor, ExtraTreesRegressor,
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge

DATA_DIR = Path(__file__).resolve().parents[1] / "data/ready/full-500"
MODEL_OUTPUT_DIR = Path("d:/model_train_output/comprehensive")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_FEATURES = [
    "shape_score", "private_mean_length", "private_p75_length", "private_length_iqr",
    "support_mean_at_k20", "coverage_mean_at_k20", "coverage_p25_at_k20", "genericity_mean_at_k20",
]
DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
BUDGETS = [18, 19, 20, 21, 22]


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
        elif callable(model):
            rewards[k] = float(model(X)[0])
    return max(rewards, key=rewards.get), rewards


def evaluate_policy(models: dict, samples: list[dict], X_all: np.ndarray) -> dict:
    context_data = {}
    for i, r in enumerate(samples):
        cid = r["context_id"]
        if cid not in context_data:
            context_data[cid] = {"features": X_all[i], "true_rewards": {}}
        context_data[cid]["true_rewards"][r["action_budget"]] = r["reward"]

    correct, total, regret_sum = 0, 0, 0.0
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


def train_model_for_budget(model, X_train_k: np.ndarray, y_train_k: np.ndarray) -> tuple:
    """Train single model for one budget with 5-fold CV."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes, fold_models = [], []

    for tr_idx, va_idx in kfold.split(X_train_k):
        import copy
        m = copy.deepcopy(model)
        m.fit(X_train_k[tr_idx], y_train_k[tr_idx])
        preds = m.predict(X_train_k[va_idx])
        mae = float(np.mean(np.abs(preds - y_train_k[va_idx])))
        fold_maes.append(mae)
        fold_models.append(m)

    cv_mae = sum(fold_maes) / len(fold_maes)
    final_model = copy.deepcopy(model)
    final_model.fit(X_train_k, y_train_k)
    return cv_mae, final_model


def stage1_compare_all_models() -> dict:
    """Stage 1: Compare many models with same base structure."""
    print("\n" + "=" * 70)
    print("STAGE 1: Comprehensive Model Comparison")
    print("=" * 70)

    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    # Define all models to compare
    models_config = {
        # Tree-based
        "XGBoost": lambda: __import__("xgboost").XGBRegressor(
            objective="reg:absoluteerror", max_depth=4, learning_rate=0.04,
            n_estimators=80, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        "CatBoost": lambda: __import__("catboost").CatBoostRegressor(
            loss_function="MAE", depth=4, learning_rate=0.04, iterations=80,
            subsample=0.8, random_seed=42, verbose=False),
        "LightGBM": lambda: __import__("lightgbm").LGBMRegressor(
            objective="regression", metric="mae", boosting_type="gbdt", num_leaves=15,
            learning_rate=0.04, n_estimators=80, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, verbose=-1, random_state=42),
        "GradientBoosting": lambda: __import__("sklearn.ensemble").GradientBoostingRegressor(
            loss="absolute_error", max_depth=4, learning_rate=0.04, n_estimators=80,
            min_samples_leaf=10, subsample=0.8, random_state=42),
        "HistGradientBoosting": lambda: __import__("sklearn.ensemble").HistGradientBoostingRegressor(
            loss="absolute_error", max_depth=4, learning_rate=0.04, max_iter=80,
            min_samples_leaf=10, random_state=42),
        "RandomForest": lambda: __import__("sklearn.ensemble").RandomForestRegressor(
            criterion="absolute_error", max_depth=8, n_estimators=80, min_samples_leaf=5, random_state=42),
        "ExtraTrees": lambda: __import__("sklearn.ensemble").ExtraTreesRegressor(
            criterion="absolute_error", max_depth=8, n_estimators=80, min_samples_leaf=5, random_state=42),
        "AdaBoost": lambda: AdaBoostRegressor(n_estimators=80, learning_rate=0.04, random_state=42),

        # Linear models
        "Ridge": lambda: Ridge(alpha=1.0, random_state=42),
        "Lasso": lambda: Lasso(alpha=0.01, random_state=42),
        "ElasticNet": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "BayesianRidge": lambda: BayesianRidge(),
        "HuberRegressor": lambda: HuberRegressor(epsilon=1.35),

        # Kernel-based
        "SVR": lambda: SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "KernelRidge": lambda: KernelRidge(kernel='rbf', alpha=0.1),
        "GaussianProcess": lambda: GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0), alpha=0.1, random_state=42),

        # Neighbors
        "KNeighbors": lambda: KNeighborsRegressor(n_neighbors=5, weights='distance'),

        # Neural Network
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(50, 25), learning_rate_init=0.04,
                                    max_iter=200, early_stopping=True, random_state=42),

        # Tree
        "DecisionTree": lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
    }

    results = {}

    for name, model_fn in models_config.items():
        print(f"\n--- Training {name} ---")

        try:
            model = model_fn()
            models = {}
            cv_scores = {}

            for k in BUDGETS:
                train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
                X_train_k = X_train[train_k_idx]
                y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

                cv_mae, trained_model = train_model_for_budget(model, X_train_k, y_train_k)
                cv_scores[k] = cv_mae
                models[k] = trained_model

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
                "model_fn_name": name,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    return results


def stage2_extended_tuning(top_models: list, n_top: int = 3) -> dict:
    """Stage 2: Extended parameter tuning for top models."""
    print("\n" + "=" * 70)
    print(f"STAGE 2: Extended Tuning for Top {n_top} Models")
    print("=" * 70)

    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    # Extended param grids for each model type
    extended_params = {
        "LightGBM": [
            {"num_leaves": 8, "learning_rate": 0.02, "n_estimators": 50, "reg_alpha": 0.2, "reg_lambda": 0.2, "min_child_samples": 20},
            {"num_leaves": 10, "learning_rate": 0.025, "n_estimators": 55, "reg_alpha": 0.15, "reg_lambda": 0.15, "min_child_samples": 15},
            {"num_leaves": 12, "learning_rate": 0.03, "n_estimators": 60, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_samples": 12},
            {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 65, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_samples": 10},
            {"num_leaves": 18, "learning_rate": 0.035, "n_estimators": 70, "reg_alpha": 0.08, "reg_lambda": 0.08, "min_child_samples": 10},
            {"num_leaves": 20, "learning_rate": 0.03, "n_estimators": 70, "reg_alpha": 0.1, "reg_lambda": 0.1, "min_child_samples": 10},
            {"num_leaves": 25, "learning_rate": 0.025, "n_estimators": 80, "reg_alpha": 0.12, "reg_lambda": 0.12, "min_child_samples": 8},
            {"num_leaves": 31, "learning_rate": 0.02, "n_estimators": 90, "reg_alpha": 0.15, "reg_lambda": 0.15, "min_child_samples": 10},
        ],
        "ExtraTrees": [
            {"max_depth": 6, "n_estimators": 60, "min_samples_leaf": 5, "bootstrap": False},
            {"max_depth": 8, "n_estimators": 70, "min_samples_leaf": 4, "bootstrap": False},
            {"max_depth": 10, "n_estimators": 80, "min_samples_leaf": 3, "bootstrap": False},
            {"max_depth": 12, "n_estimators": 90, "min_samples_leaf": 3, "bootstrap": False},
            {"max_depth": 8, "n_estimators": 100, "min_samples_leaf": 5, "bootstrap": True, "oob_score": True},
            {"max_depth": 10, "n_estimators": 120, "min_samples_leaf": 4, "bootstrap": False},
            {"max_depth": 6, "n_estimators": 80, "min_samples_leaf": 6, "bootstrap": False, "max_features": 0.8},
            {"max_depth": 8, "n_estimators": 80, "min_samples_leaf": 5, "bootstrap": False, "max_features": 0.7},
        ],
        "RandomForest": [
            {"max_depth": 6, "n_estimators": 60, "min_samples_leaf": 5},
            {"max_depth": 8, "n_estimators": 70, "min_samples_leaf": 5},
            {"max_depth": 10, "n_estimators": 80, "min_samples_leaf": 4},
            {"max_depth": 12, "n_estimators": 90, "min_samples_leaf": 3},
            {"max_depth": 8, "n_estimators": 100, "min_samples_leaf": 5, "bootstrap": True, "oob_score": True},
            {"max_depth": 10, "n_estimators": 120, "min_samples_leaf": 4},
            {"max_depth": 6, "n_estimators": 80, "min_samples_leaf": 6, "max_features": 0.8},
            {"max_depth": 8, "n_estimators": 80, "min_samples_leaf": 5, "max_features": 0.7},
        ],
        "XGBoost": [
            {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 50, "reg_alpha": 0.15, "reg_lambda": 0.15, "subsample": 0.7},
            {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 60, "reg_alpha": 0.1, "reg_lambda": 0.1, "subsample": 0.75},
            {"max_depth": 4, "learning_rate": 0.025, "n_estimators": 60, "reg_alpha": 0.12, "reg_lambda": 0.12, "subsample": 0.75},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 70, "reg_alpha": 0.1, "reg_lambda": 0.1, "subsample": 0.8},
            {"max_depth": 5, "learning_rate": 0.025, "n_estimators": 70, "reg_alpha": 0.12, "reg_lambda": 0.12, "subsample": 0.75},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 80, "reg_alpha": 0.1, "reg_lambda": 0.1, "subsample": 0.8},
            {"max_depth": 3, "learning_rate": 0.04, "n_estimators": 80, "reg_alpha": 0.08, "reg_lambda": 0.08, "subsample": 0.85},
            {"max_depth": 4, "learning_rate": 0.035, "n_estimators": 90, "reg_alpha": 0.08, "reg_lambda": 0.08, "subsample": 0.8},
        ],
        "CatBoost": [
            {"depth": 3, "learning_rate": 0.02, "iterations": 50, "l2_leaf_reg": 4},
            {"depth": 3, "learning_rate": 0.03, "iterations": 60, "l2_leaf_reg": 3},
            {"depth": 4, "learning_rate": 0.025, "iterations": 60, "l2_leaf_reg": 3},
            {"depth": 4, "learning_rate": 0.03, "iterations": 70, "l2_leaf_reg": 3},
            {"depth": 5, "learning_rate": 0.025, "iterations": 70, "l2_leaf_reg": 4},
            {"depth": 5, "learning_rate": 0.03, "iterations": 80, "l2_leaf_reg": 3},
            {"depth": 3, "learning_rate": 0.04, "iterations": 80, "l2_leaf_reg": 2},
            {"depth": 4, "learning_rate": 0.035, "iterations": 90, "l2_leaf_reg": 3},
        ],
        "GradientBoosting": [
            {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 50, "min_samples_leaf": 15},
            {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 60, "min_samples_leaf": 12},
            {"max_depth": 4, "learning_rate": 0.025, "n_estimators": 60, "min_samples_leaf": 12},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 70, "min_samples_leaf": 10},
            {"max_depth": 5, "learning_rate": 0.025, "n_estimators": 70, "min_samples_leaf": 10},
            {"max_depth": 3, "learning_rate": 0.04, "n_estimators": 80, "min_samples_leaf": 10},
            {"max_depth": 4, "learning_rate": 0.035, "n_estimators": 90, "min_samples_leaf": 8},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 100, "min_samples_leaf": 8},
        ],
        "HistGradientBoosting": [
            {"max_depth": 3, "learning_rate": 0.02, "max_iter": 50, "min_samples_leaf": 15},
            {"max_depth": 3, "learning_rate": 0.03, "max_iter": 60, "min_samples_leaf": 12},
            {"max_depth": 4, "learning_rate": 0.025, "max_iter": 60, "min_samples_leaf": 12},
            {"max_depth": 4, "learning_rate": 0.03, "max_iter": 70, "min_samples_leaf": 10},
            {"max_depth": 5, "learning_rate": 0.025, "max_iter": 70, "min_samples_leaf": 10},
            {"max_depth": 4, "learning_rate": 0.04, "max_iter": 80, "min_samples_leaf": 8},
            {"max_depth": 5, "learning_rate": 0.03, "max_iter": 90, "min_samples_leaf": 8},
            {"max_depth": 3, "learning_rate": 0.05, "max_iter": 100, "min_samples_leaf": 10},
        ],
    }

    all_tuning_results = {}

    for model_name in top_models[:n_top]:
        print(f"\n{'='*50}")
        print(f"Tuning {model_name}")
        print(f"{'='*50}")

        if model_name not in extended_params:
            print(f"  No extended params defined for {model_name}, skipping...")
            continue

        tuning_results = {}
        param_grid = extended_params[model_name]

        for i, params in enumerate(param_grid):
            param_name = f"p{i+1}"
            print(f"\n--- {param_name}: {params}")

            try:
                # Create model based on name
                if model_name == "LightGBM":
                    import lightgbm
                    model_fn = lambda p=params: lightgbm.LGBMRegressor(
                        objective="regression", metric="mae", boosting_type="gbdt",
                        verbose=-1, random_state=42, **p)
                elif model_name == "XGBoost":
                    import xgboost
                    model_fn = lambda p=params: xgboost.XGBRegressor(
                        objective="reg:absoluteerror", verbosity=0, random_state=42, **p)
                elif model_name == "CatBoost":
                    import catboost
                    model_fn = lambda p=params: catboost.CatBoostRegressor(
                        loss_function="MAE", random_seed=42, verbose=False, **p)
                elif model_name == "GradientBoosting":
                    from sklearn.ensemble import GradientBoostingRegressor
                    model_fn = lambda p=params: GradientBoostingRegressor(
                        loss="absolute_error", random_state=42, **p)
                elif model_name == "HistGradientBoosting":
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    model_fn = lambda p=params: HistGradientBoostingRegressor(
                        loss="absolute_error", random_state=42, **p)
                elif model_name == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    model_fn = lambda p=params: RandomForestRegressor(
                        criterion="absolute_error", random_state=42, **p)
                elif model_name == "ExtraTrees":
                    from sklearn.ensemble import ExtraTreesRegressor
                    model_fn = lambda p=params: ExtraTreesRegressor(
                        criterion="absolute_error", random_state=42, **p)
                else:
                    continue

                model = model_fn()
                models = {}
                cv_scores = {}

                for k in BUDGETS:
                    train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
                    X_train_k = X_train[train_k_idx]
                    y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

                    cv_mae, trained_model = train_model_for_budget(model, X_train_k, y_train_k)
                    cv_scores[k] = cv_mae
                    models[k] = trained_model

                val_results = evaluate_policy(models, val_samples, X_val)
                test_results = evaluate_policy(models, test_samples, X_test)

                print(f"  Val: {val_results['oracle_accuracy']:.1f}%, Regret: {val_results['avg_regret']:.6f}")
                print(f"  Test: {test_results['oracle_accuracy']:.1f}%, Regret: {test_results['avg_regret']:.6f}")

                tuning_results[param_name] = {
                    "params": params,
                    "cv_scores": cv_scores,
                    "val_accuracy": val_results["oracle_accuracy"],
                    "val_regret": val_results["avg_regret"],
                    "test_accuracy": test_results["oracle_accuracy"],
                    "test_regret": test_results["avg_regret"],
                }

            except Exception as e:
                print(f"  ERROR: {e}")
                tuning_results[param_name] = {"error": str(e)}

        all_tuning_results[model_name] = tuning_results

    return all_tuning_results


def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("COMPREHENSIVE MODEL OPTIMIZATION")
    print("=" * 70)

    # Stage 1
    stage1_results = stage1_compare_all_models()

    # Print Stage 1 summary
    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY: All Models")
    print("=" * 70)
    print(f"{'Model':<25} {'Val Acc':>10} {'Val Regret':>12} {'Test Acc':>10} {'Test Regret':>12}")
    print("-" * 70)

    valid_results = {k: v for k, v in stage1_results.items() if "error" not in v}
    sorted_results = sorted(valid_results.items(), key=lambda x: (-x[1]["test_accuracy"], x[1]["test_regret"]))
    for name, res in sorted_results:
        print(f"{name:<25} {res['val_accuracy']:>9.1f}% {res['val_regret']:>11.6f} {res['test_accuracy']:>9.1f}% {res['test_regret']:>11.6f}")

    # Get top 3 models
    top_models = [name for name, _ in sorted_results[:3]]
    print(f"\n>> Top 3 Models: {top_models}")

    # Stage 2: Extended tuning
    stage2_results = stage2_extended_tuning(top_models, n_top=3)

    # Find overall best
    print("\n" + "=" * 70)
    print("OVERALL BEST CONFIGURATION")
    print("=" * 70)

    all_configs = []
    for model_name, tuning in stage2_results.items():
        for param_name, result in tuning.items():
            if "error" not in result:
                all_configs.append({
                    "model": model_name,
                    "params": param_name,
                    "full_params": result["params"],
                    "test_accuracy": result["test_accuracy"],
                    "test_regret": result["test_regret"],
                })

    if all_configs:
        all_configs.sort(key=lambda x: (-x["test_accuracy"], x["test_regret"]))
        best = all_configs[0]
        print(f"\nBest Model: {best['model']}")
        print(f"Best Params: {best['params']}")
        print(f"Full Params: {best['full_params']}")
        print(f"Test Accuracy: {best['test_accuracy']:.1f}%")
        print(f"Test Regret: {best['test_regret']:.6f}")

        # Stage 2 summary
        print("\n" + "=" * 70)
        print("STAGE 2 SUMMARY: Top Configs")
        print("=" * 70)
        print(f"{'Model':<15} {'Params':>10} {'Test Acc':>10} {'Test Regret':>12}")
        print("-" * 50)
        for cfg in all_configs[:10]:
            print(f"{cfg['model']:<15} {cfg['params']:>10} {cfg['test_accuracy']:>9.1f}% {cfg['test_regret']:>11.6f}")

    # Save results
    output = {
        "stage1_all_models": stage1_results,
        "stage2_extended_tuning": stage2_results,
        "best_config": best if all_configs else None,
    }

    output_file = MODEL_OUTPUT_DIR / "comprehensive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()