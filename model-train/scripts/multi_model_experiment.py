#!/usr/bin/env python3
"""Multi-model experiment comparison for budget policy.

Tests: XGBoost, CatBoost, LightGBM variants
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data/ready/full-500"
MODEL_OUTPUT_DIR = Path("d:/model_train_output/experiments")
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
        else:
            rewards[k] = float(model(X)[0])
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
    budget_correct = {k: 0 for k in BUDGETS}
    budget_total = {k: 0 for k in BUDGETS}

    for cid, ctx in context_data.items():
        true_rewards = ctx["true_rewards"]
        oracle_k = max(true_rewards, key=true_rewards.get)

        pred_k, _ = predict_best_budget(models, ctx["features"])

        if pred_k == oracle_k:
            correct += 1
        total += 1
        regret_sum += true_rewards[oracle_k] - true_rewards.get(pred_k, 0)

        budget_total[oracle_k] += 1
        if pred_k == oracle_k:
            budget_correct[oracle_k] += 1

    return {
        "oracle_accuracy": correct / total * 100 if total > 0 else 0,
        "avg_regret": regret_sum / total if total > 0 else 0,
        "budget_correct": budget_correct,
        "budget_total": budget_total,
    }


def run_experiment(name: str, model_configs: list[dict]) -> dict:
    """Run experiment with multiple model configurations."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Load data
    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    results = {}
    for config in model_configs:
        config_name = config["name"]
        print(f"\n--- Config: {config_name} ---")

        try:
            import importlib
            model_module = importlib.import_module(config["module"])
            model_class = getattr(model_module, config["class"])

            models = {}
            cv_scores = {}

            for k in BUDGETS:
                train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
                X_train_k = X_train[train_k_idx]
                y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

                # 5-fold CV
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                fold_maes = []
                for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(X_train_k)):
                    model = model_class(**config["params"])
                    model.fit(X_train_k[tr_idx], y_train_k[tr_idx])
                    preds = model.predict(X_train_k[va_idx])
                    mae = float(np.mean(np.abs(preds - y_train_k[va_idx])))
                    fold_maes.append(mae)

                cv_mae = sum(fold_maes) / len(fold_maes)
                cv_scores[k] = cv_mae

                # Train final model
                final_model = model_class(**config["params"])
                final_model.fit(X_train_k, y_train_k)
                models[k] = final_model

            # Evaluate
            val_results = evaluate_policy(models, val_samples, X_val)
            test_results = evaluate_policy(models, test_samples, X_test)

            print(f"  Val Oracle Acc: {val_results['oracle_accuracy']:.1f}%, Avg Regret: {val_results['avg_regret']:.6f}")
            print(f"  Test Oracle Acc: {test_results['oracle_accuracy']:.1f}%, Avg Regret: {test_results['avg_regret']:.6f}")

            results[config_name] = {
                "cv_scores": cv_scores,
                "val_oracle_accuracy": val_results["oracle_accuracy"],
                "val_avg_regret": val_results["avg_regret"],
                "test_oracle_accuracy": test_results["oracle_accuracy"],
                "test_avg_regret": test_results["avg_regret"],
                "budget_correct": test_results["budget_correct"],
                "budget_total": test_results["budget_total"],
            }

            # Save models
            config_dir = MODEL_OUTPUT_DIR / name / config_name
            config_dir.mkdir(parents=True, exist_ok=True)

            for k, model in models.items():
                if hasattr(model, 'booster_'):
                    model.booster_.save_model(str(config_dir / f"model_k{k}.txt"))
                elif hasattr(model, 'save_model'):
                    model.save_model(str(config_dir / f"model_k{k}.json"))

            # Save metadata
            metadata = {
                "config_name": config_name,
                "module": config["module"],
                "class": config["class"],
                "params": config["params"],
                **results[config_name]
            }
            with open(config_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            results[config_name] = {"error": str(e)}

    return results


def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("MULTI-MODEL EXPERIMENT COMPARISON")
    print("=" * 60)

    experiments = [
        {
            "name": "xgboost_v1",
            "configs": [
                {
                    "name": "xgboost_default",
                    "module": "xgboost",
                    "class": "XGBRegressor",
                    "params": {
                        "objective": "reg:absoluteerror",
                        "max_depth": 6,
                        "learning_rate": 0.05,
                        "n_estimators": 100,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": 42,
                        "verbosity": 0,
                    }
                },
                {
                    "name": "xgboost_shallow",
                    "module": "xgboost",
                    "class": "XGBRegressor",
                    "params": {
                        "objective": "reg:absoluteerror",
                        "max_depth": 3,
                        "learning_rate": 0.03,
                        "n_estimators": 50,
                        "subsample": 0.7,
                        "colsample_bytree": 0.7,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.1,
                        "random_state": 42,
                        "verbosity": 0,
                    }
                },
            ]
        },
        {
            "name": "catboost_v1",
            "configs": [
                {
                    "name": "catboost_default",
                    "module": "catboost",
                    "class": "CatBoostRegressor",
                    "params": {
                        "loss_function": "MAE",
                        "depth": 6,
                        "learning_rate": 0.05,
                        "iterations": 100,
                        "subsample": 0.8,
                        "random_seed": 42,
                        "verbose": False,
                    }
                },
                {
                    "name": "catboost_shallow",
                    "module": "catboost",
                    "class": "CatBoostRegressor",
                    "params": {
                        "loss_function": "MAE",
                        "depth": 4,
                        "learning_rate": 0.03,
                        "iterations": 50,
                        "l2_leaf_reg": 3,
                        "subsample": 0.7,
                        "random_seed": 42,
                        "verbose": False,
                    }
                },
            ]
        },
        {
            "name": "lgbm_v3_v4",
            "configs": [
                {
                    "name": "lgbm_v3_high_reg",
                    "module": "lightgbm",
                    "class": "LGBMRegressor",
                    "params": {
                        "objective": "regression",
                        "metric": "mae",
                        "boosting_type": "gbdt",
                        "num_leaves": 10,
                        "learning_rate": 0.02,
                        "n_estimators": 40,
                        "min_child_samples": 20,
                        "reg_alpha": 0.2,
                        "reg_lambda": 0.2,
                        "subsample": 0.7,
                        "colsample_bytree": 0.7,
                        "verbose": -1,
                        "random_state": 42,
                    }
                },
                {
                    "name": "lgbm_v4_deeper",
                    "module": "lightgbm",
                    "class": "LGBMRegressor",
                    "params": {
                        "objective": "regression",
                        "metric": "mae",
                        "boosting_type": "gbdt",
                        "num_leaves": 20,
                        "learning_rate": 0.04,
                        "n_estimators": 80,
                        "min_child_samples": 10,
                        "reg_alpha": 0.05,
                        "reg_lambda": 0.05,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                        "verbose": -1,
                        "random_state": 42,
                    }
                },
            ]
        },
    ]

    all_results = {}
    for exp in experiments:
        results = run_experiment(exp["name"], exp["configs"])
        all_results[exp["name"]] = results

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: All Experiments")
    print("=" * 60)
    print(f"{'Config':<25} {'Val Acc':>10} {'Val Regret':>12} {'Test Acc':>10} {'Test Regret':>12}")
    print("-" * 70)

    for exp_name, configs in all_results.items():
        for config_name, res in configs.items():
            if "error" not in res:
                print(f"{config_name:<25} {res['val_oracle_accuracy']:>9.1f}% {res['val_avg_regret']:>11.6f} {res['test_oracle_accuracy']:>9.1f}% {res['test_avg_regret']:>11.6f}")

    # Save comparison
    comparison = {
        "experiments": all_results,
        "timestamp": "2026-05-13",
    }
    with open(MODEL_OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {MODEL_OUTPUT_DIR}")
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()