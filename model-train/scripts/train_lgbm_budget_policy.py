#!/usr/bin/env python3
"""Train LightGBM budget policy model from round22 bandit data.

For each k in {18,19,20,21,22}, train a regressor:
  features (12-d) -> reward prediction

At runtime, pick k with highest predicted reward.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data/ready/full-500"
MODEL_DIR = Path("d:/model_train_output/full-500")

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
TOTAL_FEATURES = 12


def load_action_samples(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_dataset_onehot(features: list[float], dataset_name: str) -> list[float]:
    onehot = [1.0 if d == dataset_name else 0.0 for d in DATASET_ORDER]
    return features + onehot


def build_feature_vector(record: dict) -> list[float]:
    ctx_features = [record[f] for f in CONTEXT_FEATURES]
    return append_dataset_onehot(ctx_features, record["dataset_name"])


def predict_best_budget(
    models: dict[int, lgb.LGBMRegressor],
    feature_vector: list[float],
) -> tuple[int, dict[int, float]]:
    X = np.array([feature_vector], dtype=np.float64)
    rewards = {}
    for k, model in models.items():
        rewards[k] = float(model.predict(X)[0])
    best_k = max(rewards, key=rewards.get)
    return best_k, rewards


def evaluate_policy(models: dict[int, lgb.LGBMRegressor], samples: list[dict], X_all: np.ndarray) -> dict:
    """Evaluate full policy on a set of context-level samples."""
    # Group by context_id
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
        oracle_reward = true_rewards[oracle_k]

        pred_k, _ = predict_best_budget(models, ctx["features"])

        if pred_k == oracle_k:
            correct += 1
        total += 1
        regret_sum += oracle_reward - true_rewards[pred_k]

        budget_total[oracle_k] += 1
        if pred_k == oracle_k:
            budget_correct[oracle_k] += 1

    return {
        "oracle_accuracy": correct / total * 100 if total > 0 else 0,
        "avg_regret": regret_sum / total if total > 0 else 0,
        "budget_correct": budget_correct,
        "budget_total": budget_total,
    }


def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("TRAIN LIGHTGBM BUDGET POLICY MODEL (v2 - tuned params)")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    print(f"    Train: {len(train_samples)} samples")
    print(f"    Val:   {len(val_samples)} samples")
    print(f"    Test:  {len(test_samples)} samples")

    # Build feature matrices
    print("\n[2] Building feature matrices...")
    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    print(f"    X_train shape: {X_train.shape}")
    print(f"    X_val shape:   {X_val.shape}")
    print(f"    X_test shape:  {X_test.shape}")

    # Tuned LightGBM params - reduce overfitting
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 15,        # Reduced from 31
        "learning_rate": 0.03,   # Reduced from 0.05
        "n_estimators": 50,      # Reduced from 200
        "min_child_samples": 15, # Increased from 5
        "reg_alpha": 0.1,        # L1 regularization
        "reg_lambda": 0.1,      # L2 regularization
        "subsample": 0.8,       # Row subsampling
        "colsample_bytree": 0.8, # Feature subsampling
        "verbose": -1,
        "random_state": 42,
    }
    print(f"\n[3] LightGBM params (tuned to reduce overfitting):")
    for k, v in params.items():
        if k not in ["verbose"]:
            print(f"    {k}: {v}")

    # Train one model per budget with 5-fold CV on training set
    print("\n[4] Training models per budget (with 5-fold CV)...")
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
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_k[tr_idx], y_train_k[tr_idx])
            preds = model.predict(X_train_k[va_idx])
            mae = float(np.mean(np.abs(preds - y_train_k[va_idx])))
            fold_maes.append(mae)

        cv_mae = sum(fold_maes) / len(fold_maes)
        cv_scores[k] = cv_mae
        print(f"  k={k}: 5-fold CV MAE = {cv_mae:.6f} (folds: {[f'{m:.6f}' for m in fold_maes]})")

        # Train final model on all training data for this budget
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X_train_k, y_train_k)
        models[k] = final_model

    # Evaluate on val set
    print("\n[5] Evaluating on VAL set...")
    val_results = evaluate_policy(models, val_samples, X_val)
    print(f"  Oracle accuracy: {val_results['oracle_accuracy']:.1f}%")
    print(f"  Avg regret:     {val_results['avg_regret']:.6f}")
    print(f"  Per-budget accuracy:")
    for k in BUDGETS:
        t = val_results["budget_total"][k]
        c = val_results["budget_correct"][k]
        if t > 0:
            print(f"    k={k}: {c}/{t} = {c/t*100:.1f}%")

    # Evaluate on test set
    print("\n[6] Evaluating on TEST set...")
    test_results = evaluate_policy(models, test_samples, X_test)
    print(f"  Oracle accuracy: {test_results['oracle_accuracy']:.1f}%")
    print(f"  Avg regret:     {test_results['avg_regret']:.6f}")
    print(f"  Per-budget accuracy:")
    for k in BUDGETS:
        t = test_results["budget_total"][k]
        c = test_results["budget_correct"][k]
        if t > 0:
            print(f"    k={k}: {c}/{t} = {c/t*100:.1f}%")

    # Save models
    print("\n[7] Saving models...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "version": "2.0",
        "feature_names": CONTEXT_FEATURES + [f"dataset_onehot_{d}" for d in DATASET_ORDER],
        "total_features": TOTAL_FEATURES,
        "budgets": BUDGETS,
        "reward_lambda": 0.002,
        "lightgbm_params": {k: v for k, v in params.items() if k != "verbose"},
        "training_data": "round22_bandit_full_summary (500 experiments)",
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "cv_scores": cv_scores,
        "val_oracle_accuracy": val_results["oracle_accuracy"],
        "val_avg_regret": val_results["avg_regret"],
        "test_oracle_accuracy": test_results["oracle_accuracy"],
        "test_avg_regret": test_results["avg_regret"],
    }

    for k, model in models.items():
        model.booster_.save_model(str(MODEL_DIR / f"model_k{k}.txt"))

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_bundle, f, indent=2, ensure_ascii=False)

    print(f"    Models saved to: {MODEL_DIR}")
    print(f"    metadata.json saved")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()