#!/usr/bin/env python3
"""Feature importance analysis and reward-difference prediction model.

1. Analyze feature importance for budget prediction
2. Train model predicting reward difference (relative to mean) instead of absolute reward
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parents[1] / "data/ready/full-500"
OUTPUT_DIR = Path("d:/model_train_output/feature_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        rewards[k] = float(model.predict(X)[0])
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


def analyze_feature_importance():
    """Analyze which features are most important for budget prediction."""
    print("\n" + "=" * 70)
    print("PART 1: Feature Importance Analysis")
    print("=" * 70)

    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    feature_names = CONTEXT_FEATURES + [f"dataset_onehot_{d}" for d in DATASET_ORDER]

    # Train a LightGBM model for each budget to predict absolute reward
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 65,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 10,
        "verbose": -1,
        "random_state": 42,
    }

    # Method 1: LightGBM built-in importance
    print("\n--- Method 1: LightGBM Built-in Feature Importance ---")
    feature_importance_by_k = {}

    for k in BUDGETS:
        train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
        X_train_k = X_train[train_k_idx]
        y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_k, y_train_k)

        importance = model.feature_importances_
        feature_importance_by_k[k] = dict(zip(feature_names, importance))

        print(f"\nk={k} Top 5 features:")
        sorted_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5]
        for fname, imp in sorted_imp:
            print(f"  {fname}: {imp:.4f}")

    # Aggregate importance across all budgets
    print("\n--- Aggregated Importance (avg across k) ---")
    agg_importance = {}
    for fname in feature_names:
        vals = [feature_importance_by_k[k].get(fname, 0) for k in BUDGETS]
        agg_importance[fname] = np.mean(vals)

    sorted_agg = sorted(agg_importance.items(), key=lambda x: -x[1])
    print("\nOverall Top 10 features:")
    for fname, imp in sorted_agg[:10]:
        print(f"  {fname}: {imp:.4f}")

    # Method 2: Permutation importance on test set
    print("\n--- Method 2: Permutation Importance (on test set) ---")

    # For each context, determine optimal k
    context_data = {}
    for i, r in enumerate(test_samples):
        cid = r["context_id"]
        if cid not in context_data:
            context_data[cid] = {"features": X_test[i], "true_rewards": {}}
        context_data[cid]["true_rewards"][r["action_budget"]] = r["reward"]

    # Create binary classification: predict if k is optimal
    k_oracle_accuracy = {k: {"correct": 0, "total": 0} for k in BUDGETS}

    for cid, ctx in context_data.items():
        true_rewards = ctx["true_rewards"]
        oracle_k = max(true_rewards, key=true_rewards.get)

        for k in BUDGETS:
            k_oracle_accuracy[k]["total"] += 1
            if oracle_k == k:
                k_oracle_accuracy[k]["correct"] += 1

    print("\nOracle k distribution in test set:")
    for k in BUDGETS:
        total = k_oracle_accuracy[k]["total"]
        correct = k_oracle_accuracy[k]["correct"]
        print(f"  k={k}: optimal {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else f"  k={k}: optimal 0/0")

    # Method 3: Correlation between features and oracle k
    print("\n--- Method 3: Feature-OracleK Correlation ---")

    # Create dataset: for each context, features and oracle_k
    context_features_list = []
    context_oracle_k_list = []
    context_oracle_reward_list = []

    for cid, ctx in context_data.items():
        context_features_list.append(ctx["features"])
        true_rewards = ctx["true_rewards"]
        oracle_k = max(true_rewards, key=true_rewards.get)
        context_oracle_k_list.append(oracle_k)
        context_oracle_reward_list.append(true_rewards[oracle_k])

    X_contexts = np.array(context_features_list, dtype=np.float64)
    y_oracle_k = np.array(context_oracle_k_list, dtype=np.float64)

    # Calculate correlation between each feature and oracle_k
    print("\nFeature correlation with oracle_best_k:")
    correlations = []
    for i, fname in enumerate(feature_names):
        corr = np.corrcoef(X_contexts[:, i], y_oracle_k)[0, 1]
        correlations.append((fname, corr))
        print(f"  {fname}: {corr:.4f}")

    sorted_corr = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
    print("\nTop features by correlation magnitude:")
    for fname, corr in sorted_corr[:5]:
        print(f"  {fname}: |{corr:.4f}|")

    return {
        "feature_importance_by_k": {k: {fk: float(v) for fk, v in vals.items()} for k, vals in feature_importance_by_k.items()},
        "aggregated_importance": {k: float(v) for k, v in agg_importance.items()},
        "correlations": {k: float(v) for k, v in dict(correlations).items()},
        "k_oracle_distribution": {k: {"total": v["total"], "correct": v["correct"]} for k, v in k_oracle_accuracy.items()},
    }


def train_reward_difference_model():
    """Train model predicting reward difference from mean instead of absolute reward."""
    print("\n" + "=" * 70)
    print("PART 2: Reward Difference Prediction Model")
    print("=" * 70)

    train_samples = load_action_samples(DATA_DIR / "train_action_samples.jsonl")
    val_samples = load_action_samples(DATA_DIR / "val_action_samples.jsonl")
    test_samples = load_action_samples(DATA_DIR / "test_action_samples.jsonl")

    X_train = np.array([build_feature_vector(r) for r in train_samples], dtype=np.float64)
    X_val = np.array([build_feature_vector(r) for r in val_samples], dtype=np.float64)
    X_test = np.array([build_feature_vector(r) for r in test_samples], dtype=np.float64)

    # Group by context to compute mean reward
    context_data = {}
    for i, r in enumerate(train_samples):
        cid = r["context_id"]
        if cid not in context_data:
            context_data[cid] = {"rewards": []}
        context_data[cid]["rewards"].append((r["action_budget"], r["reward"]))

    # For each context, compute mean reward and reward differences
    for cid, ctx in context_data.items():
        rewards = [r for _, r in ctx["rewards"]]
        ctx["mean_reward"] = np.mean(rewards)

    # Compare two approaches:
    # Approach A: Predict absolute reward (baseline)
    # Approach B: Predict reward difference from context mean

    print("\n--- Approach A: Absolute Reward Prediction (Baseline) ---")
    baseline_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 65,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 10,
        "verbose": -1,
        "random_state": 42,
    }

    models_a = {}
    for k in BUDGETS:
        train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
        X_train_k = X_train[train_k_idx]
        y_train_k = np.array([train_samples[i]["reward"] for i in train_k_idx], dtype=np.float64)

        model = lgb.LGBMRegressor(**baseline_params)
        model.fit(X_train_k, y_train_k)
        models_a[k] = model

    val_results_a = evaluate_policy(models_a, val_samples, X_val)
    test_results_a = evaluate_policy(models_a, test_samples, X_test)
    print(f"  Val: {val_results_a['oracle_accuracy']:.1f}%, Regret: {val_results_a['avg_regret']:.6f}")
    print(f"  Test: {test_results_a['oracle_accuracy']:.1f}%, Regret: {test_results_a['avg_regret']:.6f}")

    print("\n--- Approach B: Reward Difference from Context Mean ---")

    # Create training data with difference labels
    train_diff_samples = []
    for cid, ctx in context_data.items():
        context_mean = ctx["mean_reward"]
        for k, reward in ctx["rewards"]:
            train_diff_samples.append({
                "context_id": cid,
                "action_budget": k,
                "reward_diff": reward - context_mean,
                "reward": reward,
            })

    # Build index for quick lookup
    diff_samples_idx = {r["context_id"]: [i for i, s in enumerate(train_diff_samples) if s["context_id"] == r["context_id"]]
                        for r in train_diff_samples}

    models_b = {}
    for k in BUDGETS:
        train_k_idx = [i for i, s in enumerate(train_diff_samples) if s["action_budget"] == k]
        X_train_k = X_train[train_k_idx]
        y_train_k = np.array([train_diff_samples[i]["reward_diff"] for i in train_k_idx], dtype=np.float64)

        model = lgb.LGBMRegressor(**baseline_params)
        model.fit(X_train_k, y_train_k)
        models_b[k] = model

    # Evaluate approach B
    # For prediction: predicted_reward = predicted_diff + context_mean
    # We need context mean at prediction time - use training set mean per context

    def evaluate_policy_diff(models: dict, samples: list[dict], X_all: np.ndarray, context_means: dict) -> dict:
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
            oracle_reward = true_rewards[oracle_k]

            # Get context mean from training data
            ctx_mean = context_means.get(cid, np.mean(list(true_rewards.values())))

            # Predict best k using difference model
            X = np.array([ctx["features"]], dtype=np.float64)
            best_k = oracle_k  # placeholder
            best_reward_diff = float('-inf')

            for k, model in models.items():
                pred_diff = float(model.predict(X)[0])
                pred_reward = pred_diff + ctx_mean
                if pred_diff > best_reward_diff:
                    best_reward_diff = pred_diff
                    best_k = k

            if best_k == oracle_k:
                correct += 1
            total += 1
            regret_sum += oracle_reward - true_rewards.get(best_k, oracle_reward)

        return {
            "oracle_accuracy": correct / total * 100 if total > 0 else 0,
            "avg_regret": regret_sum / total if total > 0 else 0,
        }

    # Compute context means from training data
    context_means = {cid: ctx["mean_reward"] for cid, ctx in context_data.items()}

    val_results_b = evaluate_policy_diff(models_b, val_samples, X_val, context_means)
    test_results_b = evaluate_policy_diff(models_b, test_samples, X_test, context_means)
    print(f"  Val: {val_results_b['oracle_accuracy']:.1f}%, Regret: {val_results_b['avg_regret']:.6f}")
    print(f"  Test: {val_results_b['oracle_accuracy']:.1f}%, Regret: {test_results_b['avg_regret']:.6f}")

    # Approach C: Predict reward difference from global mean per k
    print("\n--- Approach C: Predict Reward Difference from K-specific Mean ---")

    # Compute global mean reward for each k
    k_global_mean = {}
    for k in BUDGETS:
        rewards_k = [r["reward"] for r in train_samples if r["action_budget"] == k]
        k_global_mean[k] = np.mean(rewards_k)
    print(f"  K-specific means: {k_global_mean}")

    models_c = {}
    for k in BUDGETS:
        train_k_idx = [i for i, r in enumerate(train_samples) if r["action_budget"] == k]
        X_train_k = X_train[train_k_idx]
        y_train_k = np.array([train_samples[i]["reward"] - k_global_mean[k] for i in train_k_idx], dtype=np.float64)

        model = lgb.LGBMRegressor(**baseline_params)
        model.fit(X_train_k, y_train_k)
        models_c[k] = model

    def evaluate_policy_c(models: dict, samples: list[dict], X_all: np.ndarray, k_means: dict) -> dict:
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
            oracle_reward = true_rewards[oracle_k]

            X = np.array([ctx["features"]], dtype=np.float64)
            best_k = oracle_k
            best_pred_reward = float('-inf')

            for k, model in models.items():
                pred_diff = float(model.predict(X)[0])
                pred_reward = pred_diff + k_means[k]
                if pred_reward > best_pred_reward:
                    best_pred_reward = pred_reward
                    best_k = k

            if best_k == oracle_k:
                correct += 1
            total += 1
            regret_sum += oracle_reward - true_rewards.get(best_k, oracle_reward)

        return {
            "oracle_accuracy": correct / total * 100 if total > 0 else 0,
            "avg_regret": regret_sum / total if total > 0 else 0,
        }

    val_results_c = evaluate_policy_c(models_c, val_samples, X_val, k_global_mean)
    test_results_c = evaluate_policy_c(models_c, test_samples, X_test, k_global_mean)
    print(f"  Val: {val_results_c['oracle_accuracy']:.1f}%, Regret: {val_results_c['avg_regret']:.6f}")
    print(f"  Test: {test_results_c['oracle_accuracy']:.1f}%, Regret: {test_results_c['avg_regret']:.6f}")

    # Approach D: Multi-output regression - predict all k rewards simultaneously
    print("\n--- Approach D: Direct Oracle Classification ---")

    # Instead of predicting reward, train classifier to predict which k is optimal
    # This is a 5-class classification problem

    from sklearn.ensemble import RandomForestClassifier

    # Create training data: features -> optimal k
    oracle_k_data = {}
    for cid in set(r["context_id"] for r in train_samples):
        ctx_samples = [r for r in train_samples if r["context_id"] == cid]
        true_rewards = {r["action_budget"]: r["reward"] for r in ctx_samples}
        oracle_k = max(true_rewards, key=true_rewards.get)
        oracle_k_data[cid] = oracle_k

    # Get context features
    context_features_dict = {}
    for i, r in enumerate(train_samples):
        cid = r["context_id"]
        if cid not in context_features_dict:
            context_features_dict[cid] = X_train[i]

    X_train_ctx = np.array([context_features_dict[cid] for cid in oracle_k_data.keys()], dtype=np.float64)
    y_train_ctx = np.array([oracle_k_data[cid] for cid in oracle_k_data.keys()], dtype=np.int32)

    print(f"  Training oracle classifier: {len(X_train_ctx)} samples")
    print(f"  Oracle k distribution: {np.bincount(y_train_ctx, minlength=23)[18:23]}")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
    )
    clf.fit(X_train_ctx, y_train_ctx)

    # Evaluate on test
    test_context_data = {}
    for i, r in enumerate(test_samples):
        cid = r["context_id"]
        if cid not in test_context_data:
            test_context_data[cid] = {"features": X_test[i], "true_rewards": {}}
        test_context_data[cid]["true_rewards"][r["action_budget"]] = r["reward"]

    X_test_ctx = np.array([test_context_data[cid]["features"] for cid in test_context_data.keys()], dtype=np.float64)
    y_test_true = np.array([
        max(test_context_data[cid]["true_rewards"].keys(),
            key=lambda k: test_context_data[cid]["true_rewards"][k])
        for cid in test_context_data.keys()
    ], dtype=np.int32)

    y_pred = clf.predict(X_test_ctx)

    oracle_acc_d = np.mean(y_pred == y_test_true) * 100

    # Also get predicted rewards for regret calculation
    predicted_rewards = clf.predict_proba(X_test_ctx)  # probabilities for each class
    print(f"  Test Oracle Accuracy: {oracle_acc_d:.1f}%")

    return {
        "approach_a": {"val": val_results_a, "test": test_results_a},
        "approach_b": {"val": val_results_b, "test": test_results_b},
        "approach_c": {"val": val_results_c, "test": test_results_c},
        "approach_d_oracle_accuracy": oracle_acc_d,
    }


def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("FEATURE IMPORTANCE AND REWARD DIFFERENCE ANALYSIS")
    print("=" * 70)

    # Part 1: Feature Importance
    feature_results = analyze_feature_importance()

    # Part 2: Reward Difference Model
    diff_results = train_reward_difference_model()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. Feature Importance (Top 5):")
    sorted_imp = sorted(feature_results["aggregated_importance"].items(), key=lambda x: -x[1])[:5]
    for fname, imp in sorted_imp:
        print(f"   {fname}: {imp:.4f}")

    print("\n2. Feature-OracleK Correlation (Top 5):")
    sorted_corr = sorted(feature_results["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for fname, corr in sorted_corr:
        print(f"   {fname}: {corr:.4f}")

    print("\n3. Reward Prediction Approaches Comparison:")
    print(f"   Approach A (Absolute):     Test Acc={diff_results['approach_a']['test']['oracle_accuracy']:.1f}%, Regret={diff_results['approach_a']['test']['avg_regret']:.6f}")
    print(f"   Approach B (Context Diff): Test Acc={diff_results['approach_b']['test']['oracle_accuracy']:.1f}%, Regret={diff_results['approach_b']['test']['avg_regret']:.6f}")
    print(f"   Approach C (K-specific):   Test Acc={diff_results['approach_c']['test']['oracle_accuracy']:.1f}%, Regret={diff_results['approach_c']['test']['avg_regret']:.6f}")
    print(f"   Approach D (Oracle Class): Test Acc={diff_results['approach_d_oracle_accuracy']:.1f}%")

    # Save results
    output = {
        "feature_importance": feature_results,
        "reward_difference_comparison": {
            "approach_a": {
                "val": {"oracle_accuracy": float(diff_results['approach_a']['val']['oracle_accuracy']), "avg_regret": float(diff_results['approach_a']['val']['avg_regret'])},
                "test": {"oracle_accuracy": float(diff_results['approach_a']['test']['oracle_accuracy']), "avg_regret": float(diff_results['approach_a']['test']['avg_regret'])},
            },
            "approach_b": {
                "val": {"oracle_accuracy": float(diff_results['approach_b']['val']['oracle_accuracy']), "avg_regret": float(diff_results['approach_b']['val']['avg_regret'])},
                "test": {"oracle_accuracy": float(diff_results['approach_b']['test']['oracle_accuracy']), "avg_regret": float(diff_results['approach_b']['test']['avg_regret'])},
            },
            "approach_c": {
                "val": {"oracle_accuracy": float(diff_results['approach_c']['val']['oracle_accuracy']), "avg_regret": float(diff_results['approach_c']['val']['avg_regret'])},
                "test": {"oracle_accuracy": float(diff_results['approach_c']['test']['oracle_accuracy']), "avg_regret": float(diff_results['approach_c']['test']['avg_regret'])},
            },
            "approach_d_oracle_accuracy": float(diff_results['approach_d_oracle_accuracy']),
        },
    }

    output_file = OUTPUT_DIR / "analysis_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()