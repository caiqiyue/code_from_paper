from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    BUDGETS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SPLIT_DIR,
    MODEL_TRAIN_ROOT,
    dump_json,
    load_yaml,
    read_jsonl,
)
from features import DEFAULT_INCLUDE_DATASET_ONE_HOT, encode_feature_row
from round19_replay import (
    ExplicitRound19ReplayPolicy,
    build_round19_replay_policy_from_action_samples,
    load_explicit_round19_replay,
    validate_explicit_round19_replay,
)


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "lightgbm is not installed in the current environment. "
            "Install LightGBM before running eval_round22_bandit.py."
        ) from exc
    return lgb


def _load_context_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_final_test(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _budget_reward(context_row: dict[str, Any], budget: int) -> float:
    return float(context_row[f"reward_k{int(budget)}"])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    *,
    action_samples_path: str | Path,
    context_table_path: str | Path,
    final_test_path: str | Path,
    config_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    round19_replay_path: str | Path | None = None,
) -> dict[str, Any]:
    lgb = _require_lightgbm()
    config = load_yaml(config_path)
    model_cfg = dict(config.get("model", {}))
    eval_cfg = dict(config.get("evaluation", {}))
    include_dataset_one_hot = bool(
        model_cfg.get("include_dataset_one_hot", DEFAULT_INCLUDE_DATASET_ONE_HOT)
    )
    budgets = list(model_cfg.get("budgets", BUDGETS))
    fixed_baselines = list(eval_cfg.get("fixed_budget_baselines", [18, 19, 22]))
    require_round19_replay = bool(eval_cfg.get("require_round19_replay", True))

    context_rows = _load_context_rows(context_table_path)
    final_test_ids = set(str(value) for value in _load_final_test(final_test_path)["final_test_context_ids"])
    test_rows = [row for row in context_rows if str(row["context_id"]) in final_test_ids]

    models: dict[int, Any] = {}
    for budget in budgets:
        model_path = Path(model_dir) / f"model_k{budget}.txt"
        models[int(budget)] = lgb.Booster(model_file=str(model_path))

    round19_policy: ExplicitRound19ReplayPolicy | None = None
    if require_round19_replay and not round19_replay_path:
        round19_policy = build_round19_replay_policy_from_action_samples(
            action_samples_path=action_samples_path,
            context_ids=final_test_ids,
        )
    elif round19_replay_path:
        round19_policy = load_explicit_round19_replay(round19_replay_path)
    if round19_policy is not None:
        validate_explicit_round19_replay(context_rows=test_rows, policy=round19_policy)

    per_context: list[dict[str, Any]] = []
    learned_budget_hits = 0
    round19_budget_hits = 0
    learned_better_than_round19 = 0
    round19_better_than_learned = 0
    equal_to_round19 = 0
    for row in test_rows:
        _, feature_values = encode_feature_row(
            row,
            include_dataset_one_hot=include_dataset_one_hot,
        )
        predicted_rewards = {
            int(budget): float(models[int(budget)].predict([feature_values])[0])
            for budget in budgets
        }
        learned_budget = max(predicted_rewards, key=predicted_rewards.get)
        oracle_budget = int(row["oracle_best_k"])
        learned_reward = _budget_reward(row, learned_budget)
        oracle_reward = float(row["oracle_best_reward"])
        replay_budget = round19_policy.predict(row) if round19_policy is not None else None
        replay_reward = _budget_reward(row, replay_budget) if replay_budget is not None else None
        fixed_rewards = {int(b): _budget_reward(row, int(b)) for b in fixed_baselines}
        if int(learned_budget) == oracle_budget:
            learned_budget_hits += 1
        if replay_budget is not None:
            if int(replay_budget) == oracle_budget:
                round19_budget_hits += 1
            if float(learned_reward) > float(replay_reward):
                learned_better_than_round19 += 1
            elif float(learned_reward) < float(replay_reward):
                round19_better_than_learned += 1
            else:
                equal_to_round19 += 1
        per_context.append(
            {
                "context_id": row["context_id"],
                "dataset_name": row["dataset_name"],
                "oracle_best_k": oracle_budget,
                "oracle_best_reward": oracle_reward,
                "learned_budget": int(learned_budget),
                "learned_reward": learned_reward,
                "learned_regret": oracle_reward - learned_reward,
                "round19_budget": replay_budget,
                "round19_reward": replay_reward,
                "fixed_rewards": fixed_rewards,
                "predicted_rewards": predicted_rewards,
            }
        )

    datasetwise: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({str(row["dataset_name"]) for row in per_context}):
        dataset_rows = [row for row in per_context if row["dataset_name"] == dataset_name]
        fixed_summary = {}
        for budget in fixed_baselines:
            rewards = [float(row["fixed_rewards"][int(budget)]) for row in dataset_rows]
            regrets = [float(row["oracle_best_reward"]) - float(row["fixed_rewards"][int(budget)]) for row in dataset_rows]
            fixed_summary[str(budget)] = {
                "mean_reward": _mean(rewards),
                "mean_regret": _mean(regrets),
            }
        learned_budget_accuracy = (
            sum(1 for row in dataset_rows if int(row["learned_budget"]) == int(row["oracle_best_k"])) / len(dataset_rows)
        )
        datasetwise[dataset_name] = {
            "context_count": len(dataset_rows),
            "mean_oracle_reward": sum(row["oracle_best_reward"] for row in dataset_rows) / len(dataset_rows),
            "mean_learned_reward": sum(row["learned_reward"] for row in dataset_rows) / len(dataset_rows),
            "mean_learned_regret": sum(row["learned_regret"] for row in dataset_rows) / len(dataset_rows),
            "learned_budget_accuracy": learned_budget_accuracy,
            "learned_budget_distribution": {
                str(budget): sum(1 for row in dataset_rows if int(row["learned_budget"]) == int(budget))
                for budget in budgets
            },
            "fixed_budget_baselines": fixed_summary,
        }
        if round19_policy is not None:
            datasetwise[dataset_name]["mean_round19_reward"] = (
                sum(float(row["round19_reward"]) for row in dataset_rows) / len(dataset_rows)
            )
            datasetwise[dataset_name]["round19_budget_accuracy"] = (
                sum(1 for row in dataset_rows if int(row["round19_budget"]) == int(row["oracle_best_k"])) / len(dataset_rows)
            )
            datasetwise[dataset_name]["decision_consistency"] = {
                "same_budget_choice": sum(
                    1 for row in dataset_rows if int(row["learned_budget"]) == int(row["round19_budget"])
                ),
                "learned_better_than_round19": sum(
                    1 for row in dataset_rows if float(row["learned_reward"]) > float(row["round19_reward"])
                ),
                "round19_better_than_learned": sum(
                    1 for row in dataset_rows if float(row["learned_reward"]) < float(row["round19_reward"])
                ),
                "equal_reward": sum(
                    1 for row in dataset_rows if float(row["learned_reward"]) == float(row["round19_reward"])
                ),
            }

    fixed_baseline_summary = {}
    for budget in fixed_baselines:
        rewards = [float(row["fixed_rewards"][int(budget)]) for row in per_context]
        regrets = [float(row["oracle_best_reward"]) - float(row["fixed_rewards"][int(budget)]) for row in per_context]
        fixed_baseline_summary[str(budget)] = {
            "mean_reward": _mean(rewards),
            "mean_regret": _mean(regrets),
        }

    learned_budget_accuracy = (
        learned_budget_hits / len(per_context) if per_context else 0.0
    )

    report = {
        "context_count": len(per_context),
        "action_samples_path": str(Path(action_samples_path).resolve()),
        "budgets": budgets,
        "fixed_budget_baselines": fixed_baselines,
        "include_dataset_one_hot": include_dataset_one_hot,
        "mean_oracle_reward": (
            sum(row["oracle_best_reward"] for row in per_context) / len(per_context) if per_context else 0.0
        ),
        "mean_learned_reward": (
            sum(row["learned_reward"] for row in per_context) / len(per_context) if per_context else 0.0
        ),
        "mean_learned_regret": (
            sum(row["learned_regret"] for row in per_context) / len(per_context) if per_context else 0.0
        ),
        "learned_budget_accuracy": learned_budget_accuracy,
        "fixed_budget_baseline_summary": fixed_baseline_summary,
        "round19_replay_enabled": round19_policy is not None,
        "datasetwise": datasetwise,
        "per_context": per_context,
    }
    if round19_policy is not None and per_context:
        round19_rewards = [float(row["round19_reward"]) for row in per_context]
        report["mean_round19_reward"] = sum(round19_rewards) / len(round19_rewards)
        report["round19_budget_accuracy"] = round19_budget_hits / len(per_context)
        report["learned_win_rate_vs_round19"] = (
            learned_better_than_round19 / len(per_context)
        )
        report["decision_consistency_vs_round19"] = {
            "same_budget_choice": sum(
                1 for row in per_context if int(row["learned_budget"]) == int(row["round19_budget"])
            ),
            "learned_better_than_round19": learned_better_than_round19,
            "round19_better_than_learned": round19_better_than_learned,
            "equal_reward": equal_to_round19,
        }

    output_root = Path(report_dir)
    dump_json(output_root / "round22_bandit_eval_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate round22 learned budget policy.")
    parser.add_argument(
        "--action-samples",
        default=str(DEFAULT_DATASET_DIR / "round22_action_samples.jsonl"),
    )
    parser.add_argument(
        "--context-table",
        default=str(DEFAULT_DATASET_DIR / "round22_context_table.jsonl"),
    )
    parser.add_argument(
        "--final-test",
        default=str(DEFAULT_SPLIT_DIR / "round22_final_test_contexts.json"),
    )
    parser.add_argument(
        "--config",
        default=str(MODEL_TRAIN_ROOT / "configs" / "train_round22_bandit.yaml"),
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--round19-replay-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        action_samples_path=args.action_samples,
        context_table_path=args.context_table,
        final_test_path=args.final_test,
        config_path=args.config,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        round19_replay_path=args.round19_replay_path,
    )
    print(
        f"EVAL contexts={report['context_count']} learned_reward={report['mean_learned_reward']:.6f} "
        f"regret={report['mean_learned_regret']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
