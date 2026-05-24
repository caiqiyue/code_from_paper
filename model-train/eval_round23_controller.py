from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import DEFAULT_ROUND23_DATASET_DIR, DEFAULT_ROUND23_MODEL_DIR, DEFAULT_ROUND23_REPORT_DIR, DEFAULT_ROUND23_SPLIT_DIR, DELTA_ACTIONS, REFERENCE_BUDGET, dump_json, load_yaml, read_jsonl
from features import encode_feature_row, encode_feature_row_with_fields
from round19_replay import ExplicitRound19ReplayPolicy, load_explicit_round19_replay, load_round19_replay_table
from round23_controller_models import load_regressor, predict_regressor, require_model_family
from round23_feature_sets import resolve_feature_spec


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "train_round23_controller.yaml"


def _load_context_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_context_payload(path: str | Path, key: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(value) for value in payload[key]}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _direction(delta_k: int) -> int:
    if delta_k < 0:
        return -1
    if delta_k > 0:
        return 1
    return 0


def _controller_reward_field(delta_k: int) -> str:
    if delta_k < 0:
        return f"controller_reward_dk_neg{abs(delta_k)}"
    if delta_k > 0:
        return f"controller_reward_dk_pos{delta_k}"
    return "controller_reward_dk_0"


def _delta_suffix(delta_k: int) -> str:
    if delta_k < 0:
        return f"neg{abs(delta_k)}"
    if delta_k > 0:
        return f"pos{delta_k}"
    return "0"


def _context_reward(row: dict[str, Any], delta_k: int) -> float:
    return float(row[_controller_reward_field(delta_k)])


def _context_best_top1(row: dict[str, Any], delta_k: int) -> float | None:
    field = f"best_top1_dk_{_delta_suffix(delta_k)}"
    if field not in row or row[field] in (None, ""):
        return None
    return float(row[field])


def _load_round19_policy(round19_replay_path: str | Path | None) -> ExplicitRound19ReplayPolicy | None:
    if round19_replay_path is None:
        return None
    resolved = Path(round19_replay_path)
    if resolved.suffix.lower() == ".jsonl":
        return load_round19_replay_table(resolved)
    return load_explicit_round19_replay(resolved)


def evaluate_controller(
    *,
    controller_context_table_path: str | Path,
    context_split_path: str | Path,
    context_split_key: str,
    config_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    model_family: str,
    feature_version: str | None = None,
    round19_replay_path: str | Path | None = None,
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
    delta_actions = [int(value) for value in model_cfg.get("delta_actions", DELTA_ACTIONS)]

    context_rows = _load_context_rows(controller_context_table_path)
    split_context_ids = _load_context_payload(context_split_path, context_split_key)
    test_rows = [row for row in context_rows if str(row["context_id"]) in split_context_ids]

    models: dict[int, Any] = {}
    for delta_k in delta_actions:
        action_name = f"neg{abs(delta_k)}" if delta_k < 0 else ("0" if delta_k == 0 else f"pos{delta_k}")
        extension = {
            "lightgbm": ".txt",
            "xgboost": ".json",
            "catboost": ".cbm",
        }.get(normalized_family, ".pkl")
        models[int(delta_k)] = load_regressor(
            family=normalized_family,
            path=Path(model_dir) / f"model_dk_{action_name}{extension}",
        )

    round19_policy = _load_round19_policy(round19_replay_path)

    per_context: list[dict[str, Any]] = []
    for row in test_rows:
        _, feature_values = encode_feature_row_with_fields(
            row,
            state_fields=feature_spec.feature_fields,
            include_dataset_one_hot=feature_spec.include_dataset_onehot,
            dataset_order=feature_spec.onehot_order,
        )
        predicted_rewards = {
            int(delta_k): float(
                predict_regressor(
                    family=normalized_family,
                    model=models[int(delta_k)],
                    feature_matrix=[feature_values],
                )[0]
            )
            for delta_k in delta_actions
        }
        predicted_delta = max(predicted_rewards, key=predicted_rewards.get)
        oracle_delta = int(row["oracle_best_delta_k"])
        learned_reward = _context_reward(row, predicted_delta)
        oracle_reward = float(row["oracle_best_controller_reward"])
        predicted_best_top1 = _context_best_top1(row, predicted_delta)
        oracle_best_top1 = float(row["oracle_best_top1"]) if "oracle_best_top1" in row and row["oracle_best_top1"] not in (None, "") else None
        keep_k0_best_top1 = _context_best_top1(row, 0)

        round19_budget = round19_policy.predict(row) if round19_policy is not None else None
        round19_delta = int(round19_budget) - REFERENCE_BUDGET if round19_budget is not None else None
        round19_reward = _context_reward(row, round19_delta) if round19_delta is not None else None

        per_context.append(
            {
                "context_id": row["context_id"],
                "dataset_name": row["dataset_name"],
                "label_target_mode": row.get("label_target_mode", ""),
                "tie_margin": row.get("tie_margin"),
                "oracle_delta_k": oracle_delta,
                "oracle_reward": oracle_reward,
                "predicted_delta_k": int(predicted_delta),
                "predicted_target_budget": REFERENCE_BUDGET + int(predicted_delta),
                "predicted_reward": learned_reward,
                "predicted_regret": oracle_reward - learned_reward,
                "keep_k0_reward": _context_reward(row, 0),
                "predicted_best_top1": predicted_best_top1,
                "oracle_best_top1": oracle_best_top1,
                "keep_k0_best_top1": keep_k0_best_top1,
                "best_top1_regret": (
                    None if predicted_best_top1 is None or oracle_best_top1 is None else oracle_best_top1 - predicted_best_top1
                ),
                "win_vs_keep_k0_by_best_top1": (
                    None if predicted_best_top1 is None or keep_k0_best_top1 is None else float(predicted_best_top1 > keep_k0_best_top1)
                ),
                "round19_budget": round19_budget,
                "round19_delta_k": round19_delta,
                "round19_reward": round19_reward,
                "predicted_rewards": predicted_rewards,
            }
        )

    datasetwise: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({str(row["dataset_name"]) for row in per_context}):
        dataset_rows = [row for row in per_context if row["dataset_name"] == dataset_name]
        payload = {
            "context_count": len(dataset_rows),
            "mean_predicted_reward": _mean([float(row["predicted_reward"]) for row in dataset_rows]),
            "mean_predicted_regret": _mean([float(row["predicted_regret"]) for row in dataset_rows]),
            "mean_keep_k0_reward": _mean([float(row["keep_k0_reward"]) for row in dataset_rows]),
            "mean_best_top1_regret": _mean(
                [float(row["best_top1_regret"]) for row in dataset_rows if row["best_top1_regret"] is not None]
            ),
            "win_rate_vs_keep_k0_by_best_top1": _mean(
                [
                    float(row["win_vs_keep_k0_by_best_top1"])
                    for row in dataset_rows
                    if row["win_vs_keep_k0_by_best_top1"] is not None
                ]
            ),
            "delta_k_accuracy": _mean(
                [1.0 if int(row["predicted_delta_k"]) == int(row["oracle_delta_k"]) else 0.0 for row in dataset_rows]
            ),
            "direction_accuracy": _mean(
                [1.0 if _direction(int(row["predicted_delta_k"])) == _direction(int(row["oracle_delta_k"])) else 0.0 for row in dataset_rows]
            ),
        }
        if round19_policy is not None:
            payload["mean_round19_reward"] = _mean([float(row["round19_reward"]) for row in dataset_rows])
        datasetwise[dataset_name] = payload

    report = {
        "context_count": len(per_context),
        "controller_context_table_path": str(Path(controller_context_table_path).resolve()),
        "context_split_path": str(Path(context_split_path).resolve()),
        "context_split_key": context_split_key,
        "config_path": str(Path(config_path).resolve()),
        "model_dir": str(Path(model_dir).resolve()),
        "model_family": normalized_family,
        "feature_version": feature_spec.feature_version,
        "include_dataset_one_hot": feature_spec.include_dataset_onehot,
        "onehot_order": feature_spec.onehot_order,
        "delta_actions": delta_actions,
        "mean_predicted_reward": _mean([float(row["predicted_reward"]) for row in per_context]),
        "mean_predicted_regret": _mean([float(row["predicted_regret"]) for row in per_context]),
        "mean_keep_k0_reward": _mean([float(row["keep_k0_reward"]) for row in per_context]),
        "mean_predicted_training_value": _mean([float(row["predicted_reward"]) for row in per_context]),
        "mean_predicted_training_regret": _mean([float(row["predicted_regret"]) for row in per_context]),
        "mean_best_top1_regret": _mean(
            [float(row["best_top1_regret"]) for row in per_context if row["best_top1_regret"] is not None]
        ),
        "win_rate_vs_keep_k0_by_best_top1": _mean(
            [
                float(row["win_vs_keep_k0_by_best_top1"])
                for row in per_context
                if row["win_vs_keep_k0_by_best_top1"] is not None
            ]
        ),
        "target_field": target_field,
        "target_mode": str(per_context[0].get("label_target_mode", "")) if per_context else "",
        "delta_k_accuracy": _mean(
            [1.0 if int(row["predicted_delta_k"]) == int(row["oracle_delta_k"]) else 0.0 for row in per_context]
        ),
        "direction_accuracy": _mean(
            [1.0 if _direction(int(row["predicted_delta_k"])) == _direction(int(row["oracle_delta_k"])) else 0.0 for row in per_context]
        ),
        "round19_replay_enabled": round19_policy is not None,
        "datasetwise": datasetwise,
        "per_context": per_context,
    }
    if round19_policy is not None:
        report["mean_round19_reward"] = _mean([float(row["round19_reward"]) for row in per_context])
        report["win_rate_vs_round19"] = _mean(
            [1.0 if float(row["predicted_reward"]) > float(row["round19_reward"]) else 0.0 for row in per_context]
        )

    output_root = Path(report_dir)
    dump_json(output_root / "round23_controller_eval_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate round23 controller.")
    parser.add_argument("--controller-context-table", default=str(DEFAULT_ROUND23_DATASET_DIR / "round23_controller_context_table.jsonl"))
    parser.add_argument("--final-test", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_final_test_contexts.json"))
    parser.add_argument("--unseen-test", default=None)
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--model-dir", default=str(DEFAULT_ROUND23_MODEL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_ROUND23_REPORT_DIR))
    parser.add_argument("--model-family", default="lightgbm")
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--round19-replay-path", default=None)
    parser.add_argument("--target-field", default="reward_round23_controller")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context_split_path = args.unseen_test or args.final_test
    context_split_key = "unseen_test_context_ids" if args.unseen_test else "final_test_context_ids"
    report = evaluate_controller(
        controller_context_table_path=args.controller_context_table,
        context_split_path=context_split_path,
        context_split_key=context_split_key,
        config_path=args.config,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        model_family=args.model_family,
        feature_version=args.feature_version,
        round19_replay_path=args.round19_replay_path,
        target_field=args.target_field,
    )
    print(
        f"EVAL round23 family={report['model_family']} feature_version={report['feature_version']} "
        f"contexts={report['context_count']} reward={report['mean_predicted_reward']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
