from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    DELTA_ACTIONS,
    REFERENCE_BUDGET,
    as_float,
    dump_json,
    ensure_dir,
    load_yaml,
    read_jsonl,
    write_csv,
    write_jsonl,
)
from features import encode_feature_row_with_fields
from round23_controller_models import load_regressor, model_file_extension, predict_regressor, require_model_family
from round23_feature_sets import resolve_feature_spec


POLICY_KEEP = "keep-k0=20"
POLICY_ROUND19 = "round19 resolver replay"
POLICY_ROUND23 = "round23 controller"
POLICY_ORACLE = "oracle budget"
POLICIES = [POLICY_KEEP, POLICY_ROUND19, POLICY_ROUND23, POLICY_ORACLE]
ACTION_SUFFIXES = {-2: "neg2", -1: "neg1", 0: "0", 1: "pos1", 2: "pos2"}
SEEN4_DATASETS = ["jobs", "congressional", "forums", "microblog"]
ADDED2_DATASETS = ["imdb", "openreview"]
ALL6_DATASETS = SEEN4_DATASETS + ADDED2_DATASETS


class E3InputError(RuntimeError):
    pass


def reward_field_for_delta(delta_k: int) -> str:
    return f"controller_reward_dk_{_delta_suffix(delta_k)}"


def best_top1_field_for_delta(delta_k: int) -> str:
    return f"best_top1_dk_{_delta_suffix(delta_k)}"


def direction(delta_k: int) -> int:
    if int(delta_k) < 0:
        return -1
    if int(delta_k) > 0:
        return 1
    return 0


def _delta_suffix(delta_k: int) -> str:
    normalized = int(delta_k)
    if normalized not in ACTION_SUFFIXES:
        raise E3InputError(f"Invalid delta_k={delta_k}; expected one of {DELTA_ACTIONS}")
    return ACTION_SUFFIXES[normalized]


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _round_or_none(value: float | None, digits: int = 10) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _index_unique(rows: list[dict[str, Any]], *, label: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    seen: set[str] = set()
    duplicates: list[str] = []
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        context_id = str(row["context_id"])
        if context_id in seen:
            duplicates.append(context_id)
        seen.add(context_id)
        indexed[context_id] = row
    return indexed, sorted(set(duplicates))


def _almost_equal(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _require_action_fields(row: dict[str, Any], *, context_id: str) -> list[str]:
    missing: list[str] = []
    for delta_k in DELTA_ACTIONS:
        for field in (reward_field_for_delta(delta_k), best_top1_field_for_delta(delta_k)):
            if field not in row or row[field] in (None, ""):
                missing.append(f"{context_id}:{field}")
    return missing


def audit_inputs(
    context_rows: list[dict[str, Any]],
    round19_rows: list[dict[str, Any]],
    round23_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    context_by_id, duplicate_context_ids = _index_unique(context_rows, label="context")
    round19_by_id, duplicate_round19_context_ids = _index_unique(round19_rows, label="round19")

    context_ids = set(context_by_id)
    round19_ids = set(round19_by_id)
    round23_ids = {str(key) for key in round23_predictions}

    dataset_counts = dict(sorted(Counter(str(row.get("dataset_name", "")) for row in context_rows).items()))
    missing_reward_fields: list[str] = []
    keep_k0_reward_mismatches: list[str] = []
    oracle_reward_mismatches: list[str] = []
    invalid_oracle_delta: dict[str, Any] = {}
    invalid_round19_delta: dict[str, Any] = {}
    invalid_round23_delta: dict[str, Any] = {}
    invalid_round19_budget: dict[str, Any] = {}
    invalid_round23_budget: dict[str, Any] = {}

    for row in context_rows:
        context_id = str(row["context_id"])
        missing_reward_fields.extend(_require_action_fields(row, context_id=context_id))
        try:
            oracle_delta = int(row["oracle_best_delta_k"])
            _delta_suffix(oracle_delta)
        except Exception:
            invalid_oracle_delta[context_id] = row.get("oracle_best_delta_k")
            continue
        if "keep_k0_reward" in row and not _almost_equal(row["keep_k0_reward"], row[reward_field_for_delta(0)]):
            keep_k0_reward_mismatches.append(context_id)
        if "oracle_best_controller_reward" in row and not _almost_equal(
            row["oracle_best_controller_reward"],
            row[reward_field_for_delta(oracle_delta)],
        ):
            oracle_reward_mismatches.append(context_id)

    for context_id in sorted(context_ids & round19_ids):
        row = round19_by_id[context_id]
        try:
            delta_k = int(row["round19_predicted_delta_k"])
            _delta_suffix(delta_k)
        except Exception:
            invalid_round19_delta[context_id] = row.get("round19_predicted_delta_k")
            continue
        expected_budget = REFERENCE_BUDGET + delta_k
        if "round19_predicted_budget" in row and int(row["round19_predicted_budget"]) != expected_budget:
            invalid_round19_budget[context_id] = row.get("round19_predicted_budget")

    for context_id in sorted(context_ids & round23_ids):
        prediction = round23_predictions[context_id]
        try:
            delta_k = int(prediction["predicted_delta_k"])
            _delta_suffix(delta_k)
        except Exception:
            invalid_round23_delta[context_id] = prediction.get("predicted_delta_k")
            continue
        expected_budget = REFERENCE_BUDGET + delta_k
        predicted_budget = prediction.get("predicted_target_budget")
        if predicted_budget not in (None, "") and int(predicted_budget) != expected_budget:
            invalid_round23_budget[context_id] = predicted_budget

    audit = {
        "status": "pass",
        "input_context_count": len(context_rows),
        "unique_context_count": len(context_by_id),
        "dataset_counts": dataset_counts,
        "round19_context_count": len(round19_rows),
        "round19_unique_context_count": len(round19_by_id),
        "round23_prediction_count": len(round23_predictions),
        "missing_round19_contexts": sorted(context_ids - round19_ids),
        "extra_round19_contexts": sorted(round19_ids - context_ids),
        "missing_round23_predictions": sorted(context_ids - round23_ids),
        "extra_round23_predictions": sorted(round23_ids - context_ids),
        "duplicate_context_ids": duplicate_context_ids,
        "duplicate_round19_context_ids": duplicate_round19_context_ids,
        "invalid_oracle_delta": invalid_oracle_delta,
        "invalid_round19_delta": invalid_round19_delta,
        "invalid_round23_delta": invalid_round23_delta,
        "invalid_round19_budget": invalid_round19_budget,
        "invalid_round23_budget": invalid_round23_budget,
        "missing_reward_fields": missing_reward_fields,
        "keep_k0_reward_mismatches": keep_k0_reward_mismatches,
        "oracle_reward_mismatches": oracle_reward_mismatches,
    }

    failures = [
        key
        for key in (
            "missing_round19_contexts",
            "extra_round19_contexts",
            "missing_round23_predictions",
            "extra_round23_predictions",
            "duplicate_context_ids",
            "duplicate_round19_context_ids",
            "invalid_oracle_delta",
            "invalid_round19_delta",
            "invalid_round23_delta",
            "invalid_round19_budget",
            "invalid_round23_budget",
            "missing_reward_fields",
            "keep_k0_reward_mismatches",
            "oracle_reward_mismatches",
        )
        if audit[key]
    ]
    if failures:
        audit["status"] = "fail"
        raise E3InputError(f"E3 input audit failed: {', '.join(failures)}")
    return audit


def _selected_reward(row: dict[str, Any], delta_k: int) -> float:
    return float(row[reward_field_for_delta(delta_k)])


def _selected_best_top1(row: dict[str, Any], delta_k: int) -> float:
    return float(row[best_top1_field_for_delta(delta_k)])


def _normalize_predicted_rewards(prediction: dict[str, Any]) -> dict[str, float]:
    raw = prediction.get("predicted_rewards", {})
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        normalized[str(int(key))] = float(value)
    return normalized


def build_policy_context_rows(
    *,
    context_rows: list[dict[str, Any]],
    round19_rows: list[dict[str, Any]],
    round23_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_inputs(context_rows, round19_rows, round23_predictions)
    round19_by_id, _ = _index_unique(round19_rows, label="round19")
    rows: list[dict[str, Any]] = []

    for context_row in context_rows:
        context_id = str(context_row["context_id"])
        round19_row = round19_by_id[context_id]
        round23_prediction = round23_predictions[context_id]

        keep_delta = 0
        round19_delta = int(round19_row["round19_predicted_delta_k"])
        round23_delta = int(round23_prediction["predicted_delta_k"])
        oracle_delta = int(context_row["oracle_best_delta_k"])
        oracle_reward = _selected_reward(context_row, oracle_delta)
        oracle_top1 = _selected_best_top1(context_row, oracle_delta)
        keep_reward = _selected_reward(context_row, keep_delta)
        round19_reward = _selected_reward(context_row, round19_delta)
        round23_reward = _selected_reward(context_row, round23_delta)
        keep_top1 = _selected_best_top1(context_row, keep_delta)
        round19_top1 = _selected_best_top1(context_row, round19_delta)
        round23_top1 = _selected_best_top1(context_row, round23_delta)

        rows.append(
            {
                "context_id": context_id,
                "dataset_name": context_row["dataset_name"],
                "meta_seed": context_row.get("meta_seed"),
                "label_target_mode": context_row.get("label_target_mode"),
                "tie_margin": context_row.get("tie_margin"),
                "reference_budget": int(context_row.get("reference_budget", REFERENCE_BUDGET)),
                "oracle_delta_k": oracle_delta,
                "oracle_target_budget": REFERENCE_BUDGET + oracle_delta,
                "oracle_reward": oracle_reward,
                "oracle_best_top1": oracle_top1,
                "keep_delta_k": keep_delta,
                "keep_target_budget": REFERENCE_BUDGET,
                "keep_reward": keep_reward,
                "keep_best_top1": keep_top1,
                "round19_delta_k": round19_delta,
                "round19_target_budget": REFERENCE_BUDGET + round19_delta,
                "round19_reward": round19_reward,
                "round19_best_top1": round19_top1,
                "round19_legacy_replay_reward": (
                    None
                    if round19_row.get("round19_replay_reward") in (None, "")
                    else float(round19_row["round19_replay_reward"])
                ),
                "round19_legacy_replay_best_top1": (
                    None
                    if round19_row.get("round19_replay_best_top1") in (None, "")
                    else float(round19_row["round19_replay_best_top1"])
                ),
                "round23_delta_k": round23_delta,
                "round23_target_budget": REFERENCE_BUDGET + round23_delta,
                "round23_reward": round23_reward,
                "round23_best_top1": round23_top1,
                "round23_predicted_rewards": _normalize_predicted_rewards(round23_prediction),
                "keep_regret_vs_oracle": oracle_reward - keep_reward,
                "round19_regret_vs_oracle": oracle_reward - round19_reward,
                "round23_regret_vs_oracle": oracle_reward - round23_reward,
                "keep_best_top1_regret": oracle_top1 - keep_top1,
                "round19_best_top1_regret": oracle_top1 - round19_top1,
                "round23_best_top1_regret": oracle_top1 - round23_top1,
                "keep_win_vs_keep_by_reward": 0.0,
                "round19_win_vs_keep_by_reward": float(round19_reward > keep_reward),
                "round23_win_vs_keep_by_reward": float(round23_reward > keep_reward),
                "oracle_win_vs_keep_by_reward": float(oracle_reward > keep_reward),
                "keep_win_vs_round19_by_reward": float(keep_reward > round19_reward),
                "round19_win_vs_round19_by_reward": 0.0,
                "round23_win_vs_round19_by_reward": float(round23_reward > round19_reward),
                "oracle_win_vs_round19_by_reward": float(oracle_reward > round19_reward),
                "keep_delta_k_correct": float(keep_delta == oracle_delta),
                "round19_delta_k_correct": float(round19_delta == oracle_delta),
                "round23_delta_k_correct": float(round23_delta == oracle_delta),
                "oracle_delta_k_correct": 1.0,
                "keep_direction_correct": float(direction(keep_delta) == direction(oracle_delta)),
                "round19_direction_correct": float(direction(round19_delta) == direction(oracle_delta)),
                "round23_direction_correct": float(direction(round23_delta) == direction(oracle_delta)),
                "oracle_direction_correct": 1.0,
            }
        )
    return rows


def _policy_values(row: dict[str, Any], policy: str) -> dict[str, Any]:
    prefix_by_policy = {
        POLICY_KEEP: "keep",
        POLICY_ROUND19: "round19",
        POLICY_ROUND23: "round23",
        POLICY_ORACLE: "oracle",
    }
    prefix = prefix_by_policy[policy]
    if policy == POLICY_ORACLE:
        return {
            "delta_k": row["oracle_delta_k"],
            "reward": row["oracle_reward"],
            "regret": 0.0,
            "best_top1_regret": 0.0,
            "win_vs_keep": row["oracle_win_vs_keep_by_reward"],
            "win_vs_round19": row["oracle_win_vs_round19_by_reward"],
            "direction_correct": 1.0,
            "delta_correct": 1.0,
        }
    return {
        "delta_k": row[f"{prefix}_delta_k"],
        "reward": row[f"{prefix}_reward"],
        "regret": row[f"{prefix}_regret_vs_oracle"],
        "best_top1_regret": row[f"{prefix}_best_top1_regret"],
        "win_vs_keep": None if policy == POLICY_KEEP else row[f"{prefix}_win_vs_keep_by_reward"],
        "win_vs_round19": None if policy == POLICY_ROUND19 else row[f"{prefix}_win_vs_round19_by_reward"],
        "direction_correct": row[f"{prefix}_direction_correct"],
        "delta_correct": row[f"{prefix}_delta_k_correct"],
    }


def summarize_overall(policy_context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for policy in POLICIES:
        values = [_policy_values(row, policy) for row in policy_context_rows]
        summary.append(
            {
                "policy": policy,
                "contexts": len(values),
                "mean_reward": _round_or_none(_mean([value["reward"] for value in values])),
                "mean_regret_vs_oracle": _round_or_none(_mean([value["regret"] for value in values])),
                "win_rate_vs_keep_k0_by_reward": _round_or_none(_mean([value["win_vs_keep"] for value in values])),
                "win_rate_vs_round19_by_reward": _round_or_none(_mean([value["win_vs_round19"] for value in values])),
                "direction_accuracy": _round_or_none(_mean([value["direction_correct"] for value in values])),
                "delta_k_accuracy": _round_or_none(_mean([value["delta_correct"] for value in values])),
                "mean_best_top1_regret": _round_or_none(_mean([value["best_top1_regret"] for value in values])),
            }
        )
    return summary


def summarize_datasetwise(policy_context_rows: list[dict[str, Any]], *, include_all_policies: bool = True) -> list[dict[str, Any]]:
    policies = POLICIES if include_all_policies else [POLICY_ROUND19, POLICY_ROUND23]
    summary: list[dict[str, Any]] = []
    for dataset_name in sorted({str(row["dataset_name"]) for row in policy_context_rows}):
        dataset_rows = [row for row in policy_context_rows if str(row["dataset_name"]) == dataset_name]
        for policy in policies:
            values = [_policy_values(row, policy) for row in dataset_rows]
            summary.append(
                {
                    "dataset_name": dataset_name,
                    "policy": policy,
                    "contexts": len(values),
                    "mean_reward": _round_or_none(_mean([value["reward"] for value in values])),
                    "mean_regret_vs_oracle": _round_or_none(_mean([value["regret"] for value in values])),
                    "win_rate_vs_keep_k0_by_reward": _round_or_none(_mean([value["win_vs_keep"] for value in values])),
                    "direction_accuracy": _round_or_none(_mean([value["direction_correct"] for value in values])),
                    "delta_k_accuracy": _round_or_none(_mean([value["delta_correct"] for value in values])),
                    "mean_best_top1_regret": _round_or_none(_mean([value["best_top1_regret"] for value in values])),
                }
            )
    return summary


def summarize_action_distribution(policy_context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        deltas = [int(_policy_values(row, policy)["delta_k"]) for row in policy_context_rows]
        counts = Counter(deltas)
        payload: dict[str, Any] = {"policy": policy, "contexts": len(deltas)}
        for delta_k in DELTA_ACTIONS:
            suffix = _delta_suffix(delta_k)
            count = int(counts.get(delta_k, 0))
            payload[f"count_delta_{suffix}"] = count
            payload[f"share_delta_{suffix}"] = _round_or_none(count / len(deltas) if deltas else None)
        rows.append(payload)
    return rows


def summarize_split_level(policy_context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "seen4": set(SEEN4_DATASETS),
        "added2": set(ADDED2_DATASETS),
        "all6": set(ALL6_DATASETS),
    }
    rows: list[dict[str, Any]] = []
    for split_name, datasets in groups.items():
        split_rows = [row for row in policy_context_rows if str(row["dataset_name"]) in datasets]
        for policy in [POLICY_ROUND19, POLICY_ROUND23]:
            values = [_policy_values(row, policy) for row in split_rows]
            rows.append(
                {
                    "split": split_name,
                    "policy": policy,
                    "contexts": len(values),
                    "mean_reward": _round_or_none(_mean([value["reward"] for value in values])),
                    "mean_regret_vs_oracle": _round_or_none(_mean([value["regret"] for value in values])),
                    "win_rate_vs_keep_k0_by_reward": _round_or_none(_mean([value["win_vs_keep"] for value in values])),
                    "win_rate_vs_round19_by_reward": _round_or_none(_mean([value["win_vs_round19"] for value in values])),
                    "direction_accuracy": _round_or_none(_mean([value["direction_correct"] for value in values])),
                    "delta_k_accuracy": _round_or_none(_mean([value["delta_correct"] for value in values])),
                }
            )
    return rows


def summarize_direction_baseline(policy_context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oracle_directions = [direction(int(row["oracle_delta_k"])) for row in policy_context_rows]
    counts = Counter(oracle_directions)
    majority_direction = counts.most_common(1)[0][0] if counts else 0
    return [
        {
            "method": "majority oracle direction baseline",
            "direction": majority_direction,
            "contexts": len(oracle_directions),
            "direction_accuracy": _round_or_none(_mean([float(value == majority_direction) for value in oracle_directions])),
        },
        {
            "method": "always keep direction baseline",
            "direction": 0,
            "contexts": len(oracle_directions),
            "direction_accuracy": _round_or_none(_mean([float(value == 0) for value in oracle_directions])),
        },
        {
            "method": POLICY_ROUND19,
            "direction": None,
            "contexts": len(policy_context_rows),
            "direction_accuracy": _round_or_none(_mean([row["round19_direction_correct"] for row in policy_context_rows])),
        },
        {
            "method": POLICY_ROUND23,
            "direction": None,
            "contexts": len(policy_context_rows),
            "direction_accuracy": _round_or_none(_mean([row["round23_direction_correct"] for row in policy_context_rows])),
        },
    ]


def load_round23_predictions_from_eval_report(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    predictions: dict[str, dict[str, Any]] = {}
    for row in payload.get("per_context", []):
        predictions[str(row["context_id"])] = {
            "predicted_delta_k": int(row["predicted_delta_k"]),
            "predicted_target_budget": row.get("predicted_target_budget"),
            "predicted_rewards": row.get("predicted_rewards", {}),
        }
    return predictions


def infer_round23_predictions_from_model(
    *,
    context_rows: list[dict[str, Any]],
    model_dir: str | Path,
    model_family: str,
    feature_version: str | None,
    config_path: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    normalized_family = require_model_family(model_family)
    model_cfg: dict[str, Any] = {}
    if config_path is not None:
        model_cfg = dict(load_yaml(config_path).get("model", {}))
    feature_spec = resolve_feature_spec(
        feature_version=feature_version or model_cfg.get("feature_version"),
        feature_fields=list(model_cfg.get("feature_fields", [])) or None,
        include_dataset_onehot=model_cfg.get("include_dataset_one_hot"),
        onehot_order=list(model_cfg.get("onehot_order", [])) or None,
    )
    delta_actions = [int(value) for value in model_cfg.get("delta_actions", DELTA_ACTIONS)]
    models: dict[int, Any] = {}
    extension = model_file_extension(normalized_family)
    for delta_k in delta_actions:
        action_name = _delta_suffix(delta_k)
        models[delta_k] = load_regressor(
            family=normalized_family,
            path=Path(model_dir) / f"model_dk_{action_name}{extension}",
        )

    predictions: dict[str, dict[str, Any]] = {}
    for row in context_rows:
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
        predictions[str(row["context_id"])] = {
            "predicted_delta_k": int(predicted_delta),
            "predicted_target_budget": REFERENCE_BUDGET + int(predicted_delta),
            "predicted_rewards": predicted_rewards,
        }
    provenance = {
        "prediction_source": "model_dir",
        "model_dir": str(Path(model_dir).resolve()),
        "model_family": normalized_family,
        "feature_version": feature_spec.feature_version,
        "feature_fields": feature_spec.feature_fields,
        "include_dataset_one_hot": feature_spec.include_dataset_onehot,
        "onehot_order": feature_spec.onehot_order,
        "delta_actions": delta_actions,
        "config_path": None if config_path is None else str(Path(config_path).resolve()),
    }
    return predictions, provenance


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(title: str, rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_value(row.get(field)) for field in fields) + " |")
    lines.append("")
    return "\n".join(lines)


def write_markdown_summary(
    *,
    output_path: str | Path,
    provenance: dict[str, Any],
    overall: list[dict[str, Any]],
    datasetwise: list[dict[str, Any]],
    action_distribution: list[dict[str, Any]],
    split_level: list[dict[str, Any]],
    direction_baseline: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    lines = [
        "# E3 Controller Policy Quality Summary",
        "",
        "Oracle budget is an offline upper bound, not a deployable policy.",
        "",
        "## Provenance",
        "",
        f"- controller_context_table: {provenance.get('controller_context_table')}",
        f"- round19_replay_table: {provenance.get('round19_replay_table')}",
        f"- prediction_source: {provenance.get('prediction_source')}",
        f"- scope: {provenance.get('scope')}",
        f"- audit_status: {audit.get('status')}",
        "",
    ]
    lines.append(
        _markdown_table(
            "Table E3-1 Overall Controller Policy Quality",
            overall,
            [
                "policy",
                "contexts",
                "mean_reward",
                "mean_regret_vs_oracle",
                "win_rate_vs_keep_k0_by_reward",
                "win_rate_vs_round19_by_reward",
                "direction_accuracy",
                "delta_k_accuracy",
            ],
        )
    )
    lines.append(
        _markdown_table(
            "Table E3-2 Dataset-wise Controller Quality",
            datasetwise,
            [
                "dataset_name",
                "policy",
                "contexts",
                "mean_reward",
                "mean_regret_vs_oracle",
                "win_rate_vs_keep_k0_by_reward",
                "direction_accuracy",
            ],
        )
    )
    lines.append(_markdown_table("Table E3-3 Action Distribution", action_distribution, _fieldnames(action_distribution)))
    lines.append(_markdown_table("Table E3-4 Split-level Controller Quality", split_level, _fieldnames(split_level)))
    lines.append(_markdown_table("Table E3-5 Direction Baseline Check", direction_baseline, _fieldnames(direction_baseline)))
    lines.append("## Audit Report")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit, ensure_ascii=False, indent=2))
    lines.append("```")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_audit_markdown(*, output_path: str | Path, audit: dict[str, Any], provenance: dict[str, Any]) -> None:
    lines = [
        "# E3 Audit Report",
        "",
        f"- status: {audit.get('status')}",
        f"- scope: {audit.get('scope')}",
        f"- controller_context_table: {provenance.get('controller_context_table')}",
        f"- round19_replay_table: {provenance.get('round19_replay_table')}",
        f"- prediction_source: {provenance.get('prediction_source')}",
        f"- input_context_count: {audit.get('input_context_count')}",
        f"- unique_context_count: {audit.get('unique_context_count')}",
        f"- round19_context_count: {audit.get('round19_context_count')}",
        f"- round23_prediction_count: {audit.get('round23_prediction_count')}",
        f"- policy_row_count: {audit.get('policy_row_count')}",
        "",
        "## Dataset Counts",
        "",
        "| dataset_name | contexts |",
        "| --- | --- |",
    ]
    dataset_counts = audit.get("dataset_counts", {})
    if isinstance(dataset_counts, dict):
        for dataset_name, count in dataset_counts.items():
            lines.append(f"| {dataset_name} | {count} |")
    lines.extend(["", "## Full Audit Payload", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```"])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_e3_outputs(
    *,
    output_dir: str | Path,
    policy_context_rows: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    output_root = ensure_dir(output_dir)
    write_jsonl(output_root / "e3_policy_contexts.jsonl", policy_context_rows)
    dump_json(output_root / "e3_policy_contexts.json", policy_context_rows)
    dump_json(output_root / "e3_audit_report.json", audit)
    dump_json(output_root / "e3_tables.json", tables)

    for table_name, rows in tables.items():
        if rows:
            write_csv(output_root / f"{table_name}.csv", rows, _fieldnames(rows))
            dump_json(output_root / f"{table_name}.json", rows)

    write_markdown_summary(
        output_path=output_root / "e3_summary.md",
        provenance=provenance,
        overall=tables["e3_table_overall_policy_quality"],
        datasetwise=tables["e3_table_datasetwise_policy_quality"],
        action_distribution=tables["e3_table_action_distribution"],
        split_level=tables["e3_table_split_level_policy_quality"],
        direction_baseline=tables["e3_table_direction_baseline"],
        audit=audit,
    )
    write_audit_markdown(
        output_path=output_root / "e3_audit_report.md",
        audit=audit,
        provenance=provenance,
    )


def evaluate_e3_policy_quality(
    *,
    controller_context_table: str | Path,
    round19_replay_table: str | Path,
    output_dir: str | Path,
    round23_eval_report: str | Path | None = None,
    model_dir: str | Path | None = None,
    model_family: str = "extratrees",
    feature_version: str | None = "no_dataset",
    config_path: str | Path | None = None,
    scope: str = "all_contexts",
) -> dict[str, Any]:
    context_rows = read_jsonl(controller_context_table)
    round19_rows = read_jsonl(round19_replay_table)

    if round23_eval_report is not None:
        predictions = load_round23_predictions_from_eval_report(round23_eval_report)
        prediction_provenance = {
            "prediction_source": "round23_eval_report",
            "round23_eval_report": str(Path(round23_eval_report).resolve()),
        }
    elif model_dir is not None:
        predictions, prediction_provenance = infer_round23_predictions_from_model(
            context_rows=context_rows,
            model_dir=model_dir,
            model_family=model_family,
            feature_version=feature_version,
            config_path=config_path,
        )
    else:
        raise E3InputError("Either --round23-eval-report or --model-dir must be provided")

    audit = audit_inputs(context_rows, round19_rows, predictions)
    policy_context_rows = build_policy_context_rows(
        context_rows=context_rows,
        round19_rows=round19_rows,
        round23_predictions=predictions,
    )
    audit = {
        **audit,
        "policy_row_count": len(policy_context_rows),
        "scope": scope,
    }
    tables = {
        "e3_table_overall_policy_quality": summarize_overall(policy_context_rows),
        "e3_table_datasetwise_policy_quality": summarize_datasetwise(policy_context_rows),
        "e3_table_action_distribution": summarize_action_distribution(policy_context_rows),
        "e3_table_split_level_policy_quality": summarize_split_level(policy_context_rows),
        "e3_table_direction_baseline": summarize_direction_baseline(policy_context_rows),
    }
    provenance = {
        "controller_context_table": str(Path(controller_context_table).resolve()),
        "round19_replay_table": str(Path(round19_replay_table).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "scope": scope,
        **prediction_provenance,
    }
    write_e3_outputs(
        output_dir=output_dir,
        policy_context_rows=policy_context_rows,
        tables=tables,
        audit=audit,
        provenance=provenance,
    )
    return {
        "provenance": provenance,
        "audit": audit,
        "tables": tables,
        "policy_context_rows": policy_context_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Round23 E3 controller policy quality on offline local budget sweeps.")
    parser.add_argument("--controller-context-table", required=True)
    parser.add_argument("--round19-replay-table", required=True)
    parser.add_argument("--round23-eval-report", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model-family", default="extratrees")
    parser.add_argument("--feature-version", default="no_dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", default="all_contexts", choices=["all_contexts"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_e3_policy_quality(
        controller_context_table=args.controller_context_table,
        round19_replay_table=args.round19_replay_table,
        round23_eval_report=args.round23_eval_report,
        model_dir=args.model_dir,
        model_family=args.model_family,
        feature_version=args.feature_version,
        config_path=args.config,
        output_dir=args.output_dir,
        scope=args.scope,
    )
    overall = report["tables"]["e3_table_overall_policy_quality"]
    round23_row = next(row for row in overall if row["policy"] == POLICY_ROUND23)
    print(
        "E3 policy quality "
        f"contexts={report['audit']['policy_row_count']} "
        f"round23_mean_reward={as_float(round23_row['mean_reward']):.6f} "
        f"round23_mean_regret={as_float(round23_row['mean_regret_vs_oracle']):.6f} "
        f"output_dir={Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
