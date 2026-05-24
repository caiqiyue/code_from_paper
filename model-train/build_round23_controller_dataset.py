from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    BUDGETS,
    CANONICAL_MODEL_TRAIN_ROOT,
    CANONICAL_PROJECT_ROOT,
    DEFAULT_ROUND23_DATASET_DIR,
    DELTA_ACTIONS,
    ROUND23_LABEL_TARGET_MODES,
    REFERENCE_BUDGET,
    compute_round23_shape_score_from_summary,
    compute_round23_controller_reward,
    compute_round23_training_target,
    context_id,
    controller_target_budget,
    dump_json,
    maybe_write_parquet,
    read_jsonl,
    resolve_round23_shape_regime,
    select_round23_oracle_delta,
    write_csv,
    write_jsonl,
)
from features import ROUND23_CONTROLLER_STATE_FEATURES
from round19_replay import resolve_round19_budget_from_collection_summary
from round23_dataset_registry import get_dataset_partition, summarize_dataset_rows


def _default_action_samples_path() -> Path:
    return CANONICAL_MODEL_TRAIN_ROOT / "data" / "ready" / "full-500" / "all_action_samples.jsonl"


def _default_context_table_path() -> Path:
    return CANONICAL_MODEL_TRAIN_ROOT / "data" / "ready" / "full-500" / "all_contexts.jsonl"


def _default_collection_manifest_path() -> Path:
    return (
        CANONICAL_PROJECT_ROOT
        / "paper-new-round19"
        / "configs"
        / "experiments"
        / "single_node_tuning_round19"
        / "round23_collection_repeat40"
        / "round19_round23_collection_repeat40_manifest.tsv"
    )


def _coerce_float(row: dict[str, Any], field: str, *, default: float | None = None) -> float:
    value = row.get(field)
    if value is None or value == "":
        if default is None:
            raise KeyError(f"Missing required numeric field: {field}")
        return float(default)
    return float(value)


def _resolve_collection_output_root(manifest_path: str | Path, configured_output_root: str | Path) -> Path:
    path = Path(str(configured_output_root))
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == CANONICAL_PROJECT_ROOT.name:
        path = Path(*parts[1:])
    return (CANONICAL_PROJECT_ROOT / path).resolve()


def _read_collection_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _read_budget_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _budget_metric(row: dict[str, Any], *field_names: str, default: float | None = None) -> float:
    for field_name in field_names:
        if field_name in row and row[field_name] not in (None, ""):
            return float(row[field_name])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing required budget metric. Tried: {field_names}")


def _state_from_context_summary(context_summary: dict[str, Any], budget_metrics_k20: dict[str, Any]) -> dict[str, Any]:
    try:
        shape_score = _coerce_float(context_summary, "shape_score")
    except KeyError:
        shape_score = compute_round23_shape_score_from_summary(context_summary)
    shape_regime = str(context_summary.get("shape_regime") or resolve_round23_shape_regime(shape_score))
    return {
        "context_id": str(context_summary.get("context_id") or context_id(context_summary["dataset_name"], context_summary["meta_seed"])),
        "dataset_name": str(context_summary["dataset_name"]),
        "meta_seed": int(context_summary["meta_seed"]),
        "shape_score": shape_score,
        "shape_regime": shape_regime,
        "private_mean_length": _coerce_float(context_summary, "private_mean_length"),
        "private_p75_length": _coerce_float(context_summary, "private_p75_length"),
        "private_length_iqr": _coerce_float(context_summary, "private_length_iqr"),
        "support_mean_at_k20": float(budget_metrics_k20["support_mean"]),
        "coverage_mean_at_k20": float(budget_metrics_k20["coverage_mean"]),
        "coverage_p25_at_k20": float(budget_metrics_k20["coverage_p25"]),
        "genericity_mean_at_k20": float(budget_metrics_k20["genericity_mean"]),
        "redundancy_mean_at_k20": float(budget_metrics_k20["redundancy_mean"]),
    }


def _delta_field_name(delta_k: int) -> str:
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


def _context_delta_field(prefix: str, delta_k: int) -> str:
    return f"{prefix}_dk_{_delta_suffix(delta_k)}"


def _write_dataset_outputs(
    *,
    output_root: Path,
    controller_rows: list[dict[str, Any]],
    controller_context_rows: list[dict[str, Any]],
    round19_replay_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    controller_fields = [
        "context_id",
        "dataset_name",
        "meta_seed",
        "reference_budget",
        *ROUND23_CONTROLLER_STATE_FEATURES,
        "best_top1_at_k20",
        "action_delta_k",
        "target_budget",
        "target_partition",
        "best_top1_k0",
        "best_top1_k1",
        "delta_top1",
        "target_value_top1",
        "target_delta_top1",
        "target_value_for_training",
        "label_target_mode",
        "tie_margin",
        "coverage_p25_k0",
        "coverage_p25_k1",
        "support_mean_k0",
        "support_mean_k1",
        "genericity_mean_k0",
        "genericity_mean_k1",
        "redundancy_mean_k0",
        "redundancy_mean_k1",
        "reward_round23_controller",
        "reward_round23_controller_old",
    ]
    context_fields = [
        "context_id",
        "dataset_name",
        "meta_seed",
        "reference_budget",
        *ROUND23_CONTROLLER_STATE_FEATURES,
        "best_top1_at_k20",
        "controller_reward_dk_neg2",
        "controller_reward_dk_neg1",
        "controller_reward_dk_0",
        "controller_reward_dk_pos1",
        "controller_reward_dk_pos2",
        "controller_old_reward_dk_neg2",
        "controller_old_reward_dk_neg1",
        "controller_old_reward_dk_0",
        "controller_old_reward_dk_pos1",
        "controller_old_reward_dk_pos2",
        "best_top1_dk_neg2",
        "best_top1_dk_neg1",
        "best_top1_dk_0",
        "best_top1_dk_pos1",
        "best_top1_dk_pos2",
        "delta_top1_dk_neg2",
        "delta_top1_dk_neg1",
        "delta_top1_dk_0",
        "delta_top1_dk_pos1",
        "delta_top1_dk_pos2",
        "keep_k0_reward",
        "keep_k0_training_value",
        "oracle_best_delta_k",
        "oracle_best_target_budget",
        "oracle_best_controller_reward",
        "oracle_best_controller_reward_old",
        "oracle_best_training_value",
        "oracle_best_top1",
        "label_target_mode",
        "tie_margin",
    ]
    replay_fields = [
        "context_id",
        "dataset_name",
        "meta_seed",
        "round19_predicted_budget",
        "round19_predicted_delta_k",
        "round19_replay_reward",
        "round19_reward_delta_vs_keep_k20",
        "round19_replay_best_top1",
        "round19_replay_support_mean",
        "round19_replay_coverage_mean",
        "round19_replay_coverage_p25",
        "round19_replay_genericity_mean",
        "round19_replay_redundancy_mean",
        "round19_regime",
        "round19_shape_score",
    ]

    controller_jsonl = output_root / "round23_controller_samples.jsonl"
    controller_csv = output_root / "round23_controller_samples.csv"
    context_jsonl = output_root / "round23_controller_context_table.jsonl"
    context_csv = output_root / "round23_controller_context_table.csv"
    replay_jsonl = output_root / "round19_replay_table.jsonl"
    replay_csv = output_root / "round19_replay_table.csv"
    replay_mapping = output_root / "round19_replay_mapping.json"

    sorted_rows = sorted(
        controller_rows,
        key=lambda row: (str(row["dataset_name"]), int(row["meta_seed"]), int(row["action_delta_k"])),
    )
    sorted_context_rows = sorted(
        controller_context_rows,
        key=lambda row: (str(row["dataset_name"]), int(row["meta_seed"])),
    )
    sorted_replay_rows = sorted(
        round19_replay_rows,
        key=lambda row: (str(row["dataset_name"]), int(row["meta_seed"])),
    )

    write_jsonl(controller_jsonl, sorted_rows)
    write_csv(controller_csv, sorted_rows, controller_fields)
    write_jsonl(context_jsonl, sorted_context_rows)
    write_csv(context_csv, sorted_context_rows, context_fields)
    write_jsonl(replay_jsonl, sorted_replay_rows)
    write_csv(replay_csv, sorted_replay_rows, replay_fields)
    dump_json(replay_mapping, {row["context_id"]: row["round19_predicted_budget"] for row in sorted_replay_rows})

    maybe_write_parquet(output_root / "round23_controller_samples.parquet", sorted_rows)
    maybe_write_parquet(output_root / "round23_controller_context_table.parquet", sorted_context_rows)
    maybe_write_parquet(output_root / "round19_replay_table.parquet", sorted_replay_rows)

    report["outputs"] = {
        "controller_jsonl": str(controller_jsonl),
        "controller_csv": str(controller_csv),
        "context_jsonl": str(context_jsonl),
        "context_csv": str(context_csv),
        "round19_replay_jsonl": str(replay_jsonl),
        "round19_replay_csv": str(replay_csv),
        "round19_replay_mapping": str(replay_mapping),
    }
    dump_json(output_root / "round23_controller_build_report.json", report)
    return report


def _build_from_prototype_paths(
    *,
    action_samples_path: str | Path,
    context_table_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    action_rows = read_jsonl(action_samples_path)
    context_rows = read_jsonl(context_table_path)
    action_index = {(str(row["context_id"]), int(row["action_budget"])): row for row in action_rows}
    grouped_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped_actions[str(row["context_id"])].append(row)

    controller_rows: list[dict[str, Any]] = []
    controller_context_rows: list[dict[str, Any]] = []
    availability_summary = {
        "coverage_p25_target_available_count": 0,
        "support_mean_target_available_count": 0,
        "redundancy_feature_available_count": 0,
    }

    for context_row in context_rows:
        state = {
            "context_id": str(context_row["context_id"]),
            "dataset_name": str(context_row["dataset_name"]),
            "meta_seed": int(context_row["meta_seed"]),
        }
        for field in ROUND23_CONTROLLER_STATE_FEATURES:
            default = 0.0 if field == "redundancy_mean_at_k20" else None
            state[field] = _coerce_float(context_row, field, default=default)
        context_key = state["context_id"]
        budgets = sorted(int(row["action_budget"]) for row in grouped_actions[context_key])
        if budgets != BUDGETS:
            raise ValueError(f"Context {context_key} missing budgets {BUDGETS}: got {budgets}")
        if "redundancy_mean_at_k20" in context_row:
            availability_summary["redundancy_feature_available_count"] += 1

        best_top1_k0 = _coerce_float(context_row, "best_top1_k20")
        coverage_p25_k0 = _coerce_float(context_row, "coverage_p25_k20", default=_coerce_float(context_row, "coverage_p25_at_k20"))
        support_mean_k0 = _coerce_float(context_row, "support_mean_k20", default=_coerce_float(context_row, "support_mean_at_k20"))
        action_rewards: dict[int, float] = {}
        action_targets: dict[int, int] = {}
        for delta_k in DELTA_ACTIONS:
            target_budget = controller_target_budget(delta_k)
            target_action_row = action_index[(context_key, target_budget)]
            best_top1_k1 = _coerce_float(context_row, f"best_top1_k{target_budget}")
            coverage_p25_k1 = _coerce_float(context_row, f"coverage_p25_k{target_budget}", default=coverage_p25_k0)
            support_mean_k1 = _coerce_float(context_row, f"support_mean_k{target_budget}", default=support_mean_k0)
            reward_value = compute_round23_controller_reward(
                best_top1_k0=best_top1_k0,
                best_top1_k1=best_top1_k1,
                coverage_p25_k0=coverage_p25_k0,
                coverage_p25_k1=coverage_p25_k1,
                support_mean_k0=support_mean_k0,
                support_mean_k1=support_mean_k1,
                delta_k=delta_k,
            )
            controller_rows.append(
                {
                    **state,
                    "reference_budget": REFERENCE_BUDGET,
                    "best_top1_at_k20": best_top1_k0,
                    "action_delta_k": int(delta_k),
                    "target_budget": int(target_budget),
                    "target_partition": "prototype",
                    "target_experiment_id": str(target_action_row.get("experiment_id", "")),
                    "best_top1_k0": best_top1_k0,
                    "best_top1_k1": best_top1_k1,
                    "delta_top1": best_top1_k1 - best_top1_k0,
                    "coverage_p25_k0": coverage_p25_k0,
                    "coverage_p25_k1": coverage_p25_k1,
                    "support_mean_k0": support_mean_k0,
                    "support_mean_k1": support_mean_k1,
                    "genericity_mean_k0": float(context_row.get("genericity_mean_at_k20", 0.0)),
                    "genericity_mean_k1": float(context_row.get("genericity_mean_at_k20", 0.0)),
                    "redundancy_mean_k0": float(context_row.get("redundancy_mean_at_k20", 0.0)),
                    "redundancy_mean_k1": float(context_row.get("redundancy_mean_at_k20", 0.0)),
                    "reward_round23_controller": reward_value,
                }
            )
            action_rewards[int(delta_k)] = float(reward_value)
            action_targets[int(delta_k)] = int(target_budget)
        oracle_best_delta = max(DELTA_ACTIONS, key=lambda delta_k: action_rewards[int(delta_k)])
        controller_context_rows.append(
            {
                **state,
                "reference_budget": REFERENCE_BUDGET,
                "best_top1_at_k20": best_top1_k0,
                **{_delta_field_name(delta_k): action_rewards[int(delta_k)] for delta_k in DELTA_ACTIONS},
                "keep_k0_reward": action_rewards[0],
                "oracle_best_delta_k": int(oracle_best_delta),
                "oracle_best_target_budget": action_targets[int(oracle_best_delta)],
                "oracle_best_controller_reward": action_rewards[int(oracle_best_delta)],
            }
        )

    output_root = Path(output_dir)
    report = {
        "build_mode": "prototype_full500",
        "action_samples_path": str(Path(action_samples_path).resolve()),
        "context_table_path": str(Path(context_table_path).resolve()),
        "controller_sample_count": len(controller_rows),
        "context_count": len(controller_context_rows),
        "reference_budget": REFERENCE_BUDGET,
        "delta_actions": DELTA_ACTIONS,
        "feature_fields": list(ROUND23_CONTROLLER_STATE_FEATURES),
        "metric_availability": availability_summary,
        "dataset_partition_summary": {},
    }
    return _write_dataset_outputs(
        output_root=output_root,
        controller_rows=controller_rows,
        controller_context_rows=controller_context_rows,
        round19_replay_rows=[],
        report=report,
    )


def _build_from_collection_manifest(
    *,
    collection_manifest_path: str | Path,
    output_dir: str | Path,
    target_mode: str = "top1_delta",
    tie_margin: float = 0.0005,
) -> dict[str, Any]:
    target_mode = str(target_mode).strip().lower()
    if target_mode not in ROUND23_LABEL_TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={target_mode}. Expected one of {ROUND23_LABEL_TARGET_MODES}")
    grouped_contexts: dict[str, dict[str, Any]] = {}
    for spec in _read_collection_manifest(collection_manifest_path):
        dataset_name = str(spec["dataset_name"])
        meta_seed = int(spec["meta_seed"])
        budget_k = int(spec["budget_k"])
        output_root = _resolve_collection_output_root(collection_manifest_path, spec["output_root"])
        collection_dir = output_root / "collection"
        context_summary = _read_json(collection_dir / "context_summary.json")
        budget_rows = _read_budget_rows(collection_dir / "budget_table.jsonl")
        final_result = _read_json(collection_dir / "final_result_summary.json")
        matching_budget_rows = [row for row in budget_rows if int(row.get("budget_k", row.get("seed_top_k", -1))) == budget_k]
        budget_row = matching_budget_rows[0] if matching_budget_rows else (budget_rows[0] if len(budget_rows) == 1 else None)
        if budget_row is None:
            raise KeyError(f"Missing budget row for budget_k={budget_k} under {collection_dir}")
        context_key = str(context_summary.get("context_id") or context_id(dataset_name, meta_seed))
        bucket = grouped_contexts.setdefault(
            context_key,
            {
                "context_summary": dict(context_summary, context_id=context_key, dataset_name=dataset_name, meta_seed=meta_seed),
                "budgets": {},
            },
        )
        bucket["budgets"][budget_k] = {
            "budget_k": budget_k,
            "best_top1": float(final_result["best_top1"]),
            "support_mean": _budget_metric(budget_row, "support_mean_k", "support_mean", "support_score"),
            "support_score": _budget_metric(budget_row, "support_mean_k", "support_mean", "support_score"),
            "coverage_mean": _budget_metric(budget_row, "coverage_mean_k", "coverage_mean"),
            "coverage_p25": _budget_metric(budget_row, "coverage_p25_k", "coverage_p25"),
            "coverage_min": _budget_metric(budget_row, "coverage_min_k", "coverage_min", default=_budget_metric(budget_row, "coverage_p25_k", "coverage_p25")),
            "genericity_mean": _budget_metric(budget_row, "genericity_mean_k", "genericity_mean", "genericity_score"),
            "genericity_score": _budget_metric(budget_row, "genericity_mean_k", "genericity_mean", "genericity_score"),
            "redundancy_mean": _budget_metric(budget_row, "redundancy_mean_k", "redundancy_mean", "redundancy_score", default=0.0),
            "redundancy_score": _budget_metric(budget_row, "redundancy_mean_k", "redundancy_mean", "redundancy_score", default=0.0),
            "selected_seed_count": int(budget_row.get("selected_seed_count", budget_row.get("selected_count", budget_k))),
            "budget_cost": _budget_metric(budget_row, "budget_cost", default=float(budget_k - min(BUDGETS))),
            "normalized_budget_cost": _budget_metric(
                budget_row,
                "normalized_budget_cost",
                default=float(budget_k - min(BUDGETS)) / float(max(BUDGETS) - min(BUDGETS)),
            ),
        }

    controller_rows: list[dict[str, Any]] = []
    controller_context_rows: list[dict[str, Any]] = []
    round19_replay_rows: list[dict[str, Any]] = []
    context_rows_for_summary: list[dict[str, Any]] = []

    for context_key, bundle in sorted(grouped_contexts.items()):
        metrics_by_budget = dict(bundle["budgets"])
        missing_budgets = [budget for budget in BUDGETS if budget not in metrics_by_budget]
        if missing_budgets:
            raise ValueError(f"Context {context_key} missing budgets {missing_budgets}")
        context_summary = dict(bundle["context_summary"])
        state = _state_from_context_summary(context_summary, metrics_by_budget[REFERENCE_BUDGET])
        state["context_id"] = context_key
        context_rows_for_summary.append(
            {
                "context_id": context_key,
                "dataset_name": state["dataset_name"],
                "meta_seed": state["meta_seed"],
            }
        )
        best_top1_k0 = float(metrics_by_budget[REFERENCE_BUDGET]["best_top1"])
        action_rewards_old: dict[int, float] = {}
        action_training_values: dict[int, float] = {}
        action_top1_values: dict[int, float] = {}
        action_top1_deltas: dict[int, float] = {}
        action_targets: dict[int, int] = {}
        for delta_k in DELTA_ACTIONS:
            target_budget = controller_target_budget(delta_k)
            target_metrics = metrics_by_budget[target_budget]
            best_top1_k1 = float(target_metrics["best_top1"])
            coverage_p25_k0 = float(metrics_by_budget[REFERENCE_BUDGET]["coverage_p25"])
            coverage_p25_k1 = float(target_metrics["coverage_p25"])
            support_mean_k0 = float(metrics_by_budget[REFERENCE_BUDGET]["support_mean"])
            support_mean_k1 = float(target_metrics["support_mean"])
            reward_value_old = compute_round23_controller_reward(
                best_top1_k0=best_top1_k0,
                best_top1_k1=best_top1_k1,
                coverage_p25_k0=coverage_p25_k0,
                coverage_p25_k1=coverage_p25_k1,
                support_mean_k0=support_mean_k0,
                support_mean_k1=support_mean_k1,
                delta_k=delta_k,
            )
            target_value = compute_round23_training_target(
                target_mode=target_mode,
                best_top1_k0=best_top1_k0,
                best_top1_k1=best_top1_k1,
                coverage_p25_k0=coverage_p25_k0,
                coverage_p25_k1=coverage_p25_k1,
                support_mean_k0=support_mean_k0,
                support_mean_k1=support_mean_k1,
                delta_k=delta_k,
            )
            target_delta_top1 = best_top1_k1 - best_top1_k0
            controller_rows.append(
                {
                    **state,
                    "reference_budget": REFERENCE_BUDGET,
                    "best_top1_at_k20": best_top1_k0,
                    "action_delta_k": int(delta_k),
                    "target_budget": int(target_budget),
                    "target_partition": get_dataset_partition(state["dataset_name"]),
                    "best_top1_k0": best_top1_k0,
                    "best_top1_k1": best_top1_k1,
                    "delta_top1": target_delta_top1,
                    "target_value_top1": best_top1_k1,
                    "target_delta_top1": target_delta_top1,
                    "target_value_for_training": target_value,
                    "label_target_mode": target_mode,
                    "tie_margin": float(tie_margin),
                    "coverage_p25_k0": coverage_p25_k0,
                    "coverage_p25_k1": coverage_p25_k1,
                    "support_mean_k0": support_mean_k0,
                    "support_mean_k1": support_mean_k1,
                    "genericity_mean_k0": float(metrics_by_budget[REFERENCE_BUDGET]["genericity_mean"]),
                    "genericity_mean_k1": float(target_metrics["genericity_mean"]),
                    "redundancy_mean_k0": float(metrics_by_budget[REFERENCE_BUDGET]["redundancy_mean"]),
                    "redundancy_mean_k1": float(target_metrics["redundancy_mean"]),
                    "reward_round23_controller": target_value,
                    "reward_round23_controller_old": reward_value_old,
                }
            )
            action_rewards_old[int(delta_k)] = float(reward_value_old)
            action_training_values[int(delta_k)] = float(target_value)
            action_top1_values[int(delta_k)] = float(best_top1_k1)
            action_top1_deltas[int(delta_k)] = float(target_delta_top1)
            action_targets[int(delta_k)] = int(target_budget)
        oracle_best_delta = select_round23_oracle_delta(
            action_values=action_training_values,
            top1_by_delta=action_top1_values,
            best_top1_k0=best_top1_k0,
            tie_margin=float(tie_margin),
        )
        oracle_best_delta_old = max(DELTA_ACTIONS, key=lambda delta_k: action_rewards_old[int(delta_k)])
        controller_context_rows.append(
            {
                **state,
                "reference_budget": REFERENCE_BUDGET,
                "best_top1_at_k20": best_top1_k0,
                **{_delta_field_name(delta_k): action_training_values[int(delta_k)] for delta_k in DELTA_ACTIONS},
                **{_context_delta_field("controller_old_reward", delta_k): action_rewards_old[int(delta_k)] for delta_k in DELTA_ACTIONS},
                **{_context_delta_field("best_top1", delta_k): action_top1_values[int(delta_k)] for delta_k in DELTA_ACTIONS},
                **{_context_delta_field("delta_top1", delta_k): action_top1_deltas[int(delta_k)] for delta_k in DELTA_ACTIONS},
                "keep_k0_reward": action_training_values[0],
                "keep_k0_training_value": action_training_values[0],
                "oracle_best_delta_k": int(oracle_best_delta),
                "oracle_best_target_budget": action_targets[int(oracle_best_delta)],
                "oracle_best_controller_reward": action_training_values[int(oracle_best_delta)],
                "oracle_best_controller_reward_old": action_rewards_old[int(oracle_best_delta_old)],
                "oracle_best_training_value": action_training_values[int(oracle_best_delta)],
                "oracle_best_top1": action_top1_values[int(oracle_best_delta)],
                "label_target_mode": target_mode,
                "tie_margin": float(tie_margin),
            }
        )

        round19_resolved = resolve_round19_budget_from_collection_summary(
            context_summary=context_summary,
            metrics_by_budget=metrics_by_budget,
        )
        round19_budget = int(round19_resolved["resolved_seed_top_k"])
        round19_delta = round19_budget - REFERENCE_BUDGET
        round19_target = metrics_by_budget[round19_budget]
        round19_reward = compute_round23_controller_reward(
            best_top1_k0=best_top1_k0,
            best_top1_k1=float(round19_target["best_top1"]),
            coverage_p25_k0=float(metrics_by_budget[REFERENCE_BUDGET]["coverage_p25"]),
            coverage_p25_k1=float(round19_target["coverage_p25"]),
            support_mean_k0=float(metrics_by_budget[REFERENCE_BUDGET]["support_mean"]),
            support_mean_k1=float(round19_target["support_mean"]),
            delta_k=round19_delta,
        )
        round19_replay_rows.append(
            {
                "context_id": context_key,
                "dataset_name": state["dataset_name"],
                "meta_seed": state["meta_seed"],
                "round19_predicted_budget": round19_budget,
                "round19_predicted_delta_k": round19_delta,
                "round19_replay_reward": round19_reward,
                "round19_reward_delta_vs_keep_k20": round19_reward - action_rewards_old[0],
                "round19_replay_best_top1": float(round19_target["best_top1"]),
                "round19_replay_support_mean": float(round19_target["support_mean"]),
                "round19_replay_coverage_mean": float(round19_target["coverage_mean"]),
                "round19_replay_coverage_p25": float(round19_target["coverage_p25"]),
                "round19_replay_genericity_mean": float(round19_target["genericity_mean"]),
                "round19_replay_redundancy_mean": float(round19_target["redundancy_mean"]),
                "round19_regime": str(round19_resolved.get("regime", "")),
                "round19_shape_score": float(round19_resolved.get("shape_score", 0.0)),
            }
        )

    output_root = Path(output_dir)
    report = {
        "build_mode": "round19_round23_collection_repeat40",
        "collection_manifest_path": str(Path(collection_manifest_path).resolve()),
        "controller_sample_count": len(controller_rows),
        "context_count": len(controller_context_rows),
        "reference_budget": REFERENCE_BUDGET,
        "delta_actions": DELTA_ACTIONS,
        "feature_fields": list(ROUND23_CONTROLLER_STATE_FEATURES),
        "target_mode": target_mode,
        "target_field": "target_value_for_training",
        "tie_margin": float(tie_margin),
        "reward_round23_controller_semantics": "selected_training_target",
        "reward_round23_controller_old_semantics": "legacy_round23_controller_reward",
        "dataset_partition_summary": summarize_dataset_rows(context_rows_for_summary),
        "round19_replay_count": len(round19_replay_rows),
    }
    return _write_dataset_outputs(
        output_root=output_root,
        controller_rows=controller_rows,
        controller_context_rows=controller_context_rows,
        round19_replay_rows=round19_replay_rows,
        report=report,
    )


def build_controller_dataset(
    *,
    action_samples_path: str | Path | None = None,
    context_table_path: str | Path | None = None,
    collection_manifest_path: str | Path | None = None,
    output_dir: str | Path,
    target_mode: str = "top1_delta",
    tie_margin: float = 0.0005,
) -> dict[str, Any]:
    if collection_manifest_path is not None:
        return _build_from_collection_manifest(
            collection_manifest_path=collection_manifest_path,
            output_dir=output_dir,
            target_mode=target_mode,
            tie_margin=tie_margin,
        )
    if action_samples_path is None or context_table_path is None:
        raise TypeError("Prototype mode requires action_samples_path and context_table_path.")
    return _build_from_prototype_paths(
        action_samples_path=action_samples_path,
        context_table_path=context_table_path,
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round23 controller training tables.")
    parser.add_argument("--action-samples", default=str(_default_action_samples_path()))
    parser.add_argument("--context-table", default=str(_default_context_table_path()))
    parser.add_argument("--collection-manifest", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_ROUND23_DATASET_DIR))
    parser.add_argument("--target-mode", choices=list(ROUND23_LABEL_TARGET_MODES), default="top1_delta")
    parser.add_argument("--tie-margin", type=float, default=0.0005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_controller_dataset(
        action_samples_path=args.action_samples,
        context_table_path=args.context_table,
        collection_manifest_path=args.collection_manifest,
        output_dir=args.output_dir,
        target_mode=args.target_mode,
        tie_margin=args.tie_margin,
    )
    print(
        f"BUILT round23_controller_samples={report['controller_sample_count']} "
        f"contexts={report['context_count']} mode={report['build_mode']} output_dir={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
