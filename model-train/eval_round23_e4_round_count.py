from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import BUDGETS, REFERENCE_BUDGET, as_float, dump_json, ensure_dir, load_yaml, read_jsonl, write_csv, write_jsonl
from eval_round23_e3_policy_quality import (
    E3InputError,
    _fieldnames,
    _mean,
    _round_or_none,
    audit_inputs,
    best_top1_field_for_delta,
    build_policy_context_rows,
    direction,
    infer_round23_predictions_from_model,
    load_round23_predictions_from_eval_report,
    reward_field_for_delta,
    write_audit_markdown,
)
from features import encode_feature_row_with_fields
from round23_controller_models import load_regressor, model_file_extension, predict_regressor, require_model_family
from round23_feature_sets import resolve_feature_spec


POLICY_KEEP = "keep-k0=20"
POLICY_ABSOLUTE_K = "predict absolute k"
POLICY_ROUND23 = "round23"
POLICY_ORACLE = "oracle budget"
POLICIES = [POLICY_KEEP, POLICY_ABSOLUTE_K, POLICY_ROUND23]


def infer_absolute_k_predictions_from_model(
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
    absolute_budgets = [int(value) for value in model_cfg.get("absolute_budgets", BUDGETS)]
    extension = model_file_extension(normalized_family)
    models: dict[int, Any] = {}
    for budget in absolute_budgets:
        models[budget] = load_regressor(
            family=normalized_family,
            path=Path(model_dir) / f"model_k{budget}{extension}",
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
            int(budget): float(
                predict_regressor(
                    family=normalized_family,
                    model=models[int(budget)],
                    feature_matrix=[feature_values],
                )[0]
            )
            for budget in absolute_budgets
        }
        predicted_budget = max(predicted_rewards, key=predicted_rewards.get)
        predictions[str(row["context_id"])] = {
            "predicted_absolute_k": int(predicted_budget),
            "predicted_delta_k": int(predicted_budget) - int(REFERENCE_BUDGET),
            "predicted_target_budget": int(predicted_budget),
            "predicted_rewards": predicted_rewards,
        }
    provenance = {
        "prediction_source": "absolute_k_model_dir",
        "model_dir": str(Path(model_dir).resolve()),
        "model_family": normalized_family,
        "feature_version": feature_spec.feature_version,
        "feature_fields": feature_spec.feature_fields,
        "include_dataset_one_hot": feature_spec.include_dataset_onehot,
        "onehot_order": feature_spec.onehot_order,
        "absolute_budgets": absolute_budgets,
        "config_path": None if config_path is None else str(Path(config_path).resolve()),
    }
    return predictions, provenance


def _selected_reward_for_budget(context_row: dict[str, Any], budget: int) -> float:
    delta_k = int(budget) - int(REFERENCE_BUDGET)
    return float(context_row[reward_field_for_delta(delta_k)])


def _selected_best_top1_for_budget(context_row: dict[str, Any], budget: int) -> float:
    delta_k = int(budget) - int(REFERENCE_BUDGET)
    return float(context_row[best_top1_field_for_delta(delta_k)])


def build_e4_policy_context_rows(
    *,
    context_rows: list[dict[str, Any]],
    round19_rows: list[dict[str, Any]],
    round23_predictions: dict[str, dict[str, Any]],
    absolute_k_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_inputs(context_rows, round19_rows, round23_predictions)
    context_ids = {str(row["context_id"]) for row in context_rows}
    if set(absolute_k_predictions) != context_ids:
        missing = sorted(context_ids - set(absolute_k_predictions))
        extra = sorted(set(absolute_k_predictions) - context_ids)
        raise E3InputError(f"Absolute-k prediction mismatch missing={missing} extra={extra}")

    round23_rows = {
        str(row["context_id"]): row
        for row in build_policy_context_rows(
            context_rows=context_rows,
            round19_rows=round19_rows,
            round23_predictions=round23_predictions,
        )
    }
    rows: list[dict[str, Any]] = []
    for context_row in context_rows:
        context_id = str(context_row["context_id"])
        round23_row = round23_rows[context_id]
        absolute_pred = absolute_k_predictions[context_id]
        predicted_budget = int(absolute_pred["predicted_absolute_k"])
        predicted_delta = int(absolute_pred["predicted_delta_k"])
        absolute_reward = _selected_reward_for_budget(context_row, predicted_budget)
        oracle_reward = float(round23_row["oracle_reward"])
        absolute_top1 = _selected_best_top1_for_budget(context_row, predicted_budget)
        oracle_top1 = float(round23_row["oracle_best_top1"])
        keep_top1 = float(round23_row["keep_best_top1"])
        rows.append(
            {
                **round23_row,
                "absolute_k_predicted_budget": predicted_budget,
                "absolute_k_predicted_delta_k": predicted_delta,
                "absolute_k_reward": absolute_reward,
                "absolute_k_best_top1": absolute_top1,
                "absolute_k_regret_vs_oracle": oracle_reward - absolute_reward,
                "absolute_k_best_top1_regret": oracle_top1 - absolute_top1,
                "absolute_k_win_vs_keep_by_reward": float(absolute_reward > float(round23_row["keep_reward"])),
                "absolute_k_direction_correct": float(direction(predicted_delta) == direction(int(round23_row["oracle_delta_k"]))),
                "absolute_k_delta_k_correct": float(predicted_delta == int(round23_row["oracle_delta_k"])),
                "absolute_k_predicted_rewards": absolute_pred.get("predicted_rewards", {}),
                "absolute_k_win_vs_keep_by_best_top1": float(absolute_top1 > keep_top1),
            }
        )
    return rows


def _policy_values(row: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy == POLICY_KEEP:
        return {
            "reward": row["keep_reward"],
            "regret": row["keep_regret_vs_oracle"],
            "best_top1_regret": row["keep_best_top1_regret"],
            "win_vs_keep": None,
            "direction_correct": row["keep_direction_correct"],
            "delta_correct": row["keep_delta_k_correct"],
        }
    if policy == POLICY_ABSOLUTE_K:
        return {
            "reward": row["absolute_k_reward"],
            "regret": row["absolute_k_regret_vs_oracle"],
            "best_top1_regret": row["absolute_k_best_top1_regret"],
            "win_vs_keep": row["absolute_k_win_vs_keep_by_reward"],
            "direction_correct": row["absolute_k_direction_correct"],
            "delta_correct": row["absolute_k_delta_k_correct"],
        }
    if policy == POLICY_ROUND23:
        return {
            "reward": row["round23_reward"],
            "regret": row["round23_regret_vs_oracle"],
            "best_top1_regret": row["round23_best_top1_regret"],
            "win_vs_keep": row["round23_win_vs_keep_by_reward"],
            "direction_correct": row["round23_direction_correct"],
            "delta_correct": row["round23_delta_k_correct"],
        }
    if policy == POLICY_ORACLE:
        return {
            "reward": row["oracle_reward"],
            "regret": 0.0,
            "best_top1_regret": 0.0,
            "win_vs_keep": row["oracle_win_vs_keep_by_reward"],
            "direction_correct": 1.0,
            "delta_correct": 1.0,
        }
    raise KeyError(policy)


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
                "direction_accuracy": _round_or_none(_mean([value["direction_correct"] for value in values])),
                "delta_k_accuracy": _round_or_none(_mean([value["delta_correct"] for value in values])),
                "mean_best_top1_regret": _round_or_none(_mean([value["best_top1_regret"] for value in values])),
            }
        )
    return summary


def summarize_datasetwise(policy_context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for dataset_name in sorted({str(row["dataset_name"]) for row in policy_context_rows}):
        dataset_rows = [row for row in policy_context_rows if str(row["dataset_name"]) == dataset_name]
        for policy in POLICIES:
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


def write_markdown_summary(
    *,
    output_path: str | Path,
    provenance: dict[str, Any],
    overall: list[dict[str, Any]],
    datasetwise: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    lines = [
        "# E4 Round-Count Summary",
        "",
        "This report compares keep-k0, one-shot absolute-k, and formal two-round round23.",
        "",
        "## Provenance",
        "",
        f"- controller_context_table: {provenance.get('controller_context_table')}",
        f"- round19_replay_table: {provenance.get('round19_replay_table')}",
        f"- round23_prediction_source: {provenance.get('round23_prediction_source')}",
        f"- absolute_k_prediction_source: {provenance.get('absolute_k_prediction_source')}",
        f"- scope: {provenance.get('scope')}",
        f"- audit_status: {audit.get('status')}",
        "",
        "## Table E4-2 Policy Quality",
        "",
        "| policy | contexts | mean_reward | mean_regret_vs_oracle | win_rate_vs_keep_k0_by_reward | direction_accuracy | delta_k_accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in overall:
        lines.append(
            "| {policy} | {contexts} | {mean_reward} | {mean_regret_vs_oracle} | {win_rate_vs_keep_k0_by_reward} | {direction_accuracy} | {delta_k_accuracy} |".format(
                **{key: ("" if value is None else value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Dataset-wise Policy Quality",
            "",
            "| dataset_name | policy | contexts | mean_reward | mean_regret_vs_oracle | win_rate_vs_keep_k0_by_reward | direction_accuracy | delta_k_accuracy |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in datasetwise:
        lines.append(
            "| {dataset_name} | {policy} | {contexts} | {mean_reward} | {mean_regret_vs_oracle} | {win_rate_vs_keep_k0_by_reward} | {direction_accuracy} | {delta_k_accuracy} |".format(
                **{key: ("" if value is None else value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Audit", "", "```json", __import__("json").dumps(audit, ensure_ascii=False, indent=2), "```"])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_e4_outputs(
    *,
    output_dir: str | Path,
    policy_context_rows: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    output_root = ensure_dir(output_dir)
    write_jsonl(output_root / "e4_policy_contexts.jsonl", policy_context_rows)
    dump_json(output_root / "e4_policy_contexts.json", policy_context_rows)
    dump_json(output_root / "e4_audit_report.json", audit)
    dump_json(output_root / "e4_tables.json", tables)
    for table_name, rows in tables.items():
        if rows:
            write_csv(output_root / f"{table_name}.csv", rows, _fieldnames(rows))
            dump_json(output_root / f"{table_name}.json", rows)
    write_markdown_summary(
        output_path=output_root / "e4_summary.md",
        provenance=provenance,
        overall=tables["e4_table_policy_quality"],
        datasetwise=tables["e4_table_datasetwise_policy_quality"],
        audit=audit,
    )
    write_audit_markdown(
        output_path=output_root / "e4_audit_report.md",
        audit=audit,
        provenance=provenance,
    )


def evaluate_e4_round_count(
    *,
    controller_context_table: str | Path,
    round19_replay_table: str | Path,
    output_dir: str | Path,
    absolute_k_model_dir: str | Path,
    absolute_k_model_family: str,
    absolute_k_feature_version: str | None,
    absolute_k_config_path: str | Path | None,
    round23_eval_report: str | Path | None = None,
    round23_model_dir: str | Path | None = None,
    round23_model_family: str = "extratrees",
    round23_feature_version: str | None = "no_dataset",
    round23_config_path: str | Path | None = None,
    scope: str = "seen4",
) -> dict[str, Any]:
    context_rows = read_jsonl(controller_context_table)
    round19_rows = read_jsonl(round19_replay_table)
    absolute_k_predictions, absolute_provenance = infer_absolute_k_predictions_from_model(
        context_rows=context_rows,
        model_dir=absolute_k_model_dir,
        model_family=absolute_k_model_family,
        feature_version=absolute_k_feature_version,
        config_path=absolute_k_config_path,
    )
    if round23_eval_report is not None:
        round23_predictions = load_round23_predictions_from_eval_report(round23_eval_report)
        round23_provenance = {
            "prediction_source": "round23_eval_report",
            "round23_eval_report": str(Path(round23_eval_report).resolve()),
        }
    elif round23_model_dir is not None:
        round23_predictions, round23_provenance = infer_round23_predictions_from_model(
            context_rows=context_rows,
            model_dir=round23_model_dir,
            model_family=round23_model_family,
            feature_version=round23_feature_version,
            config_path=round23_config_path,
        )
    else:
        raise E3InputError("Either round23_eval_report or round23_model_dir must be provided")

    audit = audit_inputs(context_rows, round19_rows, round23_predictions)
    policy_context_rows = build_e4_policy_context_rows(
        context_rows=context_rows,
        round19_rows=round19_rows,
        round23_predictions=round23_predictions,
        absolute_k_predictions=absolute_k_predictions,
    )
    audit = {**audit, "policy_row_count": len(policy_context_rows), "scope": scope}
    tables = {
        "e4_table_policy_quality": summarize_overall(policy_context_rows),
        "e4_table_datasetwise_policy_quality": summarize_datasetwise(policy_context_rows),
    }
    provenance = {
        "controller_context_table": str(Path(controller_context_table).resolve()),
        "round19_replay_table": str(Path(round19_replay_table).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "scope": scope,
        "round23_prediction_source": round23_provenance.get("prediction_source"),
        "absolute_k_prediction_source": absolute_provenance.get("prediction_source"),
        "round23_provenance": round23_provenance,
        "absolute_k_provenance": absolute_provenance,
    }
    write_e4_outputs(
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
    parser = argparse.ArgumentParser(description="Evaluate round23 E4 round-count evidence.")
    parser.add_argument("--controller-context-table", required=True)
    parser.add_argument("--round19-replay-table", required=True)
    parser.add_argument("--absolute-k-model-dir", required=True)
    parser.add_argument("--absolute-k-model-family", default="extratrees")
    parser.add_argument("--absolute-k-feature-version", default="no_dataset")
    parser.add_argument("--absolute-k-config", default=None)
    parser.add_argument("--round23-eval-report", default=None)
    parser.add_argument("--round23-model-dir", default=None)
    parser.add_argument("--round23-model-family", default="extratrees")
    parser.add_argument("--round23-feature-version", default="no_dataset")
    parser.add_argument("--round23-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", default="seen4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_e4_round_count(
        controller_context_table=args.controller_context_table,
        round19_replay_table=args.round19_replay_table,
        output_dir=args.output_dir,
        absolute_k_model_dir=args.absolute_k_model_dir,
        absolute_k_model_family=args.absolute_k_model_family,
        absolute_k_feature_version=args.absolute_k_feature_version,
        absolute_k_config_path=args.absolute_k_config,
        round23_eval_report=args.round23_eval_report,
        round23_model_dir=args.round23_model_dir,
        round23_model_family=args.round23_model_family,
        round23_feature_version=args.round23_feature_version,
        round23_config_path=args.round23_config,
        scope=args.scope,
    )
    round23_row = next(row for row in report["tables"]["e4_table_policy_quality"] if row["policy"] == POLICY_ROUND23)
    print(
        "E4 round-count "
        f"contexts={report['audit']['policy_row_count']} "
        f"round23_mean_reward={as_float(round23_row['mean_reward']):.6f} "
        f"output_dir={Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
