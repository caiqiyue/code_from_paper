from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent
if str(MODEL_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAIN_ROOT))

from eval_round23_e3_policy_quality import (  # noqa: E402
    E3InputError,
    audit_inputs,
    best_top1_field_for_delta,
    build_policy_context_rows,
    direction,
    evaluate_e3_policy_quality,
    reward_field_for_delta,
    summarize_action_distribution,
    summarize_datasetwise,
    summarize_overall,
)


def _context_row(context_id: str, dataset_name: str, oracle_delta: int, reward_by_delta: dict[int, float]) -> dict:
    row = {
        "context_id": context_id,
        "dataset_name": dataset_name,
        "meta_seed": int(context_id.rsplit("seed", 1)[1]),
        "reference_budget": 20,
        "label_target_mode": "top1_delta",
        "tie_margin": 0.0005,
        "oracle_best_delta_k": oracle_delta,
        "oracle_best_target_budget": 20 + oracle_delta,
        "oracle_best_controller_reward": reward_by_delta[oracle_delta],
        "oracle_best_top1": 0.50 + reward_by_delta[oracle_delta],
        "keep_k0_reward": reward_by_delta[0],
        "best_top1_at_k20": 0.50 + reward_by_delta[0],
        "shape_score": 0.1,
        "shape_regime": "synthetic",
        "private_mean_length": 10.0,
        "private_p75_length": 12.0,
        "private_length_iqr": 4.0,
        "support_mean_at_k20": 0.3,
        "coverage_mean_at_k20": 0.4,
        "coverage_p25_at_k20": 0.2,
        "genericity_mean_at_k20": 0.1,
        "redundancy_mean_at_k20": 0.05,
    }
    for delta_k, reward in reward_by_delta.items():
        suffix = "0" if delta_k == 0 else f"neg{abs(delta_k)}" if delta_k < 0 else f"pos{delta_k}"
        row[f"controller_reward_dk_{suffix}"] = reward
        row[f"best_top1_dk_{suffix}"] = 0.50 + reward
    return row


def _tiny_inputs() -> tuple[list[dict], list[dict], dict[str, dict]]:
    context_rows = [
        _context_row("jobs_seed1", "jobs", 2, {-2: -0.20, -1: -0.10, 0: 0.00, 1: 0.08, 2: 0.10}),
        _context_row("jobs_seed2", "jobs", -1, {-2: 0.01, -1: 0.06, 0: 0.00, 1: -0.03, 2: -0.04}),
        _context_row("imdb_seed1", "imdb", 0, {-2: -0.05, -1: -0.01, 0: 0.02, 1: 0.00, 2: -0.02}),
    ]
    round19_rows = [
        {
            "context_id": "jobs_seed1",
            "round19_predicted_budget": 18,
            "round19_predicted_delta_k": -2,
            "round19_replay_reward": 999.0,
            "round19_replay_best_top1": 999.0,
        },
        {
            "context_id": "jobs_seed2",
            "round19_predicted_budget": 19,
            "round19_predicted_delta_k": -1,
            "round19_replay_reward": 999.0,
            "round19_replay_best_top1": 999.0,
        },
        {
            "context_id": "imdb_seed1",
            "round19_predicted_budget": 20,
            "round19_predicted_delta_k": 0,
            "round19_replay_reward": 999.0,
            "round19_replay_best_top1": 999.0,
        },
    ]
    round23_predictions = {
        "jobs_seed1": {"predicted_delta_k": 1, "predicted_rewards": {-2: -0.2, -1: -0.1, 0: 0.0, 1: 0.3, 2: 0.2}},
        "jobs_seed2": {"predicted_delta_k": -1, "predicted_rewards": {-2: 0.0, -1: 0.3, 0: 0.0, 1: -0.1, 2: -0.2}},
        "imdb_seed1": {"predicted_delta_k": 2, "predicted_rewards": {-2: -0.2, -1: -0.1, 0: 0.0, 1: 0.1, 2: 0.2}},
    }
    return context_rows, round19_rows, round23_predictions


def test_delta_field_helpers_and_direction_mapping() -> None:
    assert reward_field_for_delta(-2) == "controller_reward_dk_neg2"
    assert reward_field_for_delta(0) == "controller_reward_dk_0"
    assert reward_field_for_delta(2) == "controller_reward_dk_pos2"
    assert best_top1_field_for_delta(-1) == "best_top1_dk_neg1"
    assert best_top1_field_for_delta(1) == "best_top1_dk_pos1"
    assert direction(-2) == -1
    assert direction(0) == 0
    assert direction(2) == 1


def test_build_policy_context_rows_recomputes_round19_reward_from_context_table() -> None:
    context_rows, round19_rows, round23_predictions = _tiny_inputs()

    rows = build_policy_context_rows(
        context_rows=context_rows,
        round19_rows=round19_rows,
        round23_predictions=round23_predictions,
    )

    first = rows[0]
    assert first["context_id"] == "jobs_seed1"
    assert first["keep_reward"] == 0.0
    assert first["round19_delta_k"] == -2
    assert first["round19_reward"] == -0.20
    assert first["round19_legacy_replay_reward"] == 999.0
    assert first["round23_delta_k"] == 1
    assert first["round23_reward"] == 0.08
    assert first["oracle_delta_k"] == 2
    assert first["oracle_reward"] == 0.10
    assert first["round23_regret_vs_oracle"] == pytest.approx(0.02)
    assert first["round23_win_vs_keep_by_reward"] == 1.0
    assert first["round23_direction_correct"] == 1.0
    assert first["round23_delta_k_correct"] == 0.0


def test_summarize_overall_reports_four_policies_regret_wins_and_accuracy() -> None:
    context_rows, round19_rows, round23_predictions = _tiny_inputs()
    rows = build_policy_context_rows(
        context_rows=context_rows,
        round19_rows=round19_rows,
        round23_predictions=round23_predictions,
    )

    overall = {row["policy"]: row for row in summarize_overall(rows)}

    assert set(overall) == {"keep-k0=20", "round19 resolver replay", "round23 controller", "oracle budget"}
    assert overall["keep-k0=20"]["contexts"] == 3
    assert overall["round23 controller"]["mean_reward"] == pytest.approx((0.08 + 0.06 - 0.02) / 3)
    assert overall["round23 controller"]["mean_regret_vs_oracle"] == pytest.approx((0.02 + 0.0 + 0.04) / 3)
    assert overall["round23 controller"]["win_rate_vs_keep_k0_by_reward"] == pytest.approx(2 / 3)
    assert overall["round23 controller"]["win_rate_vs_round19_by_reward"] == pytest.approx(1 / 3)
    assert overall["round23 controller"]["direction_accuracy"] == pytest.approx(2 / 3)
    assert overall["round23 controller"]["delta_k_accuracy"] == pytest.approx(1 / 3)
    assert overall["oracle budget"]["mean_regret_vs_oracle"] == 0.0
    assert overall["oracle budget"]["direction_accuracy"] == 1.0
    assert overall["oracle budget"]["delta_k_accuracy"] == 1.0


def test_datasetwise_and_action_distribution_include_zero_counts_for_all_contexts() -> None:
    context_rows, round19_rows, round23_predictions = _tiny_inputs()
    rows = build_policy_context_rows(
        context_rows=context_rows,
        round19_rows=round19_rows,
        round23_predictions=round23_predictions,
    )

    datasetwise = summarize_datasetwise(rows)
    action_distribution = {row["policy"]: row for row in summarize_action_distribution(rows)}

    assert {(row["dataset_name"], row["policy"]) for row in datasetwise} >= {
        ("jobs", "round23 controller"),
        ("imdb", "round23 controller"),
        ("jobs", "round19 resolver replay"),
        ("imdb", "round19 resolver replay"),
    }
    assert action_distribution["round23 controller"]["count_delta_neg2"] == 0
    assert action_distribution["round23 controller"]["count_delta_neg1"] == 1
    assert action_distribution["round23 controller"]["count_delta_0"] == 0
    assert action_distribution["round23 controller"]["count_delta_pos1"] == 1
    assert action_distribution["round23 controller"]["count_delta_pos2"] == 1
    assert action_distribution["oracle budget"]["count_delta_0"] == 1
    assert action_distribution["round19 resolver replay"]["count_delta_neg2"] == 1


def test_audit_rejects_missing_invalid_and_duplicate_contexts() -> None:
    context_rows, round19_rows, round23_predictions = _tiny_inputs()
    assert audit_inputs(context_rows, round19_rows, round23_predictions)["status"] == "pass"

    with pytest.raises(E3InputError, match="missing_round19_contexts"):
        audit_inputs(context_rows, round19_rows[:-1], round23_predictions)

    invalid_round19 = [dict(row) for row in round19_rows]
    invalid_round19[0]["round19_predicted_delta_k"] = 3
    with pytest.raises(E3InputError, match="invalid_round19_delta"):
        audit_inputs(context_rows, invalid_round19, round23_predictions)

    duplicate_contexts = context_rows + [dict(context_rows[0])]
    with pytest.raises(E3InputError, match="duplicate_context_ids"):
        audit_inputs(duplicate_contexts, round19_rows, round23_predictions)


def test_evaluate_writes_required_e3_outputs(tmp_path: Path) -> None:
    context_rows, round19_rows, round23_predictions = _tiny_inputs()
    context_path = tmp_path / "contexts.jsonl"
    round19_path = tmp_path / "round19.jsonl"
    eval_report_path = tmp_path / "round23_eval_report.json"
    output_dir = tmp_path / "e3_outputs"

    context_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in context_rows) + "\n",
        encoding="utf-8",
    )
    round19_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in round19_rows) + "\n",
        encoding="utf-8",
    )
    eval_report_path.write_text(
        json.dumps(
            {
                "per_context": [
                    {
                        "context_id": context_id,
                        "predicted_delta_k": prediction["predicted_delta_k"],
                        "predicted_target_budget": 20 + prediction["predicted_delta_k"],
                        "predicted_rewards": prediction["predicted_rewards"],
                    }
                    for context_id, prediction in round23_predictions.items()
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_e3_policy_quality(
        controller_context_table=context_path,
        round19_replay_table=round19_path,
        round23_eval_report=eval_report_path,
        output_dir=output_dir,
    )

    assert report["audit"]["status"] == "pass"
    assert report["audit"]["policy_row_count"] == 3
    assert (output_dir / "e3_policy_contexts.jsonl").exists()
    assert (output_dir / "e3_table_overall_policy_quality.csv").exists()
    assert (output_dir / "e3_table_datasetwise_policy_quality.csv").exists()
    assert (output_dir / "e3_table_action_distribution.csv").exists()
    assert (output_dir / "e3_audit_report.json").exists()
    assert (output_dir / "e3_audit_report.md").exists()
    assert (output_dir / "e3_summary.md").exists()
    assert "Table E3-1 Overall Controller Policy Quality" in (output_dir / "e3_summary.md").read_text(
        encoding="utf-8"
    )
