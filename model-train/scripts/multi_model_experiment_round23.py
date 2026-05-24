from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent

if str(MODEL_TRAIN_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MODEL_TRAIN_ROOT))

from common import DEFAULT_ROUND23_DATASET_DIR, DEFAULT_ROUND23_REPORT_DIR, DEFAULT_ROUND23_SPLIT_DIR, dump_json, ensure_dir
from eval_round23_controller import evaluate_controller
from round23_experiment_matrix import build_experiment_matrix
from round23_model_zoo import FORMAL_MODEL_FAMILIES
from train_round23_controller import train_controller_models


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Round23 Controller Multi-Model Comparison",
        "",
        "| Model | Feature Version | Target | Seen BestTop1 Regret | Seen Win vs k0 | Seen Delta-k Accuracy | Unseen BestTop1 Regret | Unseen Win vs k0 | Unseen Delta-k Accuracy |",
        "|------|-----------------|--------|----------------------|----------------|-----------------------|------------------------|------------------|-------------------------|",
    ]
    for row in rows:
        unseen_top1_regret = "" if row["unseen_mean_best_top1_regret"] is None else f"{row['unseen_mean_best_top1_regret']:.6f}"
        unseen_win = "" if row["unseen_win_rate_vs_keep_k0_by_best_top1"] is None else f"{row['unseen_win_rate_vs_keep_k0_by_best_top1']:.4f}"
        unseen_acc = "" if row["unseen_delta_k_accuracy"] is None else f"{row['unseen_delta_k_accuracy']:.4f}"
        lines.append(
            f"| {row['model_family']} | {row['feature_version']} | {row['target_mode']} | "
            f"{row['mean_best_top1_regret']:.6f} | {row['win_rate_vs_keep_k0_by_best_top1']:.4f} | "
            f"{row['delta_k_accuracy']:.4f} | {unseen_top1_regret} | {unseen_win} | {unseen_acc} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_multi_model_experiment(
    *,
    controller_samples_path: str | Path,
    controller_context_table_path: str | Path,
    final_test_path: str | Path,
    unseen_test_path: str | Path | None,
    cv_folds_path: str | Path,
    config_path: str | Path,
    work_root: str | Path,
    model_families: list[str] | None = None,
    feature_versions: list[str] | None = None,
    round19_replay_path: str | Path | None = None,
    target_field: str = "reward_round23_controller",
    target_mode: str = "",
    tie_margin: float | None = None,
) -> dict[str, Any]:
    output_root = Path(work_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in build_experiment_matrix(
        model_families=model_families,
        feature_versions=feature_versions,
    ):
        family = entry["model_family"]
        feature_version = entry["feature_version"]
        family_model_dir = ensure_dir(output_root / "models" / family / feature_version)
        family_report_dir = ensure_dir(output_root / "reports" / family / feature_version)
        try:
            train_controller_models(
                controller_samples_path=controller_samples_path,
                final_test_path=final_test_path,
                unseen_test_path=unseen_test_path,
                cv_folds_path=cv_folds_path,
                config_path=config_path,
                model_output_dir=family_model_dir,
                model_family=family,
                feature_version=feature_version,
                target_field=target_field,
            )
            eval_report = evaluate_controller(
                controller_context_table_path=controller_context_table_path,
                context_split_path=final_test_path,
                context_split_key="final_test_context_ids",
                config_path=config_path,
                model_dir=family_model_dir,
                report_dir=family_report_dir,
                model_family=family,
                feature_version=feature_version,
                round19_replay_path=round19_replay_path,
                target_field=target_field,
            )
            unseen_eval_report = None
            if unseen_test_path is not None:
                unseen_eval_report = evaluate_controller(
                    controller_context_table_path=controller_context_table_path,
                    context_split_path=unseen_test_path,
                    context_split_key="unseen_test_context_ids",
                    config_path=config_path,
                    model_dir=family_model_dir,
                    report_dir=ensure_dir(family_report_dir / "unseen"),
                    model_family=family,
                    feature_version=feature_version,
                    round19_replay_path=round19_replay_path,
                    target_field=target_field,
                )
            rows.append(
                {
                    "model_family": family,
                    "feature_version": feature_version,
                    "target_field": target_field,
                    "target_mode": target_mode or str(eval_report.get("target_mode", "")),
                    "tie_margin": tie_margin,
                    "mean_predicted_reward": float(eval_report["mean_predicted_reward"]),
                    "mean_predicted_regret": float(eval_report["mean_predicted_regret"]),
                    "mean_best_top1_regret": float(eval_report["mean_best_top1_regret"]),
                    "win_rate_vs_keep_k0_by_best_top1": float(eval_report["win_rate_vs_keep_k0_by_best_top1"]),
                    "delta_k_accuracy": float(eval_report["delta_k_accuracy"]),
                    "direction_accuracy": float(eval_report["direction_accuracy"]),
                    "unseen_mean_predicted_reward": (
                        float(unseen_eval_report["mean_predicted_reward"]) if unseen_eval_report else None
                    ),
                    "unseen_mean_predicted_regret": (
                        float(unseen_eval_report["mean_predicted_regret"]) if unseen_eval_report else None
                    ),
                    "unseen_mean_best_top1_regret": (
                        float(unseen_eval_report["mean_best_top1_regret"]) if unseen_eval_report else None
                    ),
                    "unseen_win_rate_vs_keep_k0_by_best_top1": (
                        float(unseen_eval_report["win_rate_vs_keep_k0_by_best_top1"]) if unseen_eval_report else None
                    ),
                    "unseen_delta_k_accuracy": (
                        float(unseen_eval_report["delta_k_accuracy"]) if unseen_eval_report else None
                    ),
                    "unseen_direction_accuracy": (
                        float(unseen_eval_report["direction_accuracy"]) if unseen_eval_report else None
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover
            skipped.append(
                {
                    "model_family": family,
                    "feature_version": feature_version,
                    "reason": str(exc),
                }
            )

    rows = sorted(
        rows,
        key=lambda row: (
            row["unseen_mean_best_top1_regret"] if row["unseen_mean_best_top1_regret"] is not None else row["mean_best_top1_regret"],
            row["mean_best_top1_regret"],
        ),
    )
    csv_path = output_root / "round23_model_comparison.csv"
    csv_lines = [
        "model_family,feature_version,target_field,target_mode,tie_margin,mean_predicted_reward,mean_predicted_regret,mean_best_top1_regret,win_rate_vs_keep_k0_by_best_top1,delta_k_accuracy,direction_accuracy,unseen_mean_predicted_reward,unseen_mean_predicted_regret,unseen_mean_best_top1_regret,unseen_win_rate_vs_keep_k0_by_best_top1,unseen_delta_k_accuracy,unseen_direction_accuracy"
    ]
    for row in rows:
        csv_lines.append(
            f"{row['model_family']},{row['feature_version']},{row['target_field']},{row['target_mode']},{'' if row['tie_margin'] is None else row['tie_margin']},{row['mean_predicted_reward']},"
            f"{row['mean_predicted_regret']},{row['mean_best_top1_regret']},{row['win_rate_vs_keep_k0_by_best_top1']},{row['delta_k_accuracy']},{row['direction_accuracy']},"
            f"{'' if row['unseen_mean_predicted_reward'] is None else row['unseen_mean_predicted_reward']},"
            f"{'' if row['unseen_mean_predicted_regret'] is None else row['unseen_mean_predicted_regret']},"
            f"{'' if row['unseen_mean_best_top1_regret'] is None else row['unseen_mean_best_top1_regret']},"
            f"{'' if row['unseen_win_rate_vs_keep_k0_by_best_top1'] is None else row['unseen_win_rate_vs_keep_k0_by_best_top1']},"
            f"{'' if row['unseen_delta_k_accuracy'] is None else row['unseen_delta_k_accuracy']},"
            f"{'' if row['unseen_direction_accuracy'] is None else row['unseen_direction_accuracy']}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    result = {"rows": rows, "skipped": skipped}
    dump_json(output_root / "round23_model_comparison.json", result)
    _write_markdown(output_root / "round23_model_comparison.md", rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-model comparison for round23 controller.")
    parser.add_argument("--controller-samples", default=str(DEFAULT_ROUND23_DATASET_DIR / "round23_controller_samples.jsonl"))
    parser.add_argument("--controller-context-table", default=str(DEFAULT_ROUND23_DATASET_DIR / "round23_controller_context_table.jsonl"))
    parser.add_argument("--final-test", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_final_test_contexts.json"))
    parser.add_argument("--unseen-test", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_unseen_test_contexts.json"))
    parser.add_argument("--cv-folds", default=str(DEFAULT_ROUND23_SPLIT_DIR / "round23_cv_folds.json"))
    parser.add_argument("--config", default=str(MODEL_TRAIN_ROOT / "configs" / "train_round23_controller.yaml"))
    parser.add_argument("--work-root", default=str(DEFAULT_ROUND23_REPORT_DIR / "model_comparison"))
    parser.add_argument("--model-families", nargs="*", default=list(FORMAL_MODEL_FAMILIES))
    parser.add_argument("--feature-versions", nargs="*", default=["with_dataset", "no_dataset"])
    parser.add_argument("--round19-replay-path", default=None)
    parser.add_argument("--target-field", default="reward_round23_controller")
    parser.add_argument("--target-mode", default="")
    parser.add_argument("--tie-margin", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_multi_model_experiment(
        controller_samples_path=args.controller_samples,
        controller_context_table_path=args.controller_context_table,
        final_test_path=args.final_test,
        unseen_test_path=args.unseen_test,
        cv_folds_path=args.cv_folds,
        config_path=args.config,
        work_root=args.work_root,
        model_families=args.model_families,
        feature_versions=args.feature_versions,
        round19_replay_path=args.round19_replay_path,
        target_field=args.target_field,
        target_mode=args.target_mode,
        tie_margin=args.tie_margin,
    )
    print(f"COMPARED models={len(result['rows'])} skipped={len(result['skipped'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
