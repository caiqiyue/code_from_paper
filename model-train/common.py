from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAIN_ROOT = Path(__file__).resolve().parent
ROUND22_ROOT = PROJECT_ROOT / "paper-new-round22"
ROUND23_ROOT = PROJECT_ROOT / "paper-new-round23"

DEFAULT_SUMMARY_JSONL = ROUND22_ROOT / "logs" / "round22_bandit_full_summary.jsonl"
DEFAULT_SUMMARY_TSV = ROUND22_ROOT / "logs" / "round22_bandit_full_summary.tsv"
DEFAULT_SCHEMA_YAML = (
    ROUND22_ROOT
    / "configs"
    / "experiments"
    / "bandit_data_collection"
    / "round22_bandit_record_schema.yaml"
)
DEFAULT_DATASET_DIR = MODEL_TRAIN_ROOT / "artifacts" / "datasets"
DEFAULT_SPLIT_DIR = MODEL_TRAIN_ROOT / "artifacts" / "splits"
DEFAULT_MODEL_DIR = MODEL_TRAIN_ROOT / "artifacts" / "models"
DEFAULT_REPORT_DIR = MODEL_TRAIN_ROOT / "artifacts" / "reports"
DEFAULT_DIAGNOSTIC_DIR = MODEL_TRAIN_ROOT / "artifacts" / "diagnostics"
DEFAULT_ROUND23_DATASET_DIR = MODEL_TRAIN_ROOT / "artifacts" / "round23_datasets"
DEFAULT_ROUND23_SPLIT_DIR = MODEL_TRAIN_ROOT / "artifacts" / "round23_splits"
DEFAULT_ROUND23_MODEL_DIR = MODEL_TRAIN_ROOT / "artifacts" / "round23_models"
DEFAULT_ROUND23_REPORT_DIR = MODEL_TRAIN_ROOT / "artifacts" / "round23_reports"

BUDGETS = [18, 19, 20, 21, 22]
REFERENCE_BUDGET = 20
DELTA_ACTIONS = [-2, -1, 0, 1, 2]
DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
ROUND22_REWARD_LAMBDA = 0.002
ROUND23_LABEL_TARGET_MODES = ("top1_delta", "top1_value", "calibrated_reward")
ROUND23_SHAPE_REFERENCE = {
    "median_len": {"mean": 150.0, "std": 50.0},
    "private_p75_length": {"mean": 335.0, "std": 90.0},
    "private_length_iqr": {"mean": 180.0, "std": 60.0},
}
ROUND23_SHAPE_TAU_CENTER = 0.0
ROUND23_SHAPE_DELTA_ROUTER = 0.35


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping at {path}")
    return payload


def dump_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(f"JSONL row {lineno} in {path} is not an object")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    has_pandas = importlib.util.find_spec("pandas") is not None
    has_pyarrow = importlib.util.find_spec("pyarrow") is not None
    if not (has_pandas and has_pyarrow):
        return False
    import pandas as pd  # type: ignore

    df = pd.DataFrame(rows)
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)
    return True


def as_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("Expected numeric value, got empty")
    return float(value)


def as_int(value: Any) -> int:
    if value is None or value == "":
        raise ValueError("Expected integer value, got empty")
    return int(value)


def almost_equal(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=float(tolerance))


def context_id(dataset_name: str, meta_seed: int) -> str:
    return f"{dataset_name}_seed{int(meta_seed)}"


def compute_reward(best_top1: float, normalized_budget_cost: float, reward_lambda: float = ROUND22_REWARD_LAMBDA) -> float:
    return float(best_top1) - float(reward_lambda) * float(normalized_budget_cost)


def resolve_canonical_project_root() -> Path:
    resolved = PROJECT_ROOT.resolve()
    parts = list(resolved.parts)
    if ".worktrees" in parts:
        worktree_index = parts.index(".worktrees")
        return Path(*parts[:worktree_index])
    return resolved


CANONICAL_PROJECT_ROOT = resolve_canonical_project_root()
CANONICAL_MODEL_TRAIN_ROOT = CANONICAL_PROJECT_ROOT / "model-train"
CANONICAL_ROUND22_ROOT = CANONICAL_PROJECT_ROOT / "paper-new-round22"
CANONICAL_ROUND23_ROOT = CANONICAL_PROJECT_ROOT / "paper-new-round23"


def controller_target_budget(delta_k: int, reference_budget: int = REFERENCE_BUDGET) -> int:
    return int(reference_budget) + int(delta_k)


def normalized_delta_cost(delta_k: int, *, max_abs_delta: int = 2) -> float:
    if max_abs_delta <= 0:
        return 0.0
    return abs(int(delta_k)) / float(max_abs_delta)


def compute_round23_controller_reward(
    *,
    best_top1_k0: float,
    best_top1_k1: float,
    coverage_p25_k0: float,
    coverage_p25_k1: float,
    support_mean_k0: float,
    support_mean_k1: float,
    delta_k: int,
    top1_weight: float = 1.0,
    coverage_weight: float = 0.25,
    support_penalty_weight: float = 0.20,
    movement_cost_weight: float = 0.02,
) -> float:
    return (
        float(top1_weight) * (float(best_top1_k1) - float(best_top1_k0))
        + float(coverage_weight) * (float(coverage_p25_k1) - float(coverage_p25_k0))
        - float(support_penalty_weight) * max(0.0, float(support_mean_k0) - float(support_mean_k1))
        - float(movement_cost_weight) * abs(int(delta_k))
    )


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def zscore(value: Any, *, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (_safe_float(value) - float(mean)) / float(std)


def compute_round23_shape_score_from_summary(context_summary: dict[str, Any]) -> float:
    return (
        zscore(
            context_summary.get("median_len", context_summary.get("private_median_length", context_summary.get("private_mean_length"))),
            mean=ROUND23_SHAPE_REFERENCE["median_len"]["mean"],
            std=ROUND23_SHAPE_REFERENCE["median_len"]["std"],
        )
        + zscore(
            context_summary.get("private_p75_length"),
            mean=ROUND23_SHAPE_REFERENCE["private_p75_length"]["mean"],
            std=ROUND23_SHAPE_REFERENCE["private_p75_length"]["std"],
        )
        + zscore(
            context_summary.get("private_length_iqr"),
            mean=ROUND23_SHAPE_REFERENCE["private_length_iqr"]["mean"],
            std=ROUND23_SHAPE_REFERENCE["private_length_iqr"]["std"],
        )
        + _safe_float(context_summary.get("tail_ratio"))
        - _safe_float(context_summary.get("short_ratio"))
    )


def resolve_round23_shape_regime(
    shape_score: float,
    *,
    tau_center: float = ROUND23_SHAPE_TAU_CENTER,
    delta_router: float = ROUND23_SHAPE_DELTA_ROUTER,
) -> str:
    score = float(shape_score)
    if score >= float(tau_center) + float(delta_router):
        return "broad_tail"
    if score <= float(tau_center) - float(delta_router):
        return "compact_structured"
    return "uncertain"


def compute_round23_calibrated_controller_reward(
    *,
    best_top1_k0: float,
    best_top1_k1: float,
    coverage_p25_k0: float,
    coverage_p25_k1: float,
    support_mean_k0: float,
    support_mean_k1: float,
    delta_k: int,
    top1_weight: float = 1.0,
    coverage_weight: float = 0.25,
    support_penalty_weight: float = 0.0,
    movement_cost_weight: float = 0.0005,
) -> float:
    return compute_round23_controller_reward(
        best_top1_k0=best_top1_k0,
        best_top1_k1=best_top1_k1,
        coverage_p25_k0=coverage_p25_k0,
        coverage_p25_k1=coverage_p25_k1,
        support_mean_k0=support_mean_k0,
        support_mean_k1=support_mean_k1,
        delta_k=delta_k,
        top1_weight=top1_weight,
        coverage_weight=coverage_weight,
        support_penalty_weight=support_penalty_weight,
        movement_cost_weight=movement_cost_weight,
    )


def compute_round23_training_target(
    *,
    target_mode: str,
    best_top1_k0: float,
    best_top1_k1: float,
    coverage_p25_k0: float,
    coverage_p25_k1: float,
    support_mean_k0: float,
    support_mean_k1: float,
    delta_k: int,
) -> float:
    normalized_mode = str(target_mode).strip().lower()
    if normalized_mode == "top1_delta":
        return float(best_top1_k1) - float(best_top1_k0)
    if normalized_mode == "top1_value":
        return float(best_top1_k1)
    if normalized_mode == "calibrated_reward":
        return compute_round23_calibrated_controller_reward(
            best_top1_k0=best_top1_k0,
            best_top1_k1=best_top1_k1,
            coverage_p25_k0=coverage_p25_k0,
            coverage_p25_k1=coverage_p25_k1,
            support_mean_k0=support_mean_k0,
            support_mean_k1=support_mean_k1,
            delta_k=delta_k,
        )
    raise ValueError(f"Unsupported round23 target_mode: {target_mode}")


def select_round23_oracle_delta(
    *,
    action_values: dict[int, float],
    top1_by_delta: dict[int, float],
    best_top1_k0: float,
    tie_margin: float,
) -> int:
    best_delta = max(action_values, key=lambda delta_k: (float(action_values[delta_k]), -abs(int(delta_k)), -int(delta_k)))
    best_top1_gain = float(top1_by_delta[int(best_delta)]) - float(best_top1_k0)
    if int(best_delta) != 0 and best_top1_gain < float(tie_margin):
        return 0
    return int(best_delta)


def normalize_record_key(dataset_name: Any, meta_seed: Any) -> tuple[str, int]:
    return str(dataset_name), int(meta_seed)
