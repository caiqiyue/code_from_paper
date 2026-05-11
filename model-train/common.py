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

BUDGETS = [18, 19, 20, 21, 22]
DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
ROUND22_REWARD_LAMBDA = 0.002


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


def normalize_record_key(dataset_name: Any, meta_seed: Any) -> tuple[str, int]:
    return str(dataset_name), int(meta_seed)

