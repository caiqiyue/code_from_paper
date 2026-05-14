from __future__ import annotations

from typing import Any

from common import DATASET_ORDER


STATE_FEATURES = [
    "shape_score",
    "private_mean_length",
    "private_p75_length",
    "private_length_iqr",
    "support_mean_at_k20",
    "coverage_mean_at_k20",
    "coverage_p25_at_k20",
    "genericity_mean_at_k20",
]

ROUND23_CONTROLLER_STATE_FEATURES = [
    "shape_score",
    "private_mean_length",
    "private_p75_length",
    "private_length_iqr",
    "support_mean_at_k20",
    "coverage_mean_at_k20",
    "coverage_p25_at_k20",
    "genericity_mean_at_k20",
    "redundancy_mean_at_k20",
]

DEFAULT_INCLUDE_DATASET_ONE_HOT = True


def validate_state_fields(row: dict[str, Any]) -> None:
    missing = [field for field in STATE_FEATURES if field not in row]
    if missing:
        raise KeyError(f"Missing required state fields: {missing}")


def validate_round23_state_fields(row: dict[str, Any]) -> None:
    missing = [field for field in ROUND23_CONTROLLER_STATE_FEATURES if field not in row]
    if missing:
        raise KeyError(f"Missing required round23 state fields: {missing}")


def encode_feature_row(
    row: dict[str, Any],
    *,
    include_dataset_one_hot: bool = DEFAULT_INCLUDE_DATASET_ONE_HOT,
) -> tuple[list[str], list[float]]:
    validate_state_fields(row)
    feature_names = list(STATE_FEATURES)
    feature_values = [float(row[field]) for field in STATE_FEATURES]

    if include_dataset_one_hot:
        dataset_name = str(row["dataset_name"])
        for dataset in DATASET_ORDER:
            feature_names.append(f"dataset_is_{dataset}")
            feature_values.append(1.0 if dataset_name == dataset else 0.0)
    return feature_names, feature_values


def extract_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    include_dataset_one_hot: bool = DEFAULT_INCLUDE_DATASET_ONE_HOT,
) -> tuple[list[str], list[list[float]]]:
    if not rows:
        return [], []
    feature_names, _ = encode_feature_row(rows[0], include_dataset_one_hot=include_dataset_one_hot)
    matrix = [
        encode_feature_row(row, include_dataset_one_hot=include_dataset_one_hot)[1]
        for row in rows
    ]
    return feature_names, matrix


def encode_feature_row_with_fields(
    row: dict[str, Any],
    *,
    state_fields: list[str],
    include_dataset_one_hot: bool = DEFAULT_INCLUDE_DATASET_ONE_HOT,
) -> tuple[list[str], list[float]]:
    missing = [field for field in state_fields if field not in row]
    if missing:
        raise KeyError(f"Missing required feature fields: {missing}")
    feature_names = list(state_fields)
    feature_values = [float(row[field]) for field in state_fields]

    if include_dataset_one_hot:
        dataset_name = str(row["dataset_name"])
        for dataset in DATASET_ORDER:
            feature_names.append(f"dataset_is_{dataset}")
            feature_values.append(1.0 if dataset_name == dataset else 0.0)
    return feature_names, feature_values


def extract_feature_matrix_with_fields(
    rows: list[dict[str, Any]],
    *,
    state_fields: list[str],
    include_dataset_one_hot: bool = DEFAULT_INCLUDE_DATASET_ONE_HOT,
) -> tuple[list[str], list[list[float]]]:
    if not rows:
        return [], []
    feature_names, _ = encode_feature_row_with_fields(
        rows[0],
        state_fields=state_fields,
        include_dataset_one_hot=include_dataset_one_hot,
    )
    matrix = [
        encode_feature_row_with_fields(
            row,
            state_fields=state_fields,
            include_dataset_one_hot=include_dataset_one_hot,
        )[1]
        for row in rows
    ]
    return feature_names, matrix
