#!/usr/bin/env python3
"""Shared context feature extraction for round23 dynamic controller runtime."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


DATASET_ORDER = ["jobs", "congressional", "forums", "microblog", "imdb", "openreview"]
CONTEXT_FEATURE_NAME_ORDER = [
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
SHAPE_TAIL_THRESHOLD = 300
SHAPE_SHORT_THRESHOLD = 100

ROUTER_CFG_TEMPLATE = {
    "screening_reference": {
        "median_len": {"mean": 300.0, "std": 100.0},
        "p75_len": {"mean": 360.0, "std": 120.0},
        "iqr_len": {"mean": 200.0, "std": 80.0},
    }
}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile_nearest_rank(values: list[int] | list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = int(math.ceil((float(percentile) / 100.0) * len(sorted_values)))
    return float(sorted_values[max(0, rank - 1)])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    numerator = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(x) * float(x) for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(numerator / (norm_a * norm_b))


def compute_coverage_metrics(
    private_vectors: list[list[float]],
    selected_vectors: list[list[float]],
) -> tuple[float, float]:
    if not private_vectors or not selected_vectors:
        return 0.0, 0.0
    coverage_values = [
        max(cosine_similarity(private_vector, selected_vector) for selected_vector in selected_vectors)
        for private_vector in private_vectors
    ]
    return _mean(coverage_values), _percentile_nearest_rank(coverage_values, 25)


def compute_selected_redundancy_mean(selected_vectors: list[list[float]]) -> float:
    if len(selected_vectors) < 2:
        return 0.0
    similarities: list[float] = []
    for index, anchor in enumerate(selected_vectors):
        others = selected_vectors[:index] + selected_vectors[index + 1 :]
        if not others:
            continue
        similarities.append(max(cosine_similarity(anchor, other) for other in others))
    return _mean(similarities)


def compute_shape_descriptor(
    private_lengths: list[int],
    *,
    tail_threshold: int,
    short_threshold: int,
) -> dict[str, float]:
    if not private_lengths:
        return {
            "median_len": 0.0,
            "p75_len": 0.0,
            "tail_ratio": 0.0,
            "short_ratio": 0.0,
            "iqr_len": 0.0,
        }
    q1 = _percentile_nearest_rank(private_lengths, 25)
    q3 = _percentile_nearest_rank(private_lengths, 75)
    total = float(len(private_lengths))
    return {
        "median_len": float(_percentile_nearest_rank(private_lengths, 50)),
        "p75_len": float(q3),
        "tail_ratio": float(sum(length >= int(tail_threshold) for length in private_lengths) / total),
        "short_ratio": float(sum(length <= int(short_threshold) for length in private_lengths) / total),
        "iqr_len": float(q3 - q1),
    }


def _zscore(value: float, mean: float, std: float) -> float:
    if abs(float(std)) <= 1e-8:
        return 0.0
    return float((float(value) - float(mean)) / float(std))


def compute_shape_score(descriptor: dict[str, float], router_cfg: dict[str, Any]) -> float:
    reference = dict(router_cfg.get("screening_reference", {}))
    median_stats = dict(reference.get("median_len", {"mean": 0.0, "std": 1.0}))
    p75_stats = dict(reference.get("p75_len", {"mean": 0.0, "std": 1.0}))
    iqr_stats = dict(reference.get("iqr_len", {"mean": 0.0, "std": 1.0}))
    return (
        _zscore(descriptor["median_len"], median_stats["mean"], median_stats["std"])
        + _zscore(descriptor["p75_len"], p75_stats["mean"], p75_stats["std"])
        + _zscore(descriptor["iqr_len"], iqr_stats["mean"], iqr_stats["std"])
        + float(descriptor["tail_ratio"])
        - float(descriptor["short_ratio"])
    )


def extract_context_features(
    *,
    private_lengths: list[int],
    private_vectors: list[list[float]],
    selected_vectors_k20: list[list[float]],
    support_mean_at_k20: float,
    genericity_mean_at_k20: float,
    redundancy_mean_at_k20: float | None = None,
    router_cfg: dict[str, Any] | None = None,
) -> list[float]:
    if router_cfg is None:
        router_cfg = ROUTER_CFG_TEMPLATE
    descriptor = compute_shape_descriptor(
        private_lengths,
        tail_threshold=SHAPE_TAIL_THRESHOLD,
        short_threshold=SHAPE_SHORT_THRESHOLD,
    )
    shape_score = compute_shape_score(descriptor, router_cfg)
    mean_length = float(sum(private_lengths) / len(private_lengths)) if private_lengths else 0.0
    p75_length = float(_percentile_nearest_rank(private_lengths, 75)) if private_lengths else 0.0
    q1 = float(_percentile_nearest_rank(private_lengths, 25)) if private_lengths else 0.0
    q3 = float(_percentile_nearest_rank(private_lengths, 75)) if private_lengths else 0.0
    coverage_mean_at_k20, coverage_p25_at_k20 = compute_coverage_metrics(
        private_vectors=private_vectors,
        selected_vectors=selected_vectors_k20,
    )
    if redundancy_mean_at_k20 is None:
        redundancy_mean_at_k20 = compute_selected_redundancy_mean(selected_vectors_k20)
    return [
        shape_score,
        mean_length,
        p75_length,
        q3 - q1,
        float(support_mean_at_k20),
        float(coverage_mean_at_k20),
        float(coverage_p25_at_k20),
        float(genericity_mean_at_k20),
        float(redundancy_mean_at_k20),
    ]


def append_dataset_onehot(features: list[float], dataset_name: str) -> list[float]:
    onehot = [1.0 if name == dataset_name else 0.0 for name in DATASET_ORDER]
    return features + onehot


def validate_feature_schema(schema_path: str | Path) -> dict[str, Any]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    feature_version = schema.get("feature_version", schema.get("version"))
    if not feature_version:
        raise ValueError("feature_schema.json missing required field: feature_version/version")
    if "feature_names" not in schema:
        raise ValueError("feature_schema.json missing required field: feature_names")
    if "include_dataset_onehot" not in schema:
        raise ValueError("feature_schema.json missing required field: include_dataset_onehot")
    if "total_features" not in schema:
        raise ValueError("feature_schema.json missing required field: total_features")

    feature_names = list(schema.get("feature_names", []))
    unknown_feature_names = [
        name for name in feature_names if name not in CONTEXT_FEATURE_NAME_ORDER
    ]
    if unknown_feature_names:
        raise ValueError(
            "feature_schema.json contains unsupported feature_names: "
            + ", ".join(unknown_feature_names)
        )

    include_dataset_onehot = bool(schema.get("include_dataset_onehot", False))
    onehot_order = list(schema.get("onehot_order", DATASET_ORDER if include_dataset_onehot else []))
    if include_dataset_onehot and not onehot_order:
        raise ValueError("feature_schema.json requires onehot_order when include_dataset_onehot=true")

    expected_total_features = len(feature_names) + (len(onehot_order) if include_dataset_onehot else 0)
    if int(schema["total_features"]) != expected_total_features:
        raise ValueError(
            "feature_schema total_features mismatch: "
            f"expected {expected_total_features}, got {schema['total_features']}"
        )

    schema["feature_version"] = str(feature_version)
    schema["version"] = str(schema.get("version", feature_version))
    schema["feature_names"] = feature_names
    schema["onehot_order"] = onehot_order
    schema["include_dataset_onehot"] = include_dataset_onehot
    schema["total_features"] = int(schema["total_features"])
    return schema


def _coerce_context_feature_mapping(
    context_features: list[float] | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(context_features, Mapping):
        return {
            name: float(context_features[name])
            for name in CONTEXT_FEATURE_NAME_ORDER
            if name in context_features
        }
    if len(context_features) != len(CONTEXT_FEATURE_NAME_ORDER):
        raise ValueError(
            "context_features length "
            f"{len(context_features)} != expected {len(CONTEXT_FEATURE_NAME_ORDER)}"
        )
    return {
        name: float(value)
        for name, value in zip(CONTEXT_FEATURE_NAME_ORDER, context_features)
    }


def build_feature_vector(
    *,
    context_features: list[float] | Mapping[str, float],
    dataset_name: str,
    schema: dict[str, Any],
) -> list[float]:
    feature_mapping = _coerce_context_feature_mapping(context_features)
    missing_feature_names = [
        name for name in schema.get("feature_names", []) if name not in feature_mapping
    ]
    if missing_feature_names:
        raise ValueError(
            "context_features missing values for schema feature_names: "
            + ", ".join(missing_feature_names)
        )

    vector = [float(feature_mapping[name]) for name in schema.get("feature_names", [])]
    if schema.get("include_dataset_onehot", False):
        onehot_order = list(schema.get("onehot_order", []))
        if dataset_name not in onehot_order:
            raise ValueError(
                f"dataset_name '{dataset_name}' not present in bundle onehot_order {onehot_order}"
            )
        vector.extend(1.0 if name == dataset_name else 0.0 for name in onehot_order)
    expected_len = int(schema.get("total_features", len(vector)))
    if len(vector) != expected_len:
        raise ValueError(
            f"Feature vector length {len(vector)} != total_features {expected_len}"
        )
    return vector
