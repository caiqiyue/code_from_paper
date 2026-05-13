#!/usr/bin/env python3
"""Shared context feature extraction for round22 learned budget policy runtime.

This module provides the 8 core state features used for training and inference:
  shape_score, private_mean_length, private_p75_length, private_length_iqr,
  support_mean_at_k20, coverage_mean_at_k20, coverage_p25_at_k20, genericity_mean_at_k20

It replicates the exact computation logic from round22_bandit_collection_runner.py
to ensure training and runtime feature consistency.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any


DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
SHAPE_TAIL_THRESHOLD = 300
SHAPE_SHORT_THRESHOLD = 100

# Router cfg template matching the ACTUAL structure used by round22_bandit_collection_runner.py.
# Must contain screening_reference with mean/std for z-score normalization.
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


def _percentile_nearest_rank(values: list[int], percentile: float) -> float:
    """Nearest-rank percentile. Matches the actual implementation in round22_bandit_collection_runner.py."""
    if not values:
        return 0.0
    sorted_values = sorted(int(value) for value in values)
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
    """Returns (coverage_mean, coverage_p25) using nearest-rank percentile.

    This matches the implementation in append_round22_bandit_summary.py which is
    used during training data collection.
    """
    if not private_vectors or not selected_vectors:
        return 0.0, 0.0
    coverage_values = [
        max(cosine_similarity(private_vector, selected_vector) for selected_vector in selected_vectors)
        for private_vector in private_vectors
    ]
    return _mean(coverage_values), _percentile_nearest_rank(coverage_values, 25)


def compute_shape_descriptor(
    private_lengths: list[int],
    *,
    tail_threshold: int,
    short_threshold: int,
) -> dict[str, float]:
    """Compute shape descriptor using nearest-rank percentile (matches actual code)."""
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


def compute_shape_score(
    descriptor: dict[str, float],
    router_cfg: dict[str, Any],
) -> float:
    """Compute shape score using screening_reference z-score normalization.

    router_cfg must have a 'screening_reference' key with mean/std for each metric.
    Example router_cfg structure:
        {
            "screening_reference": {
                "median_len": {"mean": 300.0, "std": 100.0},
                "p75_len": {"mean": 360.0, "std": 120.0},
                "iqr_len": {"mean": 200.0, "std": 80.0},
            }
        }
    """
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
    dataset_name: str,
    private_lengths: list[int],
    private_vectors: list[list[float]],
    selected_vectors_k20: list[list[float]],
    router_cfg: dict[str, Any] | None = None,
) -> list[float]:
    """Extract the 8 core features for a given dataset context.

    Returns a feature vector ordered as defined in feature_schema.json.
    Call this after getting k=20 reference selected_vectors.
    """
    if router_cfg is None:
        router_cfg = ROUTER_CFG_TEMPLATE

    descriptor = compute_shape_descriptor(
        private_lengths,
        tail_threshold=SHAPE_TAIL_THRESHOLD,
        short_threshold=SHAPE_SHORT_THRESHOLD,
    )
    shape_score = compute_shape_score(descriptor, router_cfg)

    mean_length = float(sum(private_lengths) / len(private_lengths)) if private_lengths else 0.0
    sorted_lengths = sorted(private_lengths)
    n = len(sorted_lengths)
    p75_length = float(_percentile_nearest_rank(private_lengths, 75)) if private_lengths else 0.0
    q1 = float(_percentile_nearest_rank(private_lengths, 25)) if private_lengths else 0.0
    q3 = float(_percentile_nearest_rank(private_lengths, 75)) if private_lengths else 0.0
    length_iqr = q3 - q1

    support_vals = []
    for priv_vec in private_vectors:
        max_sim = max((cosine_similarity(priv_vec, sel) for sel in selected_vectors_k20), default=0.0)
        support_vals.append(max_sim)
    support_mean = _mean(support_vals)

    coverage_mean, coverage_p25 = compute_coverage_metrics(private_vectors, selected_vectors_k20)

    genericity_vals = [1.0 - s for s in support_vals]
    genericity_mean = _mean(genericity_vals)

    features = [
        shape_score,
        mean_length,
        p75_length,
        length_iqr,
        support_mean,
        coverage_mean,
        coverage_p25,
        genericity_mean,
    ]
    return features


def append_dataset_onehot(features: list[float], dataset_name: str) -> list[float]:
    """Append 4-dimensional one-hot for dataset_name."""
    onehot = [1.0 if d == dataset_name else 0.0 for d in DATASET_ORDER]
    return features + onehot


def validate_feature_schema(schema_path: str | Path) -> dict[str, Any]:
    """Load and validate feature_schema.json. Returns schema dict."""
    import json
    schema = json.loads(Path(schema_path).read_text())
    required_fields = ["version", "feature_names", "include_dataset_onehot", "total_features"]
    for field in required_fields:
        if field not in schema:
            raise ValueError(f"feature_schema.json missing required field: {field}")
    return schema


def build_feature_vector(
    *,
    context_features: list[float],
    dataset_name: str,
    schema: dict[str, Any],
) -> list[float]:
    """Build the full feature vector including optional one-hot.

    Raises ValueError if feature vector length doesn't match schema.total_features.
    """
    if schema.get("include_dataset_onehot", False):
        vector = append_dataset_onehot(context_features, dataset_name)
    else:
        vector = context_features
    expected_len = schema.get("total_features", len(vector))
    if len(vector) != expected_len:
        raise ValueError(
            f"Feature vector length {len(vector)} != total_features {expected_len} "
            f"(check dataset_name one-hot encoding)"
        )
    return vector
