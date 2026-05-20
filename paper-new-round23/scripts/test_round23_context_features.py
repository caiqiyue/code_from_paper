"""Targeted tests for round23_context_features.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from round23_context_features import (  # noqa: E402
    build_feature_vector,
    compute_selected_redundancy_mean,
    extract_context_features,
    validate_feature_schema,
)


def test_compute_selected_redundancy_mean_nonzero_for_similar_vectors():
    value = compute_selected_redundancy_mean([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert value > 0.0


def test_extract_context_features_returns_nine_features():
    features = extract_context_features(
        private_lengths=[5, 7, 9, 11],
        private_vectors=[[1.0, 0.0], [0.8, 0.2]],
        selected_vectors_k20=[[1.0, 0.0], [0.7, 0.3]],
        support_mean_at_k20=0.75,
        genericity_mean_at_k20=0.12,
        redundancy_mean_at_k20=0.08,
    )
    assert len(features) == 9


def test_build_feature_vector_appends_dataset_onehot():
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "feature_schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "feature_names": [
                        "shape_score",
                        "private_mean_length",
                        "private_p75_length",
                        "private_length_iqr",
                        "support_mean_at_k20",
                        "coverage_mean_at_k20",
                        "coverage_p25_at_k20",
                        "genericity_mean_at_k20",
                        "redundancy_mean_at_k20",
                    ],
                    "include_dataset_onehot": True,
                    "onehot_order": ["jobs", "congressional", "forums", "microblog", "imdb", "openreview"],
                    "total_features": 15,
                }
            ),
            encoding="utf-8",
        )
        schema = validate_feature_schema(schema_path)
        vector = build_feature_vector(
            context_features=[0.0] * 9,
            dataset_name="jobs",
            schema=schema,
        )
        assert len(vector) == 15
        assert vector[-6:] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_build_feature_vector_follows_schema_feature_order_and_onehot_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "feature_schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "feature_version": "round23_with_dataset_v2",
                    "feature_names": [
                        "coverage_mean_at_k20",
                        "shape_score",
                        "support_mean_at_k20",
                    ],
                    "include_dataset_onehot": True,
                    "onehot_order": ["microblog", "jobs"],
                    "total_features": 5,
                }
            ),
            encoding="utf-8",
        )
        schema = validate_feature_schema(schema_path)
        vector = build_feature_vector(
            context_features=[1.5, 11.0, 22.0, 33.0, 2.5, 4.5, 5.5, 6.5, 7.5],
            dataset_name="microblog",
            schema=schema,
        )
        assert vector == [4.5, 1.5, 2.5, 1.0, 0.0]


def test_build_feature_vector_rejects_unknown_dataset_for_schema_onehot_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "feature_schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "feature_version": "round23_with_dataset_v2",
                    "feature_names": ["shape_score"] * 9,
                    "include_dataset_onehot": True,
                    "onehot_order": ["jobs", "congressional", "forums", "microblog"],
                    "total_features": 13,
                }
            ),
            encoding="utf-8",
        )
        schema = validate_feature_schema(schema_path)
        try:
            build_feature_vector(
                context_features=[0.0] * 9,
                dataset_name="imdb",
                schema=schema,
            )
        except ValueError as exc:
            assert "dataset_name" in str(exc)
        else:
            raise AssertionError("Expected ValueError for dataset outside bundle onehot_order")


if __name__ == "__main__":
    tests = [
        ("redundancy_mean", test_compute_selected_redundancy_mean_nonzero_for_similar_vectors),
        ("extract_context_features", test_extract_context_features_returns_nine_features),
        ("build_feature_vector", test_build_feature_vector_appends_dataset_onehot),
        ("schema_order", test_build_feature_vector_follows_schema_feature_order_and_onehot_order),
        ("unknown_dataset", test_build_feature_vector_rejects_unknown_dataset_for_schema_onehot_order),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as exc:
            failures += 1
            print(f"  {name}: FAILED - {exc}")
    if failures:
        raise SystemExit(1)
    print("\nALL ROUND23 CONTEXT FEATURE TESTS PASSED")
