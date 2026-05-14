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
                    "feature_names": ["f"] * 9,
                    "include_dataset_onehot": True,
                    "total_features": 13,
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
        assert len(vector) == 13
        assert vector[-4:] == [1.0, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    tests = [
        ("redundancy_mean", test_compute_selected_redundancy_mean_nonzero_for_similar_vectors),
        ("extract_context_features", test_extract_context_features_returns_nine_features),
        ("build_feature_vector", test_build_feature_vector_appends_dataset_onehot),
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
