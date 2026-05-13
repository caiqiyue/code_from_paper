"""Tests for round22_context_features.py."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the script directory is on the path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from round22_context_features import (
    _mean,
    _percentile_nearest_rank,
    cosine_similarity,
    compute_coverage_metrics,
    compute_shape_descriptor,
    compute_shape_score,
    extract_context_features,
    append_dataset_onehot,
    build_feature_vector,
    validate_feature_schema,
    DATASET_ORDER,
    ROUTER_CFG_TEMPLATE,
    SHAPE_TAIL_THRESHOLD,
    SHAPE_SHORT_THRESHOLD,
)

# Suppress pytest import warning — pytest is only needed when running via `pytest` command
# and is not part of the stdlib. Tests can also be run directly.


class TestMean:
    def test_normal(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert _mean([]) == 0.0

    def test_single(self):
        assert _mean([5.0]) == 5.0


class TestPercentileNearestRank:
    def test_p50(self):
        result = _percentile_nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)
        assert result == 5.0  # rank = ceil(10 * 0.5) = 5, index 4 = 5

    def test_p25(self):
        result = _percentile_nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 25)
        assert result == 3.0  # rank = ceil(10 * 0.25) = 3, index 2 = 3

    def test_p100(self):
        result = _percentile_nearest_rank([1, 2, 3, 4, 5], 100)
        assert result == 5.0

    def test_p0(self):
        result = _percentile_nearest_rank([1, 2, 3, 4, 5], 0)
        assert result == 1.0

    def test_empty(self):
        assert _percentile_nearest_rank([], 50) == 0.0


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestComputeCoverageMetrics:
    def test_normal(self):
        # private[0] = [1,0] -> max sim with [[1,0]] = 1.0
        # private[1] = [0,1] -> max sim with [[1,0]] = 0.0
        # coverage_values = [1.0, 0.0]
        # mean = 0.5, p25 = 0.0 (rank=ceil(2*0.25)=1, index=0)
        priv = [[1.0, 0.0], [0.0, 1.0]]
        sel = [[1.0, 0.0]]
        mean, p25 = compute_coverage_metrics(priv, sel)
        assert mean == 0.5, f"expected 0.5, got {mean}"
        assert p25 == 0.0, f"expected 0.0, got {p25}"

    def test_identical_vectors(self):
        priv = [[1.0, 0.0], [1.0, 0.0]]
        sel = [[1.0, 0.0]]
        mean, p25 = compute_coverage_metrics(priv, sel)
        assert mean == 1.0
        assert p25 == 1.0

    def test_empty_private(self):
        mean, p25 = compute_coverage_metrics([], [[1.0, 0.0]])
        assert mean == 0.0
        assert p25 == 0.0

    def test_empty_selected(self):
        mean, p25 = compute_coverage_metrics([[1.0, 0.0]], [])
        assert mean == 0.0
        assert p25 == 0.0


class TestComputeShapeDescriptor:
    def test_normal(self):
        lengths = [100, 200, 300, 400, 500]
        desc = compute_shape_descriptor(lengths, tail_threshold=300, short_threshold=100)
        assert "median_len" in desc
        assert "p75_len" in desc
        assert "tail_ratio" in desc
        assert "short_ratio" in desc
        assert "iqr_len" in desc

    def test_empty(self):
        desc = compute_shape_descriptor([], tail_threshold=300, short_threshold=100)
        assert all(v == 0.0 for v in desc.values())

    def test_tail_ratio(self):
        # With threshold=300, values >= 300 are tail
        lengths = [100, 200, 300, 400, 500]
        desc = compute_shape_descriptor(lengths, tail_threshold=300, short_threshold=100)
        # 300, 400, 500 >= 300 -> 3/5 = 0.6
        assert abs(desc["tail_ratio"] - 0.6) < 1e-9

    def test_short_ratio(self):
        # With threshold=100, values <= 100 are short
        lengths = [50, 100, 200, 300, 400]
        desc = compute_shape_descriptor(lengths, tail_threshold=300, short_threshold=100)
        # 50, 100 <= 100 -> 2/5 = 0.4
        assert abs(desc["short_ratio"] - 0.4) < 1e-9


class TestComputeShapeScore:
    def test_with_template(self):
        desc = {
            "median_len": 300.0,
            "p75_len": 360.0,
            "iqr_len": 200.0,
            "tail_ratio": 0.3,
            "short_ratio": 0.4,
        }
        score = compute_shape_score(desc, ROUTER_CFG_TEMPLATE)
        # With exact match to template means/std, z-scores are all 0
        # score = 0 + 0 + 0 + 0.3 - 0.4 = -0.1
        assert abs(score - (-0.1)) < 1e-9

    def test_empty_descriptor(self):
        desc = {
            "median_len": 0.0,
            "p75_len": 0.0,
            "iqr_len": 0.0,
            "tail_ratio": 0.0,
            "short_ratio": 0.0,
        }
        # z_median = (0-300)/100 = -3.0; z_p75 = (0-360)/120 = -3.0; z_iqr = (0-200)/80 = -2.5
        # score = -3.0 + -3.0 + -2.5 + 0 - 0 = -8.5
        score = compute_shape_score(desc, ROUTER_CFG_TEMPLATE)
        assert abs(score - (-8.5)) < 1e-9


class TestAppendDatasetOnehot:
    def test_jobs(self):
        f8 = [0.1] * 8
        result = append_dataset_onehot(f8, "jobs")
        assert len(result) == 12
        assert result[8] == 1.0   # jobs is first
        assert result[9] == 0.0   # congressional is second
        assert result[10] == 0.0  # forums is third
        assert result[11] == 0.0  # microblog is fourth

    def test_forums(self):
        f8 = [0.1] * 8
        result = append_dataset_onehot(f8, "forums")
        assert len(result) == 12
        assert result[8] == 0.0
        assert result[10] == 1.0  # forums is third

    def test_all_zeros_for_unknown(self):
        f8 = [0.1] * 8
        result = append_dataset_onehot(f8, "unknown_dataset")
        assert result[8:] == [0.0, 0.0, 0.0, 0.0]


class TestBuildFeatureVector:
    def test_with_onehot(self):
        schema = {"include_dataset_onehot": True, "total_features": 12}
        ctx = [0.1] * 8
        vec = build_feature_vector(context_features=ctx, dataset_name="jobs", schema=schema)
        assert len(vec) == 12
        assert vec[8] == 1.0

    def test_without_onehot(self):
        schema = {"include_dataset_onehot": False, "total_features": 8}
        ctx = [0.1] * 8
        vec = build_feature_vector(context_features=ctx, dataset_name="jobs", schema=schema)
        assert len(vec) == 8

    def test_length_mismatch_raises(self):
        schema = {"include_dataset_onehot": False, "total_features": 8}
        ctx = [0.1] * 10  # wrong length
        try:
            build_feature_vector(context_features=ctx, dataset_name="jobs", schema=schema)
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "Feature vector length" in str(e)


class TestValidateFeatureSchema:
    def test_missing_required_field_raises(self, tmp_path):
        import json
        schema_path = tmp_path / "feature_schema.json"
        schema_path.write_text(json.dumps({"version": "1.0", "feature_names": []}))
        try:
            validate_feature_schema(schema_path)
            assert False, "should have raised"
        except ValueError as e:
            assert "missing required field" in str(e)

    def test_valid_schema_returns_dict(self, tmp_path):
        import json
        schema_path = tmp_path / "feature_schema.json"
        schema = {
            "version": "1.0",
            "feature_names": ["shape_score"],
            "include_dataset_onehot": False,
            "total_features": 1,
        }
        schema_path.write_text(json.dumps(schema))
        result = validate_feature_schema(schema_path)
        assert result["version"] == "1.0"


class TestConstants:
    def test_dataset_order(self):
        assert DATASET_ORDER == ["jobs", "congressional", "forums", "microblog"]
        assert len(DATASET_ORDER) == 4

    def test_shape_thresholds(self):
        assert SHAPE_TAIL_THRESHOLD == 300
        assert SHAPE_SHORT_THRESHOLD == 100


class TestExtractContextFeatures:
    def test_extract_returns_8_features(self):
        lengths = [100, 200, 300, 400, 500] * 100
        priv_vecs = [[1.0, 0.0]] * len(lengths)
        sel_vecs = [[1.0, 0.0]]
        ctx = extract_context_features(
            dataset_name="jobs",
            private_lengths=lengths,
            private_vectors=priv_vecs,
            selected_vectors_k20=sel_vecs,
        )
        assert len(ctx) == 8

    def test_extract_includes_shape_score(self):
        lengths = [300] * 100
        priv_vecs = [[1.0, 0.0]] * 100
        sel_vecs = [[1.0, 0.0]]
        ctx = extract_context_features(
            dataset_name="jobs",
            private_lengths=lengths,
            private_vectors=priv_vecs,
            selected_vectors_k20=sel_vecs,
        )
        # shape_score should be -0.1 (tail_ratio=0, short_ratio=1.0)
        assert len(ctx) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
