from __future__ import annotations

from round23_feature_sets import FORMAL_FEATURE_VERSIONS
from round23_model_zoo import FORMAL_MODEL_FAMILIES


def build_experiment_matrix(
    *,
    model_families: list[str] | None = None,
    feature_versions: list[str] | None = None,
) -> list[dict[str, str]]:
    families = list(model_families or FORMAL_MODEL_FAMILIES)
    features = list(feature_versions or FORMAL_FEATURE_VERSIONS)
    return [
        {
            "model_family": family,
            "feature_version": feature_version,
        }
        for family in families
        for feature_version in features
    ]
