from __future__ import annotations

from dataclasses import dataclass

from features import ROUND23_CONTROLLER_STATE_FEATURES
from round23_dataset_registry import get_formal_onehot_order


FORMAL_FEATURE_VERSIONS = ("with_dataset", "no_dataset")


@dataclass(frozen=True)
class Round23FeatureSpec:
    feature_version: str
    feature_fields: list[str]
    include_dataset_onehot: bool
    onehot_order: list[str]

    @property
    def total_features(self) -> int:
        return len(self.feature_fields) + (len(self.onehot_order) if self.include_dataset_onehot else 0)


def get_feature_spec(feature_version: str) -> Round23FeatureSpec:
    normalized = str(feature_version).strip().lower()
    if normalized == "with_dataset":
        return Round23FeatureSpec(
            feature_version="with_dataset",
            feature_fields=list(ROUND23_CONTROLLER_STATE_FEATURES),
            include_dataset_onehot=True,
            onehot_order=get_formal_onehot_order(),
        )
    if normalized == "no_dataset":
        return Round23FeatureSpec(
            feature_version="no_dataset",
            feature_fields=list(ROUND23_CONTROLLER_STATE_FEATURES),
            include_dataset_onehot=False,
            onehot_order=[],
        )
    raise ValueError(f"Unsupported round23 feature_version: {feature_version}")


def infer_feature_version(
    *,
    feature_fields: list[str],
    include_dataset_onehot: bool,
    onehot_order: list[str],
) -> str:
    canonical_fields = list(ROUND23_CONTROLLER_STATE_FEATURES)
    if list(feature_fields) != canonical_fields:
        return "custom"
    if include_dataset_onehot and list(onehot_order) == get_formal_onehot_order():
        return "with_dataset"
    if (not include_dataset_onehot) and not onehot_order:
        return "no_dataset"
    return "custom"


def resolve_feature_spec(
    *,
    feature_version: str | None = None,
    feature_fields: list[str] | None = None,
    include_dataset_onehot: bool | None = None,
    onehot_order: list[str] | None = None,
) -> Round23FeatureSpec:
    if feature_version:
        return get_feature_spec(feature_version)
    resolved_fields = list(feature_fields or ROUND23_CONTROLLER_STATE_FEATURES)
    resolved_include = bool(include_dataset_onehot) if include_dataset_onehot is not None else True
    resolved_order = list(onehot_order or (get_formal_onehot_order() if resolved_include else []))
    inferred_version = infer_feature_version(
        feature_fields=resolved_fields,
        include_dataset_onehot=resolved_include,
        onehot_order=resolved_order,
    )
    return Round23FeatureSpec(
        feature_version=inferred_version,
        feature_fields=resolved_fields,
        include_dataset_onehot=resolved_include,
        onehot_order=resolved_order if resolved_include else [],
    )
