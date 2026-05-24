from __future__ import annotations


FORMAL_MODEL_FAMILIES = (
    "lightgbm",
    "xgboost",
    "catboost",
    "randomforest",
    "extratrees",
    "mlp",
    "linear_baseline",
)

EXTENDED_MODEL_FAMILIES = (
    "gradientboosting",
    "histgradientboosting",
    "adaboost",
    "svr",
    "knn",
    "elasticnet",
)

SUPPORTED_MODEL_FAMILIES = FORMAL_MODEL_FAMILIES + EXTENDED_MODEL_FAMILIES
