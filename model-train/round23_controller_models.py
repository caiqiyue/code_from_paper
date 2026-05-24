from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from round23_model_zoo import SUPPORTED_MODEL_FAMILIES


def require_model_family(family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {family}")
    return normalized


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lightgbm is not installed") from exc
    return lgb


def _require_xgboost() -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost is not installed") from exc
    return xgb


def _require_catboost() -> Any:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("catboost is not installed") from exc
    return CatBoostRegressor


def _require_sklearn_regressor(name: str) -> Any:
    if name == "randomforest":
        from sklearn.ensemble import RandomForestRegressor  # type: ignore

        return RandomForestRegressor
    if name == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore

        return ExtraTreesRegressor
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor  # type: ignore

        return MLPRegressor
    if name == "linear_baseline":
        from sklearn.linear_model import Ridge  # type: ignore

        return Ridge
    if name == "gradientboosting":
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

        return GradientBoostingRegressor
    if name == "histgradientboosting":
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

        return HistGradientBoostingRegressor
    if name == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor  # type: ignore

        return AdaBoostRegressor
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet  # type: ignore

        return ElasticNet
    raise ValueError(f"Unsupported sklearn regressor name: {name}")


def _build_scaled_sklearn_regressor(name: str, params: dict[str, Any]) -> Any:
    from sklearn.neighbors import KNeighborsRegressor  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.svm import SVR  # type: ignore

    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(**params))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    raise ValueError(f"Unsupported scaled sklearn regressor name: {name}")


def build_regressor(*, family: str, params: dict[str, Any]) -> Any:
    normalized = require_model_family(family)
    if normalized == "lightgbm":
        lgb = _require_lightgbm()
        return lgb.LGBMRegressor(**params)
    if normalized == "xgboost":
        xgb = _require_xgboost()
        return xgb.XGBRegressor(**params)
    if normalized == "catboost":
        estimator_cls = _require_catboost()
        return estimator_cls(**params)
    if normalized in {"svr", "knn"}:
        return _build_scaled_sklearn_regressor(normalized, params)
    estimator_cls = _require_sklearn_regressor(normalized)
    return estimator_cls(**params)


def predict_regressor(*, family: str, model: Any, feature_matrix: list[list[float]]) -> list[float]:
    require_model_family(family)
    if not feature_matrix:
        return []
    preds = model.predict(feature_matrix)
    if hasattr(preds, "tolist"):
        return [float(value) for value in preds.tolist()]
    return [float(value) for value in preds]


def model_file_extension(family: str) -> str:
    normalized = require_model_family(family)
    if normalized == "lightgbm":
        return ".txt"
    if normalized == "xgboost":
        return ".json"
    if normalized == "catboost":
        return ".cbm"
    return ".pkl"


def save_regressor(*, family: str, model: Any, path: str | Path) -> None:
    normalized = require_model_family(family)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if normalized == "lightgbm":
        model.booster_.save_model(str(resolved))
        return
    if normalized == "xgboost":
        model.save_model(str(resolved))
        return
    if normalized == "catboost":
        model.save_model(str(resolved))
        return
    with resolved.open("wb") as handle:
        pickle.dump(model, handle)


def load_regressor(*, family: str, path: str | Path) -> Any:
    normalized = require_model_family(family)
    resolved = Path(path)
    if normalized == "lightgbm":
        lgb = _require_lightgbm()
        return lgb.Booster(model_file=str(resolved))
    if normalized == "xgboost":
        xgb = _require_xgboost()
        model = xgb.XGBRegressor()
        model.load_model(str(resolved))
        return model
    if normalized == "catboost":
        estimator_cls = _require_catboost()
        model = estimator_cls()
        model.load_model(str(resolved))
        return model
    with resolved.open("rb") as handle:
        return pickle.load(handle)
