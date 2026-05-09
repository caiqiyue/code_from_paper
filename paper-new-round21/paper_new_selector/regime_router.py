from __future__ import annotations

from dataclasses import dataclass

from .shape_descriptor import ShapeDescriptor


@dataclass(frozen=True)
class RegimeDecision:
    regime: str
    shape_score: float


def _zscore(value: float, mean: float, std: float) -> float:
    if abs(float(std)) <= 1e-8:
        return 0.0
    return float((float(value) - float(mean)) / float(std))


def compute_shape_score(descriptor: ShapeDescriptor, router_cfg: dict) -> float:
    reference = dict(router_cfg.get("screening_reference", {}))
    median_stats = dict(reference.get("median_len", {"mean": 0.0, "std": 1.0}))
    p75_stats = dict(reference.get("p75_len", {"mean": 0.0, "std": 1.0}))
    iqr_stats = dict(reference.get("iqr_len", {"mean": 0.0, "std": 1.0}))
    return (
        _zscore(descriptor.median_len, median_stats["mean"], median_stats["std"])
        + _zscore(descriptor.p75_len, p75_stats["mean"], p75_stats["std"])
        + _zscore(descriptor.iqr_len, iqr_stats["mean"], iqr_stats["std"])
        + float(descriptor.tail_ratio)
        - float(descriptor.short_ratio)
    )


def route_budget_regime(descriptor: ShapeDescriptor, router_cfg: dict) -> RegimeDecision:
    score = compute_shape_score(descriptor, router_cfg)
    tau_center = float(router_cfg.get("tau_center", 0.0))
    delta_router = float(router_cfg.get("delta_router", 0.35))
    if score >= tau_center + delta_router:
        return RegimeDecision(regime="broad_tail", shape_score=score)
    if score <= tau_center - delta_router:
        return RegimeDecision(regime="compact_structured", shape_score=score)
    return RegimeDecision(regime="uncertain", shape_score=score)
