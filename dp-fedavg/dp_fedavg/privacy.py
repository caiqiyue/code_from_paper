from __future__ import annotations

import math
import random


def _flatten(update: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for tensor in update.values():
        values.extend(float(item) for item in tensor)
    return values


def clip_update(update: dict[str, list[float]], *, clip_norm: float) -> dict[str, list[float]]:
    flat = _flatten(update)
    total_norm = math.sqrt(sum(value * value for value in flat))
    if total_norm == 0.0 or total_norm <= clip_norm:
        return {name: list(values) for name, values in update.items()}
    scale = clip_norm / total_norm
    return {
        name: [float(value) * scale for value in values]
        for name, values in update.items()
    }


def compute_noise_scale(*, clip_norm: float, noise_multiplier: float) -> float:
    return float(clip_norm) * float(noise_multiplier)


def add_gaussian_noise(
    update: dict[str, list[float]],
    *,
    noise_scale: float,
    seed: int,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    return {
        name: [float(value) + rng.gauss(0.0, noise_scale) for value in values]
        for name, values in update.items()
    }


def estimate_privacy_epsilon(
    *,
    rounds: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
) -> float:
    if noise_multiplier <= 0:
        return float("inf")
    safe_delta = max(float(delta), 1e-12)
    return float(rounds) * float(sample_rate) * math.sqrt(2.0 * math.log(1.25 / safe_delta)) / float(noise_multiplier)
