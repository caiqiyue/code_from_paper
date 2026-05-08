from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics


@dataclass(frozen=True)
class ShapeDescriptor:
    median_len: float
    p75_len: float
    tail_ratio: float
    short_ratio: float
    iqr_len: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _percentile_nearest_rank(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(int(value) for value in values)
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = math.ceil((float(percentile) / 100.0) * len(sorted_values))
    return float(sorted_values[max(0, rank - 1)])


def compute_shape_descriptor(
    private_lengths: list[int],
    *,
    tail_threshold: int,
    short_threshold: int,
) -> ShapeDescriptor:
    if not private_lengths:
        return ShapeDescriptor(0.0, 0.0, 0.0, 0.0, 0.0)
    q1 = _percentile_nearest_rank(private_lengths, 25)
    q3 = _percentile_nearest_rank(private_lengths, 75)
    total = float(len(private_lengths))
    return ShapeDescriptor(
        median_len=float(statistics.median(private_lengths)),
        p75_len=float(q3),
        tail_ratio=float(sum(length >= int(tail_threshold) for length in private_lengths) / total),
        short_ratio=float(sum(length <= int(short_threshold) for length in private_lengths) / total),
        iqr_len=float(q3 - q1),
    )
