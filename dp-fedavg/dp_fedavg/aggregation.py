from __future__ import annotations


def fixed_denominator_average(
    updates: list[dict[str, list[float]]],
    *,
    expected_clients: int,
) -> dict[str, list[float]]:
    if expected_clients <= 0:
        raise ValueError("expected_clients must be positive.")
    if not updates:
        return {}
    keys = list(updates[0].keys())
    averaged: dict[str, list[float]] = {}
    for key in keys:
        width = len(updates[0][key])
        totals = [0.0] * width
        for update in updates:
            for index, value in enumerate(update[key]):
                totals[index] += float(value)
        averaged[key] = [total / float(expected_clients) for total in totals]
    return averaged
