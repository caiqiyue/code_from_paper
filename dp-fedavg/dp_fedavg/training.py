from __future__ import annotations


def compute_local_update(
    client_partition,
    *,
    server_state: dict[str, list[float]] | None = None,
    scale: float = 1.0,
) -> dict[str, list[float]]:
    sample_count = max(1, len(client_partition.samples))
    mean_length = sum(len(sample.render_text().split()) for sample in client_partition.samples) / sample_count
    lexical_diversity = len(
        {
            token.lower()
            for sample in client_partition.samples
            for token in sample.render_text().split()
        }
    )
    base = {
        "client_signal": [
            float(sample_count) * scale,
            float(mean_length) * scale,
            float(lexical_diversity) * scale,
        ]
    }
    if not server_state:
        return base
    previous = list(server_state.get("client_signal", [0.0, 0.0, 0.0]))
    return {
        "client_signal": [
            base["client_signal"][index] - float(previous[index])
            for index in range(len(base["client_signal"]))
        ]
    }
