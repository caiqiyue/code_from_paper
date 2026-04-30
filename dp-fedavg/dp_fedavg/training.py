from __future__ import annotations


def compute_local_update(client_partition, *, scale: float = 1.0) -> dict[str, list[float]]:
    sample_count = max(1, len(client_partition.samples))
    mean_length = sum(len(sample.render_text().split()) for sample in client_partition.samples) / sample_count
    lexical_diversity = len(
        {
            token.lower()
            for sample in client_partition.samples
            for token in sample.render_text().split()
        }
    )
    return {
        "client_signal": [
            float(sample_count) * scale,
            float(mean_length) * scale,
            float(lexical_diversity) * scale,
        ]
    }
