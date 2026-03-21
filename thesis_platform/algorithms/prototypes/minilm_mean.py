from __future__ import annotations

from thesis_platform.algorithms.math_utils import mean_vector, normalize
from thesis_platform.core.schemas import PrototypeFeedback, Sample


def extract_minilm_mean_prototype(
    *,
    client_id: str,
    round_id: int,
    samples: list[Sample],
    embedder,
    weight: float,
) -> PrototypeFeedback:
    """Build one normalized client prototype from retrieved real samples."""

    texts = [sample.rendered_text() for sample in samples if sample.rendered_text().strip()]
    if texts:
        prototype_vector = normalize(mean_vector(embedder.embed_texts(texts)))
    else:
        prototype_vector = []
    return PrototypeFeedback(
        client_id=client_id,
        round_id=round_id,
        prototype_vector=prototype_vector,
        weight=float(weight),
        source_sample_ids=[sample.sample_id for sample in samples],
        meta={"sample_count": len(samples), "prototype_name": "minilm_mean"},
    )
