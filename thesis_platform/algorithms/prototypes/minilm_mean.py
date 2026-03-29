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
    privatizer=None,
) -> PrototypeFeedback:
    """Build one normalized client prototype from retrieved real samples.

    Args:
        client_id: Client identifier
        round_id: Current federation round
        samples: Real samples used to extract the prototype
        embedder: Text embedding backend
        weight: Prototype weight (e.g., from influence score)
        privatizer: Optional PrivacyLedger with real DP enabled. If provided,
            the prototype vector and weight will be clipped and noisy-ed before
            being sent to the server, providing (epsilon, delta)-DP protection.
    """
    import torch

    texts = [sample.rendered_text() for sample in samples if sample.rendered_text().strip()]
    if texts:
        prototype_vector = normalize(mean_vector(embedder.embed_texts(texts)))
    else:
        prototype_vector = []

    dp_applied = False
    if privatizer is not None and getattr(privatizer, '_dp_privatizer', None) is not None:
        # Apply real DP: clip + add Gaussian noise to prototype vector
        vec_tensor = torch.tensor(prototype_vector, dtype=torch.float32)
        privatized_vec = privatizer.privatize_vector(vec_tensor)
        prototype_vector = privatized_vec.tolist()
        weight = privatizer.privatize_scalar(weight)
        dp_applied = True

    return PrototypeFeedback(
        client_id=client_id,
        round_id=round_id,
        prototype_vector=prototype_vector,
        weight=float(weight),
        source_sample_ids=[sample.sample_id for sample in samples],
        meta={
            "sample_count": len(samples),
            "prototype_name": "minilm_mean",
            "dp_applied": dp_applied,
        },
    )
