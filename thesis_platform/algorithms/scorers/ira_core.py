from __future__ import annotations

from thesis_platform.core.schemas import Sample


def compute_ira_scores(samples: list[Sample], *, text_backend) -> tuple[list[float], list[dict[str, float]]]:
    """Compute instruction-response alignment badness via conditional language-model loss."""

    scores: list[float] = []
    meta: list[dict[str, float]] = []
    for sample in samples:
        # Use response if available, otherwise fall back to text field
        response_text = sample.response if sample.response else sample.text
        instruction_text = sample.instruction or ""
        unconditional = float(text_backend.negative_log_likelihood("", response_text))
        conditional = float(text_backend.negative_log_likelihood(instruction_text, response_text))
        score = conditional - unconditional
        scores.append(score)
        meta.append({"loss_response_only": unconditional, "loss_response_given_instruction": conditional, "response_source": "response" if sample.response else "text_fallback"})
    return scores, meta
