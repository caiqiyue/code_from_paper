from __future__ import annotations

from thesis_platform.core.schemas import Sample


def compute_ira_scores(samples: list[Sample], *, text_backend) -> tuple[list[float], list[dict[str, float]]]:
    """Compute instruction-response alignment badness via conditional language-model loss."""

    scores: list[float] = []
    meta: list[dict[str, float]] = []
    for sample in samples:
        if not sample.response:
            raise ValueError("IRA requires samples with response fields.")
        unconditional = float(text_backend.negative_log_likelihood("", sample.response))
        conditional_prompt = sample.instruction or ""
        conditional = float(text_backend.negative_log_likelihood(conditional_prompt, sample.response))
        score = conditional - unconditional
        scores.append(score)
        meta.append({"loss_response_only": unconditional, "loss_response_given_instruction": conditional})
    return scores, meta
