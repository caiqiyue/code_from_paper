from __future__ import annotations

import statistics
from typing import Any

from thesis_platform.core.schemas import Critique, Sample


def _jaccard(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in left.split() if token.strip()}
    right_tokens = {token.lower() for token in right.split() if token.strip()}
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def compute_generation_metrics(
    samples: list[Sample],
    *,
    prompt_text: str | None = None,
    previous_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Compute research-mode generation statistics for one round."""

    texts = [sample.rendered_text() for sample in samples]
    lengths = [len(text.split()) for text in texts]
    prompt_sensitivity = statistics.mean(_jaccard(prompt_text or "", text) for text in texts) if texts else 0.0
    shift = 0.0
    if texts and previous_texts:
        pairs = zip(texts, previous_texts[: len(texts)])
        similarities = [_jaccard(current, previous) for current, previous in pairs]
        if similarities:
            shift = 1.0 - statistics.mean(similarities)
    return {
        "generated_count": len(texts),
        "avg_length": statistics.mean(lengths) if lengths else 0.0,
        "diversity": (len(set(texts)) / len(texts)) if texts else 0.0,
        "prompt_sensitivity": prompt_sensitivity,
        "generation_shift_vs_prev_round": shift,
    }


def compute_critique_metrics(critiques: list[Critique]) -> dict[str, Any]:
    """Compute aggregate statistics over uploaded critique rules."""

    rule_count = sum(len(item.rules) for item in critiques)
    text_lengths = [len(item.text.split()) for item in critiques]
    return {
        "critique_count": len(critiques),
        "critique_rule_count": rule_count,
        "avg_critique_length": statistics.mean(text_lengths) if text_lengths else 0.0,
    }


def compute_system_metrics(
    *,
    client_latency_s: float,
    server_latency_s: float,
    upload_tokens: int,
    prompt_text: str,
    backend_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute runtime, communication, and backend metadata for one round."""

    return {
        "client_latency_s": round(client_latency_s, 6),
        "server_latency_s": round(server_latency_s, 6),
        "upload_tokens": upload_tokens,
        "prompt_length_tokens": len(prompt_text.split()),
        "backend_names": backend_names or {},
    }
