from __future__ import annotations

import statistics
from typing import Any

from thesis_platform.core.schemas import Critique, Sample


def compute_generation_metrics(samples: list[Sample]) -> dict[str, Any]:
    """Compute light-weight generation statistics for one round."""

    texts = [sample.text for sample in samples]
    lengths = [len(text.split()) for text in texts]
    return {
        "generated_count": len(texts),
        "avg_length": statistics.mean(lengths) if lengths else 0.0,
        "diversity": (len(set(texts)) / len(texts)) if texts else 0.0,
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


def compute_system_metrics(*, client_latency_s: float, server_latency_s: float, upload_tokens: int, prompt_text: str) -> dict[str, Any]:
    """Compute simple runtime and communication metrics for one round."""

    return {
        "client_latency_s": round(client_latency_s, 6),
        "server_latency_s": round(server_latency_s, 6),
        "upload_tokens": upload_tokens,
        "prompt_length_tokens": len(prompt_text.split()),
    }
