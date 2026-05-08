from __future__ import annotations

import random
from typing import Any


def extract_texts(samples: list[Any]) -> list[str]:
    texts: list[str] = []
    for sample in samples:
        if hasattr(sample, "text"):
            raw = getattr(sample, "text")
        elif hasattr(sample, "render_text"):
            raw = sample.render_text()
        else:
            raw = str(sample)
        text = str(raw).strip()
        if len(text.split()) >= 2:
            texts.append(text)
    return texts


def _truncate_words(text: str, *, max_words: int | None) -> str:
    normalized = str(text).strip()
    if not normalized or max_words is None or int(max_words) <= 0:
        return normalized
    words = normalized.split()
    if len(words) <= int(max_words):
        return normalized
    return " ".join(words[: int(max_words)])


def _sample_texts(texts: list[str], *, count: int, seed: int) -> list[str]:
    cleaned = [text.strip() for text in texts if isinstance(text, str) and len(text.strip().split()) >= 2]
    if len(cleaned) <= count:
        return cleaned
    rng = random.Random(int(seed))
    chosen = sorted(rng.sample(range(len(cleaned)), k=count))
    return [cleaned[index] for index in chosen]


def build_c4_only_summary(*, init_texts: list[str], final_budget: int, seed: int) -> dict:
    synthetic_texts = _sample_texts(init_texts, count=final_budget, seed=seed)
    return {
        "mode": "c4_only",
        "selected_indices": [],
        "hard_negative_indices": [],
        "selected_texts": [],
        "hard_negative_texts": [],
        "hard_negative_reason": {},
        "boundary_state": {"negative_pattern_stats": {"count": 0}},
        "skip_bootstrap": True,
        "direct_synthetic_texts": synthetic_texts,
        "generator_contract": {"backend": "public_c4_sampler", "llm_backend": "none"},
        "privacy": {"enabled": False, "sigma": 0.0, "delta": 0.0},
        "seed_budget": {
            "mode": "direct_public_budget",
            "configured_seed_top_k": 0,
            "resolved_seed_top_k": 0,
            "final_direct_synthetic_count": len(synthetic_texts),
        },
        "decision": {"selected_indices": [], "hard_negative_indices": []},
    }


def build_expand_only_summary(
    *,
    init_texts: list[str],
    seed_top_k: int,
    seed: int,
    seed_text_max_words: int | None = None,
) -> dict:
    selected_texts = [
        _truncate_words(text, max_words=seed_text_max_words)
        for text in _sample_texts(init_texts, count=seed_top_k, seed=seed)
    ]
    return {
        "mode": "expand_only",
        "selected_indices": list(range(len(selected_texts))),
        "hard_negative_indices": [],
        "selected_texts": selected_texts,
        "hard_negative_texts": [],
        "hard_negative_reason": {},
        "boundary_state": {"negative_pattern_stats": {"count": 0}},
        "skip_bootstrap": False,
        "direct_synthetic_texts": [],
        "generator_contract": {"backend": "public_seed_sampler", "llm_backend": "none"},
        "privacy": {"enabled": False, "sigma": 0.0, "delta": 0.0},
        "seed_budget": {
            "mode": "fixed_public_seed_budget",
            "configured_seed_top_k": int(seed_top_k),
            "resolved_seed_top_k": len(selected_texts),
            "seed_text_max_words": None if seed_text_max_words is None else int(seed_text_max_words),
        },
        "decision": {"selected_indices": list(range(len(selected_texts))), "hard_negative_indices": []},
    }


def build_expand_private_summary(
    *,
    private_texts: list[str],
    seed_top_k: int,
    seed: int,
    seed_text_max_words: int | None = None,
) -> dict:
    selected_texts = [
        _truncate_words(text, max_words=seed_text_max_words)
        for text in _sample_texts(private_texts, count=seed_top_k, seed=seed)
    ]
    return {
        "mode": "expand_private",
        "selected_indices": list(range(len(selected_texts))),
        "hard_negative_indices": [],
        "selected_texts": selected_texts,
        "hard_negative_texts": [],
        "hard_negative_reason": {},
        "boundary_state": {"negative_pattern_stats": {"count": 0}},
        "skip_bootstrap": False,
        "direct_synthetic_texts": [],
        "generator_contract": {"backend": "private_seed_sampler", "llm_backend": "none"},
        "privacy": {"enabled": False, "sigma": 0.0, "delta": 0.0},
        "seed_budget": {
            "mode": "fixed_private_seed_budget",
            "configured_seed_top_k": int(seed_top_k),
            "resolved_seed_top_k": len(selected_texts),
            "seed_text_max_words": None if seed_text_max_words is None else int(seed_text_max_words),
        },
        "decision": {"selected_indices": list(range(len(selected_texts))), "hard_negative_indices": []},
    }
