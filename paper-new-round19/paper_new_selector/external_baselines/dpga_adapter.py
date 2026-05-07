from __future__ import annotations


def build_dpga_stage1_summary(*, texts: list[str], budget: int) -> dict:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_text in texts:
        text = str(raw_text).strip()
        if len(text.split()) < 2:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    final_texts = cleaned[: max(0, int(budget))]
    return {
        "mode": "dpga_adapter",
        "selected_indices": [],
        "hard_negative_indices": [],
        "selected_texts": [],
        "hard_negative_texts": [],
        "hard_negative_reason": {},
        "boundary_state": {
            "reject_score_ceiling": 0.0,
            "reject_score_floor": 0.0,
            "negative_centroid": [],
            "negative_pattern_stats": {"count": 0},
        },
        "skip_bootstrap": True,
        "direct_synthetic_texts": final_texts,
        "generator_contract": {
            "backend": "dpga_textsyn",
            "llm_backend": "external_repo",
        },
        "privacy": {
            "enabled": False,
            "sigma": 0.0,
            "delta": 0.0,
        },
        "seed_budget": {
            "mode": "external_direct_budget",
            "configured_seed_top_k": 0,
            "resolved_seed_top_k": 0,
            "final_direct_synthetic_count": len(final_texts),
        },
        "decision": {
            "source": "dpga_external_export",
            "final_count": len(final_texts),
        },
    }
