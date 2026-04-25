from __future__ import annotations


def compute_template_penalty(
    text: str,
    prompt_text: str,
    seed_texts: list[str],
    *,
    min_words: int,
    prompt_echo_ngram: int,
    unique_token_ratio_floor: float,
) -> float:
    normalized = str(text).strip()
    words = [token for token in normalized.split() if token]
    if not words:
        return 10.0

    penalty = 0.0
    if len(words) < int(min_words):
        penalty += 1.0

    prompt_tokens = [token for token in str(prompt_text).split() if token]
    if len(prompt_tokens) >= int(prompt_echo_ngram):
        prompt_window = " ".join(prompt_tokens[: int(prompt_echo_ngram)]).lower()
        if prompt_window in normalized.lower():
            penalty += 1.0

    lowered_words = [token.lower() for token in words]
    unique_ratio = len(set(lowered_words)) / max(1, len(lowered_words))
    if unique_ratio < float(unique_token_ratio_floor):
        penalty += 0.5

    normalized_lower = normalized.lower()
    for seed_text in seed_texts:
        seed_lower = str(seed_text).strip().lower()
        if seed_lower and normalized_lower == seed_lower:
            penalty += 0.5
            break

    return penalty
