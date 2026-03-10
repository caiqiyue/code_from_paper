from __future__ import annotations

import re
from collections import Counter

from thesis_platform.core.schemas import Critique, PromptUpdate


def _normalize_rule(rule: str) -> str:
    """Normalize one critique rule before frequency-based aggregation."""

    return re.sub(r"\s+", " ", rule.strip().lower())


def summarize_critiques(
    critiques: list[Critique],
    *,
    round_id: int,
    mode: str,
    max_rules: int = 5,
) -> PromptUpdate | None:
    """Aggregate critique rules into a single prompt update."""

    if not critiques:
        return None

    counter: Counter[str] = Counter()
    original_rules: dict[str, str] = {}
    for critique in critiques:
        for rule in critique.rules:
            normalized = _normalize_rule(rule)
            if not normalized:
                continue
            counter[normalized] += 1
            original_rules.setdefault(normalized, rule.strip())

    if not counter:
        return None

    if mode == "uid":
        ranked = sorted(
            counter.items(),
            key=lambda item: (item[1] / max(1, len(item[0].split())), item[1], -len(item[0])),  # Prefer dense and repeated rules.
            reverse=True,
        )
    else:
        ranked = sorted(counter.items(), key=lambda item: (item[1], -len(item[0])), reverse=True)

    rules = [original_rules[key] for key, _ in ranked[:max_rules]]
    summary = " ".join(rules)
    return PromptUpdate(
        update_id=f"{mode}_update_r{round_id}",
        round_id=round_id,
        rules=rules,
        summary=summary,
        prompt_text=summary,
        meta={"mode": mode, "source_critique_count": len(critiques)},
    )
