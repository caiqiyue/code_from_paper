from __future__ import annotations

import re
from collections import Counter

from thesis_platform.core.schemas import Critique, PairedSample

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _redact(text: str) -> str:
    text = re.sub(r"\b\d+\b", "<NUMBER>", text)
    text = re.sub(r"\b[A-Z][a-z]{2,}\b", "<ENTITY>", text)
    return text


def build_critique(
    pair: PairedSample,
    *,
    max_rules: int = 2,
    redact_enable: bool = True,
) -> Critique:
    bad_tokens = Counter(_tokens(pair.bad_sample.text))
    real_text = " ".join(sample.text for sample in pair.real_samples)
    real_tokens = Counter(_tokens(real_text))

    missing = [token for token, _ in (real_tokens - bad_tokens).most_common(5) if len(token) > 4]
    extra = [token for token, _ in (bad_tokens - real_tokens).most_common(5) if len(token) > 4]

    rules: list[str] = []
    if pair.real_samples:
        avg_real_len = sum(len(sample.text.split()) for sample in pair.real_samples) / len(pair.real_samples)
        if len(pair.bad_sample.text.split()) < avg_real_len * 0.7:
            rules.append("Add more concrete detail and domain-specific structure to match the retrieved real examples.")
    if missing:
        rules.append(f"Add important domain terms and concepts that appear in real samples, such as: {', '.join(missing[:3])}.")
    if extra:
        rules.append(f"Reduce off-topic or overly generic wording that does not align with the real samples, such as: {', '.join(extra[:3])}.")
    if not rules:
        rules.append("Align the synthetic text more closely with the tone, specificity, and structure of the retrieved real samples.")

    rules = rules[:max_rules]
    text = " ".join(rules)
    if redact_enable:
        rules = [_redact(rule) for rule in rules]
        text = _redact(text)

    return Critique(
        critique_id=f"critique_{pair.pair_id}",
        client_id=pair.client_id,
        round_id=pair.round_id,
        bad_sample_id=pair.bad_sample.sample_id,
        real_sample_ids=[sample.sample_id for sample in pair.real_samples],
        rules=rules,
        text=text,
        meta={"rule_count": len(rules), "redacted": redact_enable},
    )
