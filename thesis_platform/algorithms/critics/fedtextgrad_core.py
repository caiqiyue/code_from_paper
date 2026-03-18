from __future__ import annotations

import json
import re

from thesis_platform.algorithms.redaction import redact_rule_text
from thesis_platform.core.schemas import Critique, PairedSample


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_rules(raw_text: str, *, max_rules: int) -> list[str]:
    """Parse critique rules from JSON or bullet-list output."""

    match = JSON_RE.search(raw_text)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
                rules = [str(rule).strip() for rule in payload["rules"] if str(rule).strip()]
                if rules:
                    return rules[:max_rules]
        except json.JSONDecodeError:
            pass
    rules: list[str] = []
    for line in raw_text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            rules.append(cleaned)
    if not rules and raw_text.strip():
        rules = [raw_text.strip()]
    return rules[:max_rules]


def build_textual_gradient_critique(
    pair: PairedSample,
    *,
    text_backend,
    max_rules: int,
    redact_enable: bool,
) -> Critique:
    """Generate textual-gradient critique rules with a local small language model."""

    bad_text = pair.bad_sample.rendered_text()
    real_examples = "\n\n".join(
        f"Real sample {idx + 1}:\n{sample.rendered_text()}" for idx, sample in enumerate(pair.real_samples)
    )
    prompt = (
        "You are a local critique model for federated prompt optimization.\n"
        "Compare one synthetic bad sample against retrieved real samples from the client's private domain.\n"
        "Return JSON with a `rules` array.\n"
        "Rules must be concise natural-language improvement instructions.\n"
        "Do not rewrite the bad sample. Do not reveal private entities.\n"
        f"Limit to {max_rules} rules.\n\n"
        f"Bad sample:\n{bad_text}\n\n"
        f"{real_examples}\n"
    )
    raw_text = text_backend.generate(prompt, max_new_tokens=196)
    rules = _parse_rules(raw_text, max_rules=max_rules)
    if redact_enable:
        rules = [redact_rule_text(rule) for rule in rules]
    critique_text = " ".join(rules)
    return Critique(
        critique_id=f"critique_{pair.pair_id}",
        client_id=pair.client_id,
        round_id=pair.round_id,
        bad_sample_id=pair.bad_sample.sample_id,
        real_sample_ids=[sample.sample_id for sample in pair.real_samples],
        rules=rules,
        text=critique_text,
        meta={
            "rule_count": len(rules),
            "redacted": redact_enable,
            "source_score": float(getattr(pair.bad_sample, "score", 0.0)),
            "backend_name": getattr(text_backend, "backend_name", type(text_backend).__name__),
        },
    )
