from __future__ import annotations

import json
import logging
import re

from thesis_platform.algorithms.redaction import redact_rule_text
from thesis_platform.core.schemas import Critique, PairedSample
from thesis_platform.core.llm_utils import safe_llm_generate, parse_json_with_fallback

logger = logging.getLogger(__name__)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Fallback rules when LLM fails completely
FALLBACK_RULES = [
    "Improve clarity and coherence",
    "Ensure factual accuracy",
    "Maintain consistent style",
]


def _parse_rules(raw_text: str, *, max_rules: int) -> list[str]:
    """Parse critique rules from JSON or bullet-list output."""

    match = JSON_RE.search(raw_text)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
                rules = [
                    str(rule).strip() for rule in payload["rules"] if str(rule).strip()
                ]
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
    max_retries: int = 3,
    use_fallback: bool = True,
) -> Critique:
    """Generate textual-gradient critique rules with a local small language model.

    Args:
        pair: PairedSample containing bad sample and real samples
        text_backend: LLM backend for generation
        max_rules: Maximum number of rules to generate
        redact_enable: Whether to redact sensitive information
        max_retries: Maximum retry attempts for LLM call
        use_fallback: Whether to use fallback rules on failure

    Returns:
        Critique object with generated rules
    """

    bad_text = pair.bad_sample.rendered_text()
    real_examples = "\n\n".join(
        f"Real sample {idx + 1}:\n{sample.rendered_text()}"
        for idx, sample in enumerate(pair.real_samples)
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

    # Use safe LLM generation with retry logic
    raw_text = safe_llm_generate(
        backend=text_backend,
        prompt=prompt,
        max_new_tokens=196,
        temperature=0.7,
        max_retries=max_retries,
        fallback_response="" if use_fallback else None,
    )

    # Parse rules
    rules = _parse_rules(raw_text, max_rules=max_rules)

    # Use fallback if no rules generated and fallback is enabled
    if not rules and use_fallback:
        logger.warning(f"No rules generated for pair {pair.pair_id}, using fallback")
        rules = FALLBACK_RULES[:max_rules]

    # Apply redaction if enabled
    if redact_enable:
        rules = [redact_rule_text(rule) for rule in rules]

    critique_text = " ".join(rules)

    # Determine if fallback was used
    used_fallback = len(rules) > 0 and rules[0] in FALLBACK_RULES

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
            "backend_name": getattr(
                text_backend, "backend_name", type(text_backend).__name__
            ),
            "used_fallback": used_fallback,
            "llm_response_length": len(raw_text) if raw_text else 0,
        },
    )
