from __future__ import annotations

import logging
import re

from thesis_platform.algorithms.critics.contrastive_critic_core import build_critique
from thesis_platform.algorithms.redaction import redact_rule_text
from thesis_platform.algorithms.rule_text import (
    extract_rules_from_text,
    guidance_tokens,
    is_actionable_guidance,
    looks_generic_instruction,
    looks_like_content_span,
)
from thesis_platform.core.schemas import Critique, PairedSample
from thesis_platform.core.llm_utils import safe_llm_generate

logger = logging.getLogger(__name__)
_LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _build_fallback_rules(pair: PairedSample, *, max_rules: int) -> list[str]:
    """Build deterministic critique guidance when LLM output is unusable."""

    heuristic = build_critique(pair, max_rules=max_rules, redact_enable=False)
    return [rule.strip() for rule in heuristic.rules if rule.strip()][:max_rules]


def _lexical_tokens(text: str) -> set[str]:
    return {token.lower() for token in _LEXICAL_TOKEN_RE.findall(text)}


def _overlap_ratio(rule_tokens: set[str], reference_tokens: set[str]) -> float:
    if not rule_tokens or not reference_tokens:
        return 0.0
    return len(rule_tokens & reference_tokens) / float(len(rule_tokens))


def _is_usable_rule(rule: str, pair: PairedSample) -> bool:
    """Reject low-value rules that mostly copy sample content instead of guiding edits."""

    tokens = guidance_tokens(rule)
    if not tokens:
        return False
    if len(tokens) < 4:
        return False
    if len(tokens) > 28:
        return False
    if looks_like_content_span(rule):
        return False
    if looks_generic_instruction(rule):
        return False
    if not is_actionable_guidance(rule):
        return False

    lexical_rule_tokens = _lexical_tokens(rule)
    real_tokens = _lexical_tokens(
        " ".join(sample.rendered_text() for sample in pair.real_samples)
    )
    bad_tokens = _lexical_tokens(pair.bad_sample.rendered_text())
    if len(tokens) >= 10 and (
        _overlap_ratio(lexical_rule_tokens, real_tokens) >= 0.75
        or _overlap_ratio(lexical_rule_tokens, bad_tokens) >= 0.75
    ):
        return False
    return True


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
        "Return valid JSON only with one top-level object: {\"rules\": [\"...\"]}.\n"
        "Each rule must be a concise natural-language improvement instruction.\n"
        "Each rule must be abstract guidance, not a rewritten sample sentence.\n"
        "Each rule must start with an imperative edit verb such as Add, Use, Remove, Focus, Clarify, Keep, Replace, or Reduce.\n"
        "Do not copy dates, names, locations, contact details, or event descriptions from the samples.\n"
        "Do not start a rule with a person, company, location, role title, or pronoun.\n"
        "Do not emit markdown fences. Do not emit partial JSON keys. Do not rewrite the bad sample.\n"
        "Do not reveal private entities.\n"
        f"Limit to {max_rules} rules.\n\n"
        f"Bad sample:\n{bad_text}\n\n"
        f"{real_examples}\n"
    )

    # Use safe LLM generation with retry logic
    raw_text = safe_llm_generate(
        backend=text_backend,
        prompt=prompt,
        max_new_tokens=196,
        temperature=0.2,
        max_retries=max_retries,
        fallback_response="" if use_fallback else None,
    )

    extracted_rules = extract_rules_from_text(raw_text, max_rules=max_rules * 2)
    rules = [rule for rule in extracted_rules if _is_usable_rule(rule, pair)][:max_rules]
    rejected_rule_count = max(len(extracted_rules) - len(rules), 0)
    fallback_reason = ""
    if not rules and use_fallback:
        logger.warning(
            "No usable critique rules generated for pair %s, using heuristic fallback",
            pair.pair_id,
        )
        rules = _build_fallback_rules(pair, max_rules=max_rules)
        fallback_reason = (
            "low_signal_llm_output"
            if not extracted_rules
            else "copy_like_llm_output"
        )

    # Apply redaction if enabled
    if redact_enable:
        rules = [redact_rule_text(rule) for rule in rules]

    critique_text = " ".join(rules)

    used_fallback = bool(fallback_reason)

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
            "fallback_reason": fallback_reason,
            "rejected_rule_count": rejected_rule_count,
            "llm_response_length": len(raw_text) if raw_text else 0,
        },
    )
