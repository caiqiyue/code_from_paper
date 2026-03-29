from __future__ import annotations

import ast
import json
import re
from typing import Any

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
CODE_FENCE_TOKEN_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?", re.IGNORECASE)
KEY_VALUE_FRAGMENT_RE = re.compile(
    r'^[\\"]*(?P<key>[A-Za-z_][\w-]*)[\\"]*\s*:\s*[\\"]?(?P<value>.+?)[\\"]?\s*,?$'
)

_LOW_SIGNAL_EXACT = {
    "",
    "json",
    "```",
    "```json",
    "[",
    "]",
    "{",
    "}",
    ",",
    ":",
    "rules",
}
_LOW_SIGNAL_LINE_RE = re.compile(
    r'^"?[A-Za-z_][\w-]*"?\s*:\s*(?:\{|\[)?$|^[\[\]\{\},:"]+$'
)
_PREFERRED_RULE_KEYS = (
    "description",
    "instruction",
    "rule",
    "guidance",
    "summary",
    "text",
    "action",
)
_GUIDANCE_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>-]+")
ACTIONABLE_LEAD_TOKENS = {
    "add",
    "adjust",
    "align",
    "avoid",
    "clarify",
    "emphasize",
    "focus",
    "highlight",
    "improve",
    "include",
    "increase",
    "keep",
    "make",
    "match",
    "preserve",
    "reduce",
    "remove",
    "replace",
    "retain",
    "rewrite",
    "tighten",
    "use",
}
_ACTIONABLE_SUBJECT_TOKENS = {
    "answer",
    "content",
    "description",
    "draft",
    "instruction",
    "output",
    "prompt",
    "response",
    "sample",
    "summary",
    "text",
    "tone",
    "wording",
}
_LEADING_NOISE_TOKENS = {"a", "an", "the", "this", "that", "these", "those", "<entity>"}
_ACTIONABLE_SUBJECT_PATTERN = re.compile(
    r"^(?:(?:the|this|that|a|an)\s+)?"
    r"(?:answer|content|description|draft|instruction|output|prompt|response|sample|summary|text|tone|wording)\s+"
    r"(?:should|must|needs?\s+to)\b",
    re.IGNORECASE,
)
_ACTIONABLE_PREFIX_PATTERN = re.compile(
    r"^(?:add|adjust|align|avoid|clarify|emphasize|focus|highlight|improve|include|increase|keep|make|match|preserve|reduce|remove|replace|retain|rewrite|tighten|use)\b",
    re.IGNORECASE,
)
_DESCRIPTIVE_PREFIX_PATTERN = re.compile(
    r"^(?:(?:the|this|that|a|an)\s+)?"
    r"(?:company|candidate|employee|user|platform|process|project|team|role|department|organization|organisation|system|tool|document|product|service|person|individual)\b",
    re.IGNORECASE,
)
_BE_VERB_RE = re.compile(r"\b(?:is|are|was|were|be|been|being)\b", re.IGNORECASE)
_GENERIC_INSTRUCTION_PHRASES = (
    "word or phrase",
    "in a sentence",
    "the sentence",
    "the document",
    "the system",
    "concept or idea",
    "instruction or task",
    "new piece of information",
    "existing information",
    "other sources",
    "new line at the beginning",
)
_CONTEXTUAL_ANCHOR_TOKENS = {
    "alignment",
    "client",
    "concrete",
    "context",
    "detail",
    "details",
    "domain",
    "examples",
    "generic",
    "off-domain",
    "prompt",
    "real",
    "retrieved",
    "sample",
    "samples",
    "specificity",
    "structure",
    "technical",
    "terminology",
    "tone",
}


def strip_code_fences(text: str) -> str:
    """Remove markdown code fence tokens while keeping inner content."""

    cleaned = CODE_FENCE_TOKEN_RE.sub("", str(text))
    return cleaned.replace("```", "").strip()


def is_low_signal_rule(text: str) -> bool:
    """Return true when one candidate looks like formatting or JSON scaffolding."""

    normalized = re.sub(r"\s+", " ", strip_code_fences(text)).strip().strip(",")
    lowered = normalized.lower()
    if lowered in _LOW_SIGNAL_EXACT:
        return True
    if _LOW_SIGNAL_LINE_RE.fullmatch(normalized):
        return True
    if len(normalized) <= 2:
        return True
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", normalized):
        return True
    return False


def normalize_rule_candidate(text: str) -> str:
    """Normalize one rule candidate into concise prompt-safe natural language."""

    cleaned = strip_code_fences(text).replace('\\"', '"').replace("\\'", "'")
    fragment = KEY_VALUE_FRAGMENT_RE.match(cleaned.strip())
    if fragment:
        key = fragment.group("key").lower()
        if key not in _PREFERRED_RULE_KEYS:
            return ""
        cleaned = fragment.group("value")
    cleaned = cleaned.strip().strip(",")
    cleaned = cleaned.lstrip("-*").strip()
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    if re.fullmatch(r"[a-z]+(?:_[a-z]+)+", cleaned):
        cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if is_low_signal_rule(cleaned):
        return ""
    return cleaned


def guidance_tokens(text: str) -> list[str]:
    """Tokenize one guidance string for heuristic quality checks."""

    return [token.lower() for token in _GUIDANCE_TOKEN_RE.findall(strip_code_fences(text))]


def has_actionable_lead(text: str) -> bool:
    """Return true when guidance begins like an editing instruction."""

    normalized = re.sub(r"\s+", " ", strip_code_fences(text)).strip()
    if not normalized:
        return False
    if _ACTIONABLE_PREFIX_PATTERN.match(normalized):
        return True

    tokens = guidance_tokens(normalized)
    meaningful: list[str] = []
    for token in tokens:
        if token in _LEADING_NOISE_TOKENS:
            continue
        meaningful.append(token)
        if len(meaningful) >= 3:
            break
    if meaningful and meaningful[0] in ACTIONABLE_LEAD_TOKENS:
        return True
    if meaningful[:2] and any(token in ACTIONABLE_LEAD_TOKENS for token in meaningful[:2]):
        return True
    return bool(_ACTIONABLE_SUBJECT_PATTERN.match(normalized))


def is_actionable_guidance(text: str) -> bool:
    """Return true when a rule looks like abstract instruction instead of copied content."""

    normalized = re.sub(r"\s+", " ", strip_code_fences(text)).strip()
    tokens = guidance_tokens(normalized)
    if not tokens or not has_actionable_lead(normalized):
        return False
    if _DESCRIPTIVE_PREFIX_PATTERN.match(normalized) and _BE_VERB_RE.search(normalized):
        return bool(_ACTIONABLE_SUBJECT_PATTERN.match(normalized))
    return True


def looks_generic_instruction(text: str) -> bool:
    """Return true when a rule is actionable but too generic to guide prompt updates."""

    normalized = re.sub(r"\s+", " ", strip_code_fences(text)).strip().lower()
    if not normalized:
        return False
    tokens = guidance_tokens(normalized)
    if "something" in tokens:
        return True
    if any(phrase in normalized for phrase in _GENERIC_INSTRUCTION_PHRASES):
        return not any(token in _CONTEXTUAL_ANCHOR_TOKENS for token in tokens)
    return False


def looks_like_content_span(text: str) -> bool:
    """Return true when text resembles copied sample content instead of prompt guidance."""

    tokens = guidance_tokens(text)
    if len(tokens) >= 32:
        return True
    placeholder_count = sum(
        1 for token in tokens if token.startswith("<") and token.endswith(">")
    )
    if len(tokens) >= 12 and placeholder_count >= max(4, len(tokens) // 3):
        return True
    return len(tokens) >= 18 and not is_actionable_guidance(text)


def _parse_structured_payload(text: str) -> Any | None:
    """Try parsing one text blob as JSON or Python literal."""

    candidates = [str(text).strip(), strip_code_fences(text)]
    for raw in list(candidates):
        object_match = JSON_OBJECT_RE.search(raw)
        if object_match:
            candidates.append(object_match.group(0).strip())
        array_match = JSON_ARRAY_RE.search(raw)
        if array_match:
            candidates.append(array_match.group(0).strip())

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return None


def _extract_rule_strings_from_payload(payload: Any) -> list[str]:
    """Recursively extract human-usable rule strings from structured payloads."""

    if payload is None:
        return []
    if isinstance(payload, str):
        nested = _parse_structured_payload(payload)
        if nested is not None and nested != payload:
            return _extract_rule_strings_from_payload(nested)
        return [payload]
    if isinstance(payload, (list, tuple, set)):
        rules: list[str] = []
        for item in payload:
            rules.extend(_extract_rule_strings_from_payload(item))
        return rules
    if isinstance(payload, dict):
        if "rules" in payload:
            return _extract_rule_strings_from_payload(payload.get("rules"))
        for key in _PREFERRED_RULE_KEYS:
            value = payload.get(key)
            if value not in (None, ""):
                return _extract_rule_strings_from_payload(value)
        return []
    return [str(payload)]


def extract_rules_from_text(raw_text: str, *, max_rules: int) -> list[str]:
    """Extract deduplicated, prompt-safe rules from raw model output."""

    candidates: list[str] = []

    structured = _parse_structured_payload(raw_text)
    if structured is not None:
        candidates.extend(_extract_rule_strings_from_payload(structured))

    for line in strip_code_fences(raw_text).splitlines():
        line = line.strip()
        if not line:
            continue
        nested = _parse_structured_payload(line)
        if nested is not None and nested != line:
            candidates.extend(_extract_rule_strings_from_payload(nested))
        else:
            candidates.append(line)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_rule_candidate(candidate)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= max_rules:
            break
    return deduped


def extract_memory_summary(raw_text: str) -> str:
    """Extract one optional memory summary string from structured LLM output."""

    payload = _parse_structured_payload(raw_text)
    if isinstance(payload, dict):
        summary = normalize_rule_candidate(str(payload.get("memory_summary", "")))
        return summary
    return ""
