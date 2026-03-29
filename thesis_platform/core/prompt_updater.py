from __future__ import annotations

from thesis_platform.algorithms.rule_text import normalize_rule_candidate
from thesis_platform.core.schemas import PromptUpdate

BASE_HEADER = "### Base Instruction"
RULES_HEADER = "### Current Round Guidance"
MEMORY_HEADER = "### Memory Summary"
CLUSTER_HEADER = "### Cluster Guidance"


def _extract_base_prompt(current_prompt: str) -> str:
    """Recover the immutable base instruction from a formatted prompt."""

    if BASE_HEADER in current_prompt:
        remainder = current_prompt.split(BASE_HEADER, maxsplit=1)[1]
        for stop_header in (RULES_HEADER, MEMORY_HEADER):
            if stop_header in remainder:
                remainder = remainder.split(stop_header, maxsplit=1)[0]
        return remainder.strip()
    for stop_header in (RULES_HEADER, MEMORY_HEADER):
        if stop_header in current_prompt:
            return current_prompt.split(stop_header, maxsplit=1)[0].strip()
    return current_prompt.strip()


def build_prompt_text(
    base_prompt: str,
    *,
    global_rules: list[str],
    memory_rules: list[str] | None = None,
    local_rules: list[str] | None = None,
) -> str:
    """Render one prompt from base/global/local guidance sections."""

    memory_rules = [
        cleaned
        for cleaned in (normalize_rule_candidate(rule) for rule in (memory_rules or []))
        if cleaned
    ]
    local_rules = [
        cleaned
        for cleaned in (normalize_rule_candidate(rule) for rule in (local_rules or []))
        if cleaned
    ]
    global_rules = [
        cleaned
        for cleaned in (normalize_rule_candidate(rule) for rule in global_rules)
        if cleaned
    ]
    sections = [
        BASE_HEADER,
        str(base_prompt).strip(),
        "",
        RULES_HEADER,
        "\n".join(f"- {rule}" for rule in global_rules).strip(),
    ]
    if local_rules:
        sections.extend(
            [
                "",
                CLUSTER_HEADER,
                "\n".join(f"- {rule}" for rule in local_rules).strip(),
            ]
        )
    if memory_rules:
        sections.extend(
            [
                "",
                MEMORY_HEADER,
                "\n".join(f"- {rule}" for rule in memory_rules).strip(),
            ]
        )
    return "\n".join(part for part in sections if part is not None).strip()


def apply_prompt_update(current_prompt: str, update: PromptUpdate) -> str:
    """Rebuild the server-global prompt using base instruction and aggregated rules."""

    base_prompt = update.meta.get("base_prompt") or _extract_base_prompt(current_prompt)
    current_rules = update.global_rules or update.rules
    memory_rules = update.meta.get("memory_rules", [])
    return build_prompt_text(
        str(base_prompt).strip(),
        global_rules=current_rules,
        memory_rules=memory_rules,
    )


def render_cluster_prompt(current_prompt: str, update: PromptUpdate, cluster_id: str) -> str:
    """Render a cluster-specific prompt from one prompt update."""

    base_prompt = update.meta.get("base_prompt") or _extract_base_prompt(current_prompt)
    global_rules = update.global_rules or update.rules
    local_rules = list(update.cluster_rules.get(cluster_id, []))
    memory_rules = update.meta.get("memory_rules", [])
    return build_prompt_text(
        str(base_prompt).strip(),
        global_rules=global_rules,
        memory_rules=memory_rules,
        local_rules=local_rules,
    )
