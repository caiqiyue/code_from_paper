from __future__ import annotations


def resolve_downstream_synthetic_texts(
    *,
    stage1_summary: dict,
    bootstrap_outputs: list[str],
) -> list[str]:
    if bool(stage1_summary.get("skip_bootstrap", False)):
        return list(stage1_summary.get("direct_synthetic_texts", []))
    return list(bootstrap_outputs)
