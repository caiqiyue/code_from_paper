from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

import numpy as np

from thesis_platform.algorithms.math_utils import cosine_similarity
from thesis_platform.core.schemas import Critique, PromptUpdate

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_rule(rule: str) -> str:
    return re.sub(r"\s+", " ", rule.strip().lower())


def _parse_summary_rules(raw_text: str, *, fallback_rules: list[str], max_rules: int) -> tuple[list[str], str]:
    match = JSON_RE.search(raw_text)
    if match:
        try:
            payload = json.loads(match.group(0))
            rules = payload.get("rules", []) if isinstance(payload, dict) else []
            memory_summary = str(payload.get("memory_summary", "")).strip() if isinstance(payload, dict) else ""
            parsed_rules = [str(rule).strip() for rule in rules if str(rule).strip()]
            if parsed_rules:
                return parsed_rules[:max_rules], memory_summary
        except json.JSONDecodeError:
            pass
    lines = []
    for line in raw_text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            lines.append(cleaned)
    return (lines[:max_rules] or fallback_rules[:max_rules]), ""


def _cluster_rule_entries(rule_entries: list[dict[str, Any]], vectors: np.ndarray, *, eps: float, min_samples: int) -> list[dict[str, Any]]:
    from sklearn.cluster import DBSCAN

    if not rule_entries:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(vectors)
    clusters: list[dict[str, Any]] = []
    next_noise_cluster = 0
    for raw_label in sorted(set(int(label) for label in labels)):
        indices = [idx for idx, label in enumerate(labels) if int(label) == raw_label]
        if raw_label < 0:
            for index in indices:
                centroid = vectors[index]
                clusters.append(
                    {
                        "cluster_id": f"noise_{next_noise_cluster}",
                        "member_indices": [index],
                        "members": [rule_entries[index]],
                        "centroid": centroid.tolist(),
                    }
                )
                next_noise_cluster += 1
            continue
        centroid = np.mean(vectors[indices], axis=0)
        clusters.append(
            {
                "cluster_id": f"cluster_{raw_label}",
                "member_indices": indices,
                "members": [rule_entries[index] for index in indices],
                "centroid": centroid.tolist(),
            }
        )
    return clusters


def _rank_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for cluster in clusters:
        members = cluster["members"]
        support = len(members)
        severity = sum(float(member["source_score"]) for member in members) / max(support, 1)
        density = support / max(
            1.0,
            sum(len(str(member["rule"]).split()) for member in members) / max(support, 1),
        )
        counts = Counter(_normalize_rule(str(member["rule"])) for member in members)
        representative_key = counts.most_common(1)[0][0]
        representative_rule = next(
            str(member["rule"]).strip() for member in members if _normalize_rule(str(member["rule"])) == representative_key
        )
        ranked.append(
            {
                **cluster,
                "support": support,
                "severity": severity,
                "density": density,
                "score": support * (1.0 + severity) * density,
                "representative_rule": representative_rule,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _merge_memory(
    ranked_clusters: list[dict[str, Any]],
    memory: dict[str, Any],
    *,
    momentum_beta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    memory_entries = list(memory.get("entries", []))
    updated_entries: list[dict[str, Any]] = []

    for cluster in ranked_clusters:
        centroid = cluster["centroid"]
        matched_index = None
        for index, entry in enumerate(memory_entries):
            similarity = cosine_similarity(list(map(float, centroid)), list(map(float, entry.get("centroid", []))))
            if similarity >= 0.85:
                matched_index = index
                break
        if matched_index is None:
            updated_entries.append(
                {
                    "rule": cluster["representative_rule"],
                    "weight": float(cluster["score"]),
                    "centroid": centroid,
                    "last_round_score": float(cluster["score"]),
                }
            )
            cluster["memory_weight"] = float(cluster["score"])
        else:
            previous = memory_entries.pop(matched_index)
            memory_weight = momentum_beta * float(previous.get("weight", 0.0)) + (1.0 - momentum_beta) * float(cluster["score"])
            updated_entries.append(
                {
                    "rule": cluster["representative_rule"],
                    "weight": memory_weight,
                    "centroid": centroid,
                    "last_round_score": float(cluster["score"]),
                }
            )
            cluster["memory_weight"] = memory_weight

    updated_entries.extend(memory_entries)
    updated_entries.sort(key=lambda item: float(item.get("weight", 0.0)), reverse=True)
    memory_rules = [str(item["rule"]).strip() for item in updated_entries[:5]]

    for cluster in ranked_clusters:
        cluster["score"] = float(cluster.get("memory_weight", cluster["score"]))
    ranked_clusters.sort(key=lambda item: item["score"], reverse=True)
    return ranked_clusters, {"entries": updated_entries[:16]}, memory_rules


def summarize_rules_with_llm(rules: list[str], *, text_backend, max_rules: int) -> tuple[list[str], str]:
    """Compress representative rules into a dense non-redundant summary."""

    if not rules:
        return [], ""
    prompt = (
        "You are the server aggregation model for federated prompt optimization.\n"
        "Return JSON with keys `rules` and `memory_summary`.\n"
        "Compress the candidate guidance below into a dense, non-redundant summary.\n"
        f"Keep at most {max_rules} rules.\n\n"
        + "\n".join(f"- {rule}" for rule in rules)
    )
    raw_text = text_backend.generate(prompt, max_new_tokens=220)
    return _parse_summary_rules(raw_text, fallback_rules=rules, max_rules=max_rules)


def aggregate_dbscan_critiques(
    critiques: list[Critique],
    *,
    round_id: int,
    max_rules: int,
    embedder,
    text_backend=None,
    eps: float = 0.35,
    min_samples: int = 2,
    use_memory: bool = False,
    memory: dict[str, Any] | None = None,
    momentum_beta: float = 0.7,
    base_prompt: str | None = None,
) -> tuple[PromptUpdate | None, dict[str, Any]]:
    """Cluster, rank, and summarize critique rules into a prompt update."""

    rule_entries: list[dict[str, Any]] = []
    for critique in critiques:
        for rule in critique.rules:
            normalized = _normalize_rule(rule)
            if not normalized:
                continue
            rule_entries.append(
                {
                    "rule": rule.strip(),
                    "normalized": normalized,
                    "source_score": float(critique.meta.get("source_score", 0.0)),
                    "client_id": critique.client_id,
                }
            )
    if not rule_entries:
        return None, memory or {"entries": []}

    vectors = np.asarray(embedder.embed_texts([entry["rule"] for entry in rule_entries]), dtype=np.float64)
    ranked_clusters = _rank_clusters(_cluster_rule_entries(rule_entries, vectors, eps=eps, min_samples=min_samples))

    memory_rules: list[str] = []
    updated_memory = memory or {"entries": []}
    if use_memory:
        ranked_clusters, updated_memory, memory_rules = _merge_memory(
            ranked_clusters,
            updated_memory,
            momentum_beta=momentum_beta,
        )

    representative_rules = [cluster["representative_rule"] for cluster in ranked_clusters[: max_rules * 2]]
    final_rules = representative_rules[:max_rules]
    memory_summary = ""
    if text_backend is not None and final_rules:
        final_rules, memory_summary = summarize_rules_with_llm(final_rules, text_backend=text_backend, max_rules=max_rules)
    if not final_rules:
        return None, updated_memory

    compression_ratio = len(rule_entries) / max(len(final_rules), 1)
    prompt_update = PromptUpdate(
        update_id=f"dbscan_update_r{round_id}",
        round_id=round_id,
        rules=final_rules,
        summary=" ".join(final_rules),
        prompt_text=" ".join(final_rules),
        meta={
            "mode": "dbscan_attn_tsgdm" if use_memory else "dbscan_attn",
            "cluster_count": len(ranked_clusters),
            "clusters": ranked_clusters,
            "compression_ratio": compression_ratio,
            "memory_rules": memory_rules,
            "memory_summary": memory_summary,
            "source_critique_count": len(critiques),
            "base_prompt": base_prompt or "",
        },
    )
    return prompt_update, updated_memory


def aggregate_uid_critiques(
    critiques: list[Critique],
    *,
    round_id: int,
    max_rules: int,
    text_backend,
    base_prompt: str | None = None,
) -> PromptUpdate | None:
    """Summarize raw critique rules directly with the server LLM."""

    rules = [rule.strip() for critique in critiques for rule in critique.rules if rule.strip()]
    if not rules:
        return None
    final_rules, memory_summary = summarize_rules_with_llm(rules[: max_rules * 3], text_backend=text_backend, max_rules=max_rules)
    return PromptUpdate(
        update_id=f"uid_llm_update_r{round_id}",
        round_id=round_id,
        rules=final_rules,
        summary=" ".join(final_rules),
        prompt_text=" ".join(final_rules),
        meta={
            "mode": "uid_llm",
            "cluster_count": len(final_rules),
            "compression_ratio": len(rules) / max(len(final_rules), 1),
            "memory_rules": final_rules,
            "memory_summary": memory_summary,
            "source_critique_count": len(critiques),
            "base_prompt": base_prompt or "",
        },
    )
