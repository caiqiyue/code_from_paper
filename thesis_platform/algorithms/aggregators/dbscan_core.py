from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

import numpy as np

from thesis_platform.algorithms.math_utils import cosine_similarity
from thesis_platform.core.schemas import Critique, PromptUpdate, PrototypeFeedback

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_rule(rule: str) -> str:
    return re.sub(r"\s+", " ", rule.strip().lower())


def _parse_summary_rules(
    raw_text: str, *, fallback_rules: list[str], max_rules: int
) -> tuple[list[str], str]:
    match = JSON_RE.search(raw_text)
    if match:
        try:
            payload = json.loads(match.group(0))
            rules = payload.get("rules", []) if isinstance(payload, dict) else []
            memory_summary = (
                str(payload.get("memory_summary", "")).strip()
                if isinstance(payload, dict)
                else ""
            )
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


def _cluster_rule_entries(
    rule_entries: list[dict[str, Any]],
    vectors: np.ndarray,
    *,
    eps: float,
    min_samples: int,
) -> list[dict[str, Any]]:
    from sklearn.cluster import DBSCAN

    if not rule_entries:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(
        vectors
    )
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


def _cluster_client_prototypes(
    prototype_feedbacks: list[PrototypeFeedback],
    *,
    eps: float,
    min_samples: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Cluster client prototypes for personalized routing."""

    if not prototype_feedbacks:
        return {}, []

    usable_feedbacks = [
        feedback for feedback in prototype_feedbacks if feedback.prototype_vector
    ]
    if len(usable_feedbacks) <= 1:
        feedbacks = usable_feedbacks or prototype_feedbacks
        mapping = {
            feedback.client_id: f"cluster_{idx}"
            for idx, feedback in enumerate(feedbacks)
        }
        payload = [
            {
                "cluster_id": mapping[feedback.client_id],
                "client_ids": [feedback.client_id],
                "weight_sum": float(feedback.weight),
                "prototype_count": 1,
            }
            for feedback in feedbacks
        ]
        return mapping, payload

    from sklearn.cluster import DBSCAN

    vectors = np.asarray(
        [feedback.prototype_vector for feedback in usable_feedbacks], dtype=np.float64
    )
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(
        vectors
    )
    mapping: dict[str, str] = {}
    grouped: dict[str, list[PrototypeFeedback]] = {}
    noise_index = 0
    for feedback, raw_label in zip(usable_feedbacks, labels):
        if int(raw_label) < 0:
            cluster_id = f"noise_{noise_index}"
            noise_index += 1
        else:
            cluster_id = f"cluster_{int(raw_label)}"
        mapping[feedback.client_id] = cluster_id
        grouped.setdefault(cluster_id, []).append(feedback)

    payload = []
    for cluster_id, members in grouped.items():
        payload.append(
            {
                "cluster_id": cluster_id,
                "client_ids": [member.client_id for member in members],
                "weight_sum": float(sum(member.weight for member in members)),
                "prototype_count": len(members),
            }
        )
    payload.sort(key=lambda item: item["cluster_id"])
    return mapping, payload


def _compute_attention_weight(r_k: float) -> float:
    """Compute attention weight using softmax-style normalization: exp(R_k) / Σ exp(R_j)."""
    return np.exp(r_k)


def _rank_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank clusters by utility-weighted severity, support, and density."""
    ranked: list[dict[str, Any]] = []
    for cluster in clusters:
        members = cluster["members"]
        support = len(members)

        # Extract R_k utility scores for attention-weighted severity
        r_k_values = [
            float(member.get("r_k_utility", member.get("source_score", 0.0)))
            for member in members
        ]
        attention_weights = [_compute_attention_weight(r_k) for r_k in r_k_values]
        total_attention = sum(attention_weights)

        # Utility-weighted severity: Σ α_k * source_score, where α_k = exp(R_k) / Σ exp(R_j)
        if total_attention > 0 and support > 0:
            weighted_severity = sum(
                (attention_weights[i] / total_attention)
                * float(members[i]["source_score"])
                for i in range(support)
            )
        else:
            weighted_severity = sum(
                float(member["source_score"]) for member in members
            ) / max(support, 1)

        severity = weighted_severity
        density = support / max(
            1.0,
            sum(len(str(member["rule"]).split()) for member in members)
            / max(support, 1),
        )
        counts = Counter(_normalize_rule(str(member["rule"])) for member in members)
        representative_key = counts.most_common(1)[0][0]
        representative_rule = next(
            str(member["rule"]).strip()
            for member in members
            if _normalize_rule(str(member["rule"])) == representative_key
        )
        ranked.append(
            {
                **cluster,
                "support": support,
                "severity": severity,
                "density": density,
                "score": support * (1.0 + severity) * density,
                "representative_rule": representative_rule,
                "total_attention": total_attention,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _project_conflicting_rules(
    vectors: np.ndarray,
    weights: list[float],
    *,
    conflict_threshold: float = 0.0,
) -> tuple[np.ndarray, list[float], list[dict[str, Any]]]:
    """Resolve rule conflicts using KKT-inspired orthogonal projection.

    For each pair of conflicting rules (cos < conflict_threshold):
    - Project the weaker rule's vector onto the hyperplane orthogonal to the stronger rule
    - v_weak_proj = v_weak - (v_weak · v_strong_normalized) * v_strong_normalized
    - This retains the non-conflicting component of the weaker rule

    Returns:
        projected_vectors: vectors with conflicts resolved via projection
        adjusted_weights: weights with conflict penalty applied
        projection_log: details of each projection for transparency
    """
    if len(vectors) <= 1:
        return vectors, weights, []

    projected = vectors.copy()
    adjusted_weights = list(weights)
    projection_log: list[dict[str, Any]] = []

    # Build conflict graph: which vectors conflict with which
    conflicts: dict[int, list[int]] = {i: [] for i in range(len(vectors))}
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i].tolist(), vectors[j].tolist())
            if sim < conflict_threshold:
                # They conflict - weaker one gets projected
                if weights[i] < weights[j]:
                    conflicts[j].append(i)  # i is dominated by j
                elif weights[j] < weights[i]:
                    conflicts[i].append(j)  # j is dominated by i
                else:
                    # Equal weights - both get projected partially
                    conflicts[i].append(j)
                    conflicts[j].append(i)

    # Process each vector that has conflicts
    processed = set()
    for strong_idx in range(len(vectors)):
        if not conflicts[strong_idx]:
            continue

        strong_vector = vectors[strong_idx]
        strong_norm = np.linalg.norm(strong_vector)
        if strong_norm < 1e-8:
            continue
        strong_normalized = strong_vector / strong_norm

        for weak_idx in conflicts[strong_idx]:
            if weak_idx in processed:
                continue
            weak_vector = vectors[weak_idx]

            # Orthogonal projection: remove component along strong direction
            # v_weak_proj = v_weak - (v_weak · v_strong_normalized) * v_strong_normalized
            dot_product = np.dot(weak_vector, strong_normalized)
            projected_component = dot_product * strong_normalized
            projected[weak_idx] = weak_vector - projected_component

            # Reduce weight proportionally to conflict severity
            conflict_severity = abs(
                cosine_similarity(weak_vector.tolist(), strong_vector.tolist())
            )
            adjusted_weights[weak_idx] = weights[weak_idx] * (1.0 - conflict_severity)

            projection_log.append(
                {
                    "weak_index": weak_idx,
                    "strong_index": strong_idx,
                    "conflict_severity": float(conflict_severity),
                    "original_weight": float(weights[weak_idx]),
                    "adjusted_weight": float(adjusted_weights[weak_idx]),
                    "projection_magnitude": float(np.linalg.norm(projected_component)),
                }
            )
            processed.add(weak_idx)

    return projected, adjusted_weights


def _svd_rank_rule_entries(
    rule_entries: list[dict[str, Any]],
    vectors: np.ndarray,
    *,
    max_rules: int,
) -> list[str]:
    """Rank cluster-local rules after KKT conflict projection and SVD decoupling."""

    if not rule_entries:
        return []
    if len(rule_entries) == 1:
        return [str(rule_entries[0]["rule"]).strip()]

    weights = [float(entry.get("source_score", 0.0)) + 1.0 for entry in rule_entries]

    # Step 1: Apply KKT-inspired orthogonal projection to resolve conflicts
    projected_vectors, adjusted_weights = _project_conflicting_rules(
        vectors, weights, conflict_threshold=0.0
    )

    # Step 2: Filter out rules that are completely dominated (weight dropped too low)
    kept_indices = [
        i for i in range(len(rule_entries)) if adjusted_weights[i] > 0.1 * weights[i]
    ]
    if not kept_indices:
        kept_indices = list(range(len(rule_entries)))

    filtered_entries = [rule_entries[index] for index in kept_indices]
    filtered_vectors = projected_vectors[kept_indices]
    filtered_weights = [adjusted_weights[index] for index in kept_indices]

    if len(filtered_entries) == 1:
        return [str(filtered_entries[0]["rule"]).strip()]

    # Step 3: SVD-based ranking using projected vectors
    centered = filtered_vectors - np.mean(filtered_vectors, axis=0, keepdims=True)
    if centered.size == 0 or not np.any(centered):
        return [str(entry["rule"]).strip() for entry in filtered_entries[:max_rules]]

    try:
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [str(entry["rule"]).strip() for entry in filtered_entries[:max_rules]]

    principal_axis = right[0]
    projections = filtered_vectors @ principal_axis
    ranked_indices = sorted(
        range(len(filtered_entries)),
        key=lambda index: (abs(float(projections[index])), filtered_weights[index]),
        reverse=True,
    )
    selected_rules = [
        str(filtered_entries[index]["rule"]).strip()
        for index in ranked_indices[:max_rules]
    ]
    if singular_values.size > 0 and singular_values[0] <= 1e-8:
        return [str(entry["rule"]).strip() for entry in filtered_entries[:max_rules]]
    return selected_rules


def _build_rule_entries(
    critiques: list[Critique],
    *,
    r_k_by_client: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Flatten critiques into normalized rule entries with optional R_k utility scores."""

    r_k_by_client = r_k_by_client or {}
    rule_entries: list[dict[str, Any]] = []
    for critique in critiques:
        client_r_k = r_k_by_client.get(
            critique.client_id, 0.5
        )  # default 0.5 if not found
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
                    "r_k_utility": client_r_k,
                }
            )
    return rule_entries


def _summarize_cluster_local_rules(
    critiques: list[Critique],
    *,
    embedder,
    text_backend,
    eps: float,
    min_samples: int,
    max_rules: int,
    r_k_by_client: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Summarize cluster-local critiques into a small personalized rule list."""

    rule_entries = _build_rule_entries(critiques, r_k_by_client=r_k_by_client)
    if not rule_entries:
        return [], {"rule_count": 0}
    vectors = np.asarray(
        embedder.embed_texts([entry["rule"] for entry in rule_entries]),
        dtype=np.float64,
    )
    local_rules = _svd_rank_rule_entries(rule_entries, vectors, max_rules=max_rules * 2)
    ranked_clusters = _rank_clusters(
        _cluster_rule_entries(rule_entries, vectors, eps=eps, min_samples=min_samples)
    )
    if text_backend is not None and local_rules:
        local_rules, _ = summarize_rules_with_llm(
            local_rules, text_backend=text_backend, max_rules=max_rules
        )
    return local_rules[:max_rules], {
        "rule_count": len(rule_entries),
        "critique_cluster_count": len(ranked_clusters),
    }


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
            similarity = cosine_similarity(
                list(map(float, centroid)), list(map(float, entry.get("centroid", [])))
            )
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
            memory_weight = momentum_beta * float(previous.get("weight", 0.0)) + (
                1.0 - momentum_beta
            ) * float(cluster["score"])
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


def summarize_rules_with_llm(
    rules: list[str],
    *,
    text_backend,
    max_rules: int,
    max_retries: int = 3,
) -> tuple[list[str], str]:
    """Compress representative rules into a dense non-redundant summary.

    Args:
        rules: List of rules to summarize
        text_backend: LLM backend for generation
        max_rules: Maximum number of rules to keep
        max_retries: Maximum retry attempts for LLM call

    Returns:
        Tuple of (summarized_rules, memory_summary)
    """
    import logging

    from thesis_platform.core.llm_utils import safe_llm_generate

    logger = logging.getLogger(__name__)

    if not rules:
        return [], ""

    prompt = (
        "You are the server aggregation model for federated prompt optimization.\n"
        "Return JSON with keys `rules` and `memory_summary`.\n"
        "Compress the candidate guidance below into a dense, non-redundant summary.\n"
        f"Keep at most {max_rules} rules.\n\n"
        + "\n".join(f"- {rule}" for rule in rules)
    )

    # Use safe LLM generation with retry and fallback
    raw_text = safe_llm_generate(
        backend=text_backend,
        prompt=prompt,
        max_new_tokens=220,
        temperature=0.7,
        max_retries=max_retries,
        fallback_response="",  # Use fallback rules on failure
    )

    # If LLM returned empty (failed), use original rules as fallback
    if not raw_text or not raw_text.strip():
        logger.warning("LLM summarization failed, using original rules as fallback")
        return rules[:max_rules], ""

    parsed_rules, memory_summary = _parse_summary_rules(
        raw_text, fallback_rules=rules, max_rules=max_rules
    )

    # If parsing failed, use original rules
    if not parsed_rules:
        logger.warning("Failed to parse LLM response, using original rules as fallback")
        return rules[:max_rules], ""

    return parsed_rules, memory_summary


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
    prototype_feedbacks: list[PrototypeFeedback] | None = None,
    personalized_mix_ratio: float | None = None,
) -> tuple[PromptUpdate | None, dict[str, Any]]:
    """Cluster, rank, and summarize critique rules into a prompt update."""

    # Build R_k utility mapping from prototype feedbacks for attention-weighted aggregation
    r_k_by_client: dict[str, float] = {}
    if prototype_feedbacks:
        for pf in prototype_feedbacks:
            r_k_by_client[pf.client_id] = float(
                pf.weight
            )  # weight is R_k utility score

    rule_entries = _build_rule_entries(critiques, r_k_by_client=r_k_by_client)
    if not rule_entries:
        return None, memory or {"entries": []}

    vectors = np.asarray(
        embedder.embed_texts([entry["rule"] for entry in rule_entries]),
        dtype=np.float64,
    )
    ranked_clusters = _rank_clusters(
        _cluster_rule_entries(rule_entries, vectors, eps=eps, min_samples=min_samples)
    )

    memory_rules: list[str] = []
    updated_memory = memory or {"entries": []}
    if use_memory:
        ranked_clusters, updated_memory, memory_rules = _merge_memory(
            ranked_clusters,
            updated_memory,
            momentum_beta=momentum_beta,
        )

    representative_rules = [
        cluster["representative_rule"] for cluster in ranked_clusters[: max_rules * 2]
    ]
    final_rules = representative_rules[:max_rules]
    memory_summary = ""
    if text_backend is not None and final_rules:
        final_rules, memory_summary = summarize_rules_with_llm(
            final_rules, text_backend=text_backend, max_rules=max_rules
        )
    if not final_rules:
        return None, updated_memory

    compression_ratio = len(rule_entries) / max(len(final_rules), 1)
    client_cluster_map: dict[str, str] = {}
    prototype_clusters: list[dict[str, Any]] = []
    cluster_rules: dict[str, list[str]] = {}
    cluster_payload: dict[str, Any] = {}
    if prototype_feedbacks:
        client_cluster_map, prototype_clusters = _cluster_client_prototypes(
            prototype_feedbacks,
            eps=eps,
            min_samples=min_samples,
        )
        critiques_by_cluster: dict[str, list[Critique]] = {}
        for critique in critiques:
            cluster_id = client_cluster_map.get(
                critique.client_id, f"noise_{critique.client_id}"
            )
            critiques_by_cluster.setdefault(cluster_id, []).append(critique)
        for cluster_id, cluster_critiques in critiques_by_cluster.items():
            local_rules, local_meta = _summarize_cluster_local_rules(
                cluster_critiques,
                embedder=embedder,
                text_backend=text_backend,
                eps=eps,
                min_samples=min_samples,
                max_rules=max_rules,
                r_k_by_client=r_k_by_client,
            )
            cluster_rules[cluster_id] = local_rules
            cluster_payload[cluster_id] = local_meta

    prompt_update = PromptUpdate(
        update_id=f"dbscan_update_r{round_id}",
        round_id=round_id,
        rules=final_rules,
        summary=" ".join(final_rules),
        prompt_text=" ".join(final_rules),
        global_rules=final_rules,
        cluster_rules=cluster_rules,
        client_cluster_map=client_cluster_map,
        routing_state={
            "cluster_ids": sorted(cluster_rules.keys()),
            "personalized_mix_ratio": personalized_mix_ratio,
        },
        meta={
            "mode": "dbscan_attn_tsgdm" if use_memory else "dbscan_attn",
            "cluster_count": len(ranked_clusters),
            "clusters": ranked_clusters,
            "prototype_clusters": prototype_clusters,
            "cluster_payload": cluster_payload,
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

    rules = [
        rule.strip()
        for critique in critiques
        for rule in critique.rules
        if rule.strip()
    ]
    if not rules:
        return None
    final_rules, memory_summary = summarize_rules_with_llm(
        rules[: max_rules * 3], text_backend=text_backend, max_rules=max_rules
    )
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
