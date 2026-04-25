from __future__ import annotations

from .consistency import compute_consistency_score
from .contracts import GeneratedSampleRecord, Stage2SelectionResult
from .dedup import compute_duplicate_penalty
from .template_penalty import compute_template_penalty


def select_seed_aware_records(
    *,
    records: list[GeneratedSampleRecord],
    generated_vectors: list[list[float]],
    prompt_seed_vectors: list[list[list[float]]],
    selector_cfg: dict,
) -> Stage2SelectionResult:
    if not (len(records) == len(generated_vectors) == len(prompt_seed_vectors)):
        raise ValueError("records, generated_vectors, and prompt_seed_vectors must align.")

    raw_clean_count = sum(1 for record in records if record.baseline_text)
    target_count_mode = str(selector_cfg.get("target_count_mode", "")).strip().lower()
    if target_count_mode in {"match_eval_clean_count", "match_baseline_clean_count"}:
        target_count = raw_clean_count
    else:
        target_count = len(records)

    survivors: list[GeneratedSampleRecord] = []
    rejected: list[GeneratedSampleRecord] = []

    for record, generated_vector, seed_vectors in zip(records, generated_vectors, prompt_seed_vectors):
        record.consistency_score = compute_consistency_score(generated_vector, seed_vectors)
        if not record.baseline_text:
            record.rejected_reason = "baseline_clean_empty"
            rejected.append(record)
            continue
        if record.consistency_score < float(selector_cfg["consistency_threshold"]):
            record.rejected_reason = "low_consistency"
            rejected.append(record)
            continue
        record.template_penalty = compute_template_penalty(
            record.raw_text,
            record.prompt_text,
            record.seed_texts,
            min_words=int(selector_cfg["min_words"]),
            prompt_echo_ngram=int(selector_cfg["prompt_echo_ngram"]),
            unique_token_ratio_floor=float(selector_cfg["unique_token_ratio_floor"]),
        )
        survivors.append(record)

    survivors.sort(
        key=lambda record: (
            float(selector_cfg["w_consistency"]) * record.consistency_score
            - float(selector_cfg["w_template"]) * record.template_penalty
        ),
        reverse=True,
    )

    kept_vectors: list[list[float]] = []
    selected: list[GeneratedSampleRecord] = []
    for record in survivors:
        vector = generated_vectors[record.record_index]
        record.duplicate_penalty = compute_duplicate_penalty(vector, kept_vectors)
        if record.duplicate_penalty >= float(selector_cfg["duplicate_threshold"]):
            record.rejected_reason = "near_duplicate"
            rejected.append(record)
            continue
        record.final_score = (
            float(selector_cfg["w_consistency"]) * record.consistency_score
            - float(selector_cfg["w_template"]) * record.template_penalty
            - float(selector_cfg["w_duplicate"]) * record.duplicate_penalty
        )
        selected.append(record)
        kept_vectors.append(vector)
        if len(selected) >= target_count:
            break

    for record in survivors[len(selected) :]:
        if record not in rejected and record not in selected:
            record.rejected_reason = "trimmed_by_target_count"
            rejected.append(record)

    return Stage2SelectionResult(
        selected_records=selected,
        rejected_records=rejected,
        raw_clean_count=raw_clean_count,
        target_count=target_count,
    )
