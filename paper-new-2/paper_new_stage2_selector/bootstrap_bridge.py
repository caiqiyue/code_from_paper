from __future__ import annotations

import random

from .contracts import BootstrapPromptRecord, GeneratedSampleRecord
from .corpus_loader import extract_baseline_training_text

PROMPT_TEMPLATE = (
    "List of 3 diverse original text samples:\n"
    "Original Text Sample 1\n{0}\n"
    "Original Text Sample 2\n{1}\n"
    "Original Text Sample 3\n{2}\n"
)


def build_bootstrap_prompt_records(
    seed_texts: list[str],
    *,
    num_prompts: int,
    seed: int,
) -> list[BootstrapPromptRecord]:
    """Mirror PrE-Text bootstrap prompt construction while preserving seed metadata."""

    if not seed_texts:
        raise ValueError("Stage 2 bootstrap requires at least 1 seed text.")

    rng = random.Random(seed)
    records: list[BootstrapPromptRecord] = []
    for prompt_index in range(int(num_prompts)):
        if len(seed_texts) >= 3:
            examples = rng.sample(seed_texts, 3)
        else:
            examples = [rng.choice(seed_texts) for _ in range(3)]
        prompt_text = PROMPT_TEMPLATE.format(
            examples[0].replace("\n", " ").replace("\t", " "),
            examples[1].replace("\n", " ").replace("\t", " "),
            examples[2].replace("\n", " ").replace("\t", " "),
        )
        records.append(
            BootstrapPromptRecord(
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                seed_texts=list(examples),
            )
        )
    return records


def attach_generated_outputs(
    prompt_records: list[BootstrapPromptRecord],
    outputs: list[str],
) -> list[GeneratedSampleRecord]:
    """Attach raw bootstrap outputs back to their originating prompt/seed metadata."""

    if len(prompt_records) != len(outputs):
        raise ValueError("prompt_records and outputs must have the same length.")

    attached: list[GeneratedSampleRecord] = []
    for record_index, (prompt_record, raw_text) in enumerate(zip(prompt_records, outputs)):
        attached.append(
            GeneratedSampleRecord(
                record_index=record_index,
                prompt_index=prompt_record.prompt_index,
                prompt_text=prompt_record.prompt_text,
                seed_texts=list(prompt_record.seed_texts),
                raw_text=str(raw_text),
                baseline_text=extract_baseline_training_text(str(raw_text)),
            )
        )
    return attached
