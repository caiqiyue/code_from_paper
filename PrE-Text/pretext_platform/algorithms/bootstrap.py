"""Stage 2 bootstrap generation with LLaMA 2."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.io_utils import ensure_dir
from pretext_platform.core.types import ModelPaths, StageSummary


def load_surviving_seed_texts(stage1_dir: Path, *, num_rounds: int) -> list[str]:
    """Load all surviving seed texts produced during Stage 1."""

    seed_texts: list[str] = []
    for round_idx in range(num_rounds):
        path = stage1_dir / f"surviving_text_it{round_idx}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 1 artifact: {path}")
        with path.open("r", encoding="utf-8") as handle:
            seed_texts.extend(json.load(handle))
    return seed_texts


def build_bootstrap_prompts(seed_texts: list[str], *, num_prompts: int, seed: int) -> list[str]:
    """Create few-shot prompts that ask LLaMA 2 to continue the synthetic corpus."""

    single_prompt = (
        "List of 6 diverse original text samples:\n"
        "Original Text Sample 1\n{0}\n"
        "Original Text Sample 2\n{1}\n"
        "Original Text Sample 3\n{2}\n"
        "Original Text Sample 4\n"
    )
    rng = random.Random(seed)
    prompt_list = []
    for _ in range(num_prompts):
        examples = rng.sample(seed_texts, 3)
        prompt_list.append(
            single_prompt.format(
                examples[0].replace("\n", " ").replace("\t", " "),
                examples[1].replace("\n", " ").replace("\t", " "),
                examples[2].replace("\n", " ").replace("\t", " "),
            )
        )
    return prompt_list


def generate_bootstrapped_samples(prompt_list: list[str], model_path: Path, bootstrap_cfg: dict) -> list[str]:
    """Run vLLM generation for the bootstrap prompts and return raw outputs."""

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        max_model_len=int(bootstrap_cfg.get("max_model_len", 1000)),
    )
    sampling_params = SamplingParams(
        temperature=float(bootstrap_cfg.get("temperature", 1.0)),
        top_p=float(bootstrap_cfg.get("top_p", 1.0)),
        max_tokens=int(bootstrap_cfg.get("max_tokens", 85)),
    )
    outputs = llm.generate(prompt_list, sampling_params)
    return [output.outputs[0].text for output in outputs]


def run_bootstrap_stage(
    config: ExperimentConfig,
    model_paths: ModelPaths,
    stage1_dir: Path,
    output_dir: Path,
    *,
    generator_fn: Callable[[list[str], Path, dict], list[str]] = generate_bootstrapped_samples,
) -> StageSummary:
    """Load Stage 1 seeds, bootstrap them with LLaMA 2, and save the outputs."""

    bootstrap_cfg = config.bootstrap
    output_dir = ensure_dir(output_dir)
    rounds = int(config.stage1.get("rounds", 11))
    seed_texts = load_surviving_seed_texts(stage1_dir, num_rounds=rounds)
    prompt_list = build_bootstrap_prompts(
        seed_texts,
        num_prompts=int(bootstrap_cfg.get("num_prompts", 50000)),
        seed=int(config.meta.get("seed", 42)),
    )
    output_list = generator_fn(prompt_list, model_paths.llama2_7b, bootstrap_cfg)

    output_path = output_dir / "llama7b_text_syn.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_list, handle, ensure_ascii=False)

    return StageSummary(
        stage_name="stage2",
        output_dir=output_dir,
        artifacts={
            "synthetic_corpus_path": str(output_path),
            "prompt_count": len(prompt_list),
        },
        metrics={
            "seed_text_count": len(seed_texts),
            "prompt_count": len(prompt_list),
            "generated_count": len(output_list),
        },
    )
