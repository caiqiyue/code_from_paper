"""Stage 2 bootstrap generation with the local LLaMA-2-7B checkpoint."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.io_utils import ensure_dir
from pretext_platform.core.resource_cleanup import release_gpu_memory
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
        "List of 3 diverse original text samples:\n"
        "Original Text Sample 1\n{0}\n"
        "Original Text Sample 2\n{1}\n"
        "Original Text Sample 3\n{2}\n"
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


def generate_bootstrapped_samples_vllm(prompt_list: list[str], model_path: Path, bootstrap_cfg: dict) -> list[str]:
    """Run vLLM generation for the bootstrap prompts and return raw outputs.
    Linux/macOS only - vLLM is not available on Windows.
    """

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


def generate_bootstrapped_samples_hf(
    prompt_list: list[str], model_path: Path, bootstrap_cfg: dict
) -> list[str]:
    """Run HuggingFace Transformers generation for bootstrap prompts.
    Cross-platform alternative to vLLM - works on Windows, Linux, and macOS.
    Loads model on CPU first, then moves to GPU for inference.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Check device configuration - support "cpu" for Windows compatibility
    device = bootstrap_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device == "cuda" and torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    # Load model with appropriate dtype for target device
    # On CPU: use float32 (fp16 not well supported on CPU)
    # On CUDA: use float16 for memory efficiency
    torch_dtype = torch.float16 if use_cuda else torch.float32
    # device_map='auto' causes segfaults on Windows with certain PyTorch/CUDA versions
    # Force use_safetensors=True for Windows compatibility (avoids segfault with large bin files)
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

        if use_cuda:
            model = model.to("cuda")

        temperature = float(bootstrap_cfg.get("temperature", 1.0))
        top_p = float(bootstrap_cfg.get("top_p", 1.0))
        max_tokens = int(bootstrap_cfg.get("max_tokens", 85))
        batch_size = int(bootstrap_cfg.get("batch_size", 2))

        outputs = []
        model.eval()

        with torch.no_grad():
            for i in range(0, len(prompt_list), batch_size):
                batch_prompts = prompt_list[i : i + batch_size]
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(bootstrap_cfg.get("max_model_len", 1000)) - max_tokens,
                )

                if use_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                for generated in generated_ids:
                    prompt_length = inputs["input_ids"].shape[1]
                    generated_text = tokenizer.decode(
                        generated[prompt_length:], skip_special_tokens=True
                    )
                    outputs.append(generated_text)

                if use_cuda:
                    torch.cuda.empty_cache()

        return outputs
    finally:
        release_gpu_memory(model)


def generate_bootstrapped_samples(
    prompt_list: list[str], model_path: Path, bootstrap_cfg: dict
) -> list[str]:
    """Run generation for the bootstrap prompts using the best available backend.

    On Linux/macOS with vLLM installed: uses vLLM for fast batched inference.
    On Windows or when vLLM is unavailable: falls back to HuggingFace Transformers.

    To explicitly choose a backend, set `bootstrap.generator_backend` in config:
    - "vllm": Use vLLM (Linux only)
    - "huggingface": Use HuggingFace Transformers (cross-platform)
    """

    backend = bootstrap_cfg.get("generator_backend", "auto")

    if backend == "vllm":
        return generate_bootstrapped_samples_vllm(prompt_list, model_path, bootstrap_cfg)
    elif backend == "huggingface":
        return generate_bootstrapped_samples_hf(prompt_list, model_path, bootstrap_cfg)
    else:  # "auto"
        # Try vLLM first, fall back to HuggingFace
        try:
            return generate_bootstrapped_samples_vllm(prompt_list, model_path, bootstrap_cfg)
        except ImportError:
            import platform

            if platform.system() == "Windows":
                print(
                    "vLLM not available on Windows, falling back to HuggingFace Transformers. "
                    "Note: This is slower than vLLM but produces equivalent results."
                )
            return generate_bootstrapped_samples_hf(prompt_list, model_path, bootstrap_cfg)


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

    bootstrap_model = str(bootstrap_cfg.get("generator_model", "llama2_7b"))
    if bootstrap_model != "llama2_7b":
        raise ValueError(
            "Stage 2 bootstrap only supports generator_model='llama2_7b' on the fixed Linux server."
        )
    model_path = model_paths.llama2_7b

    output_list = generator_fn(prompt_list, model_path, bootstrap_cfg)

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
