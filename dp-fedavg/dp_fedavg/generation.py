from __future__ import annotations

from typing import Any

from thesis_platform.models.backends import build_text_backend

from .paths import resolve_repo_root


def build_generation_backend(llm_cfg: dict[str, Any]):
    if str(llm_cfg.get("engine", "")).strip().lower() != "vllm":
        raise ValueError("dp-fedavg generation requires llm.engine='vllm'.")
    return build_text_backend({**llm_cfg, "role": "generator"}, repo_root=resolve_repo_root())


def build_generation_prompts(
    partitions,
    *,
    generated_per_round: int,
    exemplars_per_prompt: int,
    system_prompt: str,
) -> list[str]:
    prompts: list[str] = []
    for partition in partitions:
        exemplar_texts = [
            sample.render_text().strip()
            for sample in partition.samples[: max(1, exemplars_per_prompt)]
            if sample.render_text().strip()
        ]
        if not exemplar_texts:
            continue
        prompt = (
            f"{system_prompt}\n\n"
            f"Client: {partition.client_id}\n"
            f"Exemplars:\n- " + "\n- ".join(exemplar_texts)
        ).strip()
        for _ in range(max(1, generated_per_round)):
            prompts.append(prompt)
    return prompts


def generate_synthetic_texts(
    backend,
    *,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    if not prompts:
        return []
    if hasattr(backend, "generate_batch"):
        return backend.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    return [
        backend.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        for prompt in prompts
    ]
