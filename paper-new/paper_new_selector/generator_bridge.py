from __future__ import annotations

import sys
from pathlib import Path

from .contracts import CandidateGeneratorHandle, GeneratorContract
from .thesis_bridge import load_yaml_config, resolve_repo_root, resolve_worktree_root


def build_candidate_generator(config_path: str | Path) -> CandidateGeneratorHandle:
    """Build the only legal fixed Stage 1 candidate generator."""

    config = load_yaml_config(config_path)
    repo_root = resolve_worktree_root()
    resource_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(resource_root) not in sys.path:
        sys.path.insert(0, str(resource_root))

    from thesis_platform.adapters.generators.pretext_prompt_generator import PretextPromptLLMGenerator
    from thesis_platform.models.backends import build_text_backend

    generator_cfg = dict(config["generator"])
    llm_cfg = dict(config.get("llm", {}).get("generator", {}))
    if not llm_cfg:
        raise ValueError("llm.generator must be configured for a real local model backend.")
    llm_engine = str(llm_cfg.get("engine", "")).strip().lower()
    if llm_engine not in {"transformers", "vllm"}:
        raise ValueError(
            f"llm.generator.engine must be 'transformers' or 'vllm', got '{llm_engine or '<empty>'}'."
        )
    model_name_or_path = str(llm_cfg.get("model_name_or_path", "")).strip()
    if not model_name_or_path:
        raise ValueError("llm.generator.model_name_or_path must be configured.")
    text_backend = build_text_backend({**llm_cfg, "role": "generator"}, repo_root=resource_root)
    generator = PretextPromptLLMGenerator(generator_cfg, resource_root)
    contract = GeneratorContract(
        backend=str(generator_cfg["backend"]),
        source=str(generator_cfg["source"]),
        initial_prompt=str(generator_cfg["initial_prompt"]),
        candidate_count=int(generator_cfg["candidate_count"]),
        generated_per_round=int(generator_cfg["generated_per_round"]),
        exemplars_per_prompt=int(generator_cfg["exemplars_per_prompt"]),
        max_new_tokens=int(generator_cfg["max_new_tokens"]),
        max_prompt_chars=int(generator_cfg.get("max_prompt_chars", 0)),
        max_exemplar_chars=int(generator_cfg.get("max_exemplar_chars", 0)),
        llm_backend=llm_engine,
    ).to_dict()
    return CandidateGeneratorHandle(
        generator=generator,
        text_backend=text_backend,
        contract=contract,
        repo_root=repo_root,
        resource_root=resource_root,
    )
