from __future__ import annotations

from pathlib import Path

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.types import ModelPaths


def _resolve_model_dir(config: ExperimentConfig, value: str | None, default_dir: str) -> Path:
    """Resolve one logical model path to a concrete local directory."""

    model_root = config.model_root()
    if value:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (config.repo_root() / candidate).resolve()
    return (model_root / default_dir).resolve()


def resolve_model_paths(config: ExperimentConfig) -> ModelPaths:
    """Resolve all logical model names to local paths."""

    models_cfg = config.models
    checkpoint_value = models_cfg.get("c4_checkpoint_path")
    c4_checkpoint = config.resolve_path(checkpoint_value) if checkpoint_value not in (None, "") else None
    return ModelPaths(
        minilm=_resolve_model_dir(config, models_cfg.get("minilm_path"), "all_minilm_l6_v2"),
        roberta_large=_resolve_model_dir(config, models_cfg.get("roberta_large_path"), "roberta_large"),
        llama2_7b=_resolve_model_dir(config, models_cfg.get("llama2_7b_path"), "llama_2_7b_hf"),
        llama_3_2_3b_instruct=_resolve_model_dir(
            config, models_cfg.get("llama_3_2_3b_instruct_path"), "llama_3_2_3b_instruct"
        ),
        llama_3_1_8b_instruct=_resolve_model_dir(
            config, models_cfg.get("llama_3_1_8b_instruct_path"), "llama_3_1_8b_instruct"
        ),
        meta_llama_2_7b_chat=_resolve_model_dir(
            config, models_cfg.get("meta_llama_2_7b_chat_path"), "Meta-Llama-2-7b-chat-hf"
        ),
        distilgpt2=_resolve_model_dir(config, models_cfg.get("distilgpt2_path"), "distilgpt2"),
        gpt2_xl=_resolve_model_dir(config, models_cfg.get("gpt2_xl_path"), "gpt2_xl"),
        c4_checkpoint=c4_checkpoint,
    )
