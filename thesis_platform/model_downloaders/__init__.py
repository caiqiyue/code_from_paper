"""Model download registry and controller."""

from . import (
    all_minilm_l6_v2,
    deepseek_r1_distill_llama_70b,
    distilgpt2,
    flan_t5_3b,
    llama_2_13b_chat_hf,
    llama_2_7b_hf,
    llama_3_1_405b_instruct,
    llama_3_1_8b_instruct,
    llama_3_2_11b_vision_instruct,
    llama_3_2_3b_instruct,
    opt_1_3b,
    opt_125m,
    opt_350m,
    phi_1_5,
    qwen_2_0_5b_instruct,
    roberta_large,
    stable_diffusion_v1_5,
)
from .controller import download_models, list_model_downloaders, resolve_model_downloaders
from .registry import create_model_downloader, get_registered_model_names, resolve_model_names

__all__ = [
    "create_model_downloader",
    "download_models",
    "get_registered_model_names",
    "list_model_downloaders",
    "resolve_model_downloaders",
    "resolve_model_names",
]
