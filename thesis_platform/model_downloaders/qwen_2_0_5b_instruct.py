from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Qwen20_5BDownloader(HuggingFaceModelDownloader):
    """Download the Qwen2-0.5B-Instruct checkpoint used for local LLM-based critique."""

    name = "qwen_2_0_5b_instruct"
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    description = "Download the Qwen2-0.5B-Instruct checkpoint for local LLM-based critique (fedtextgrad_qwen upgrade)."
    parameter_count_billions = 0.5
    model_size_label = "0.5B"
