from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class MetaLlama3_8BDownloader(HuggingFaceModelDownloader):
    """Download the Meta Llama 3 8B model."""

    name = "Meta-Llama-3-8B"
    repo_id = "meta-llama/Meta-Llama-3-8B"
    description = "Download the Meta Llama 3 8B model."
    optional = True
    parameter_count_billions = 8.0
    model_size_label = "8B"
