from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class MetaLlama2_7BChatHfDownloader(HuggingFaceModelDownloader):
    """Download the Meta Llama 2 7B chat model."""

    name = "Meta-Llama-2-7b-chat-hf"
    repo_id = "meta-llama/Meta-Llama-2-7b-chat-hf"
    description = "Download the Meta Llama 2 7B chat model."
    optional = True
    parameter_count_billions = 7.0
    model_size_label = "7B"
