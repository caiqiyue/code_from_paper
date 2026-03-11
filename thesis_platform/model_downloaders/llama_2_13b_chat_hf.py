from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Llama2_13BChatDownloader(HuggingFaceModelDownloader):
    """Download the gated Llama 2 13B chat checkpoint used in DataInf."""

    name = "llama_2_13b_chat_hf"
    repo_id = "NousResearch/Llama-2-13b-chat-hf"
    description = "Download the community-mirrored Llama 2 13B chat checkpoint used in DataInf."
    optional = True
    community_mirror_only = True
    parameter_count_billions = 13.0
    model_size_label = "13B"
