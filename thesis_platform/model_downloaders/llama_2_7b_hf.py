from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Llama2_7BDownloader(HuggingFaceModelDownloader):
    """Download the Llama 2 7B base model used by PrE-Text."""

    name = "llama_2_7b_hf"
    repo_id = "NousResearch/Llama-2-7b-hf"
    description = "Download the community-mirrored Llama 2 7B base checkpoint used by PrE-Text."
    optional = True
    community_mirror_only = True
    parameter_count_billions = 7.0
    model_size_label = "7B"
