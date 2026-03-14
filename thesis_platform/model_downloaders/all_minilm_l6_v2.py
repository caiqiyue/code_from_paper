from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class AllMiniLML6V2Downloader(HuggingFaceModelDownloader):
    """Download the MiniLM sentence encoder used by PrE-Text."""

    name = "all_minilm_l6_v2"
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    description = "Download the all-MiniLM-L6-v2 sentence embedding model used by PrE-Text."
    parameter_count_billions = 0.0227
    model_size_label = "22.7M"
