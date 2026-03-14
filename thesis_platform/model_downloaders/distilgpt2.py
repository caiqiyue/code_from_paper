from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class DistilGPT2Downloader(HuggingFaceModelDownloader):
    """Download DistilGPT2 for the PrE-Text downstream small-model evaluation."""

    name = "distilgpt2"
    repo_id = "distilgpt2"
    description = "Download DistilGPT2 for the PrE-Text downstream next-token prediction evaluation."
    optional = True
    parameter_count_billions = 0.082
    model_size_label = "82M"
