from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class FlanT53BDownloader(HuggingFaceModelDownloader):
    """Download the approximate 3B FLAN-T5 checkpoint used by the DP-Prompt baseline."""

    name = "flan_t5_3b"
    repo_id = "google/flan-t5-xl"
    description = (
        "Download FLAN-T5-XL, the public approximately-3B checkpoint aligned with the paper's flan-t5-3b baseline."
    )
    optional = True
    parameter_count_billions = 3.0
    model_size_label = "3B"
