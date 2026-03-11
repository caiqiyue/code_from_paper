from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Phi15Downloader(HuggingFaceModelDownloader):
    """Download the Phi-1.5 checkpoint used in GRADMM."""

    name = "phi_1_5"
    repo_id = "microsoft/phi-1_5"
    description = "Download the Phi-1.5 checkpoint used in GRADMM."
    parameter_count_billions = 1.5
    model_size_label = "1.5B"
