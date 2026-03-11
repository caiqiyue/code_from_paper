from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class OPT350MDownloader(HuggingFaceModelDownloader):
    """Download the OPT-350M checkpoint used in GRADMM."""

    name = "opt_350m"
    repo_id = "facebook/opt-350m"
    description = "Download the OPT-350M checkpoint used in GRADMM."
    parameter_count_billions = 0.35
    model_size_label = "350M"
