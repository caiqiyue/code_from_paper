from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class OPT125MDownloader(HuggingFaceModelDownloader):
    """Download the OPT-125M checkpoint used in GRADMM."""

    name = "opt_125m"
    repo_id = "facebook/opt-125m"
    description = "Download the OPT-125M checkpoint used in GRADMM."
    parameter_count_billions = 0.125
    model_size_label = "125M"
