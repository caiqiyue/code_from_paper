from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class OPT13BDownloader(HuggingFaceModelDownloader):
    """Download the OPT-1.3B checkpoint used in GRADMM appendix experiments."""

    name = "opt_1_3b"
    repo_id = "facebook/opt-1.3b"
    description = "Download the OPT-1.3B checkpoint used in GRADMM appendix experiments."
    optional = True
    parameter_count_billions = 1.3
    model_size_label = "1.3B"
