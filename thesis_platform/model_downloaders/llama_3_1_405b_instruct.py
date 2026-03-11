from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Llama31_405BInstructDownloader(HuggingFaceModelDownloader):
    """Download the optional Llama 3.1 405B instruct checkpoint mentioned in FedTextGrad."""

    name = "llama_3_1_405b_instruct"
    repo_id = "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic"
    description = "Download the optional community-mirrored Llama 3.1 405B FP8 dynamic checkpoint mentioned in FedTextGrad."
    optional = True
    community_mirror_only = True
    parameter_count_billions = 405.0
    model_size_label = "405B"
