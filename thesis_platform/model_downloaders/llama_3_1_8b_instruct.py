from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Llama31_8BInstructDownloader(HuggingFaceModelDownloader):
    """Download the main FedTextGrad local Llama 3.1 8B instruct checkpoint."""

    name = "llama_3_1_8b_instruct"
    repo_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
    description = "Download the community-mirrored FedTextGrad local Llama 3.1 8B instruct checkpoint."
    community_mirror_only = True
