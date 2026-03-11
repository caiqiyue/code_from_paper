from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Llama32_3BInstructDownloader(HuggingFaceModelDownloader):
    """Download the prompt-transfer target Llama 3.2 3B instruct checkpoint."""

    name = "llama_3_2_3b_instruct"
    repo_id = "unsloth/Llama-3.2-3B-Instruct"
    description = "Download the community-mirrored prompt-transfer target Llama 3.2 3B instruct checkpoint."
    optional = True
    community_mirror_only = True
