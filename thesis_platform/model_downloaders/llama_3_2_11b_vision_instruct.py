from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class Llama32_11BVisionInstructDownloader(HuggingFaceModelDownloader):
    """Download the Llama 3.2 11B Vision Instruct checkpoint used in FedTextGrad docs."""

    name = "llama_3_2_11b_vision_instruct"
    repo_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
    description = "Download the community-mirrored Llama 3.2 11B Vision Instruct checkpoint used in FedTextGrad docs."
    optional = True
    community_mirror_only = True
