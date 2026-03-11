from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class StableDiffusionV15Downloader(HuggingFaceModelDownloader):
    """Download the Stable Diffusion v1.5 checkpoint used in DataInf."""

    name = "stable_diffusion_v1_5"
    repo_id = "runwayml/stable-diffusion-v1-5"
    description = "Download the Stable Diffusion v1.5 checkpoint used in DataInf."
