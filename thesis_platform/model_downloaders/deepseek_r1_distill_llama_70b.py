from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class DeepSeekR1DistillLlama70BDownloader(HuggingFaceModelDownloader):
    """Download the DeepSeek R1 Distill Llama 70B checkpoint mentioned in FedTextGrad."""

    name = "deepseek_r1_distill_llama_70b"
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    description = "Download the DeepSeek R1 Distill Llama 70B checkpoint mentioned in FedTextGrad."
    optional = True
