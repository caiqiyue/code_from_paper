from __future__ import annotations

from .hf import HuggingFaceModelDownloader
from .registry import register_model_downloader


@register_model_downloader
class RobertaLargeDownloader(HuggingFaceModelDownloader):
    """Download the RoBERTa-large checkpoint used in DataInf."""

    name = "roberta_large"
    repo_id = "roberta-large"
    description = "Download the RoBERTa-large checkpoint used in DataInf."
