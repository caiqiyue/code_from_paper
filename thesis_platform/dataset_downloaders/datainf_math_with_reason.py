from __future__ import annotations

from .datainf_generation import DataInfGeneratedDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class DataInfMathWithReasonDownloader(DataInfGeneratedDatasetDownloader):
    """Validate the generated reasoning math dataset from DataInf."""

    name = "datainf_math_with_reason"
    description = "Generate the DataInf math-with-reason dataset into thesis_platform/datasets."
    dataset_key = "datainf_math_with_reason"
