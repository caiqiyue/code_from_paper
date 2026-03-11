from __future__ import annotations

from .datainf_generation import DataInfGeneratedDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class DataInfMathWithoutReasonDownloader(DataInfGeneratedDatasetDownloader):
    """Validate the generated no-reasoning math dataset from DataInf."""

    name = "datainf_math_without_reason"
    description = "Generate the DataInf math-without-reason dataset into thesis_platform/datasets."
    dataset_key = "datainf_math_without_reason"
