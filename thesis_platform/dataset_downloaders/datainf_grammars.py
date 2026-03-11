from __future__ import annotations

from .datainf_generation import DataInfGeneratedDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class DataInfGrammarsDownloader(DataInfGeneratedDatasetDownloader):
    """Validate the generated sentence transformation dataset from DataInf."""

    name = "datainf_grammars"
    description = "Generate the DataInf sentence transformation dataset into thesis_platform/datasets."
    dataset_key = "datainf_grammars"
