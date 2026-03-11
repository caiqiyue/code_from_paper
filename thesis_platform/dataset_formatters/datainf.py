from __future__ import annotations

from thesis_platform.dataset_downloaders.datainf_generation import (
    datainf_formatted_output_paths,
    move_datainf_outputs_into_dataset,
)

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class DataInfGeneratedFormatter(BaseDatasetFormatter):
    """Move DataInf-generated artifacts into dataset-specific formatted directories."""

    name = "datainf"

    def required_paths(self, downloader):
        return list(datainf_formatted_output_paths(downloader))

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        move_result = move_datainf_outputs_into_dataset(downloader, force=force)
        return {
            "message": "Generated DataInf dataset artifacts are ready in the formatted directory.",
            "metadata": {
                "formatted_format": "huggingface_save_to_disk",
                "provenance_note": "Formatted artifacts come from the shared DataInf generation script and are moved into dataset-specific directories.",
                **move_result,
            },
        }
