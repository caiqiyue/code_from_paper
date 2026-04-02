from __future__ import annotations

from .base import BaseDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextCongressionalDownloader(BaseDatasetDownloader):
    """Format manually downloaded Congressional records for PrE-Text experiments.

    Raw data: monthly JSON files from Canadian parliamentary records.
    Each record contains: url, date_str, title, speaker, data (speech text), chamber, country

    This downloader expects raw JSON files at:
        datasets/congressional/raw/congressional_data_YYYY-MM.json
    (already manually downloaded to datasets/congressional/raw/)
    """

    name = "congressional"
    description = "Congressional records dataset for PrE-Text experiments"
    formatter_name = "congressional"
    optional = True
    congressional_output_prefix = "congressional"

    def perform_download_raw(self, force: bool):
        """Validate that manually downloaded congressional raw data exists.

        No download needed - raw data is already present from manual setup.
        """

        target = self.raw_path()
        if target is None:
            raise ValueError(f"{self.name} must define a raw path.")

        if not target.exists():
            raise FileNotFoundError(
                f"Congressional raw data not found at {target}. "
                "Please manually download congressional_data_YYYY-MM.json files to this directory."
            )

        # Count monthly files present
        monthly_files = list(target.glob("congressional_data_*.json"))
        if not monthly_files:
            raise FileNotFoundError(
                f"No congressional_data_*.json files found in {target}. "
                "Please ensure raw data files are present."
            )

        return {
            "message": f"Congressional raw data validated. Found {len(monthly_files)} monthly files.",
            "metadata": {
                "source_type": "manual_download",
                "source_location": str(target),
                "raw_format": "json",
                "monthly_file_count": len(monthly_files),
                "paper_alignment": {
                    "paper": "PrE-Text",
                    "dataset": "Congressional",
                    "approximation": False,
                    "note": "Canadian parliamentary records from ourcommons.ca",
                },
            },
        }
