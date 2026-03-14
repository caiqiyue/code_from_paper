from __future__ import annotations

from .base import BaseDatasetDownloader
from .pretext_utils import PRETEXT_PRIVATE_EVAL_SIZE, PRETEXT_PRIVATE_TRAIN_SIZE, stage_pretext_private_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextJobsDownloader(BaseDatasetDownloader):
    """Approximate the PrE-Text Jobs dataset from C4 job-site pages."""

    name = "pretext_jobs"
    description = "Approximate the PrE-Text Jobs dataset from c4-en job-site pages."
    formatter_name = "pretext_json"
    optional = True
    pretext_c4_category = "jobs"
    pretext_dataset_kind = "private_dataset"
    pretext_output_prefix = "jobs"

    def perform_download_raw(self, force: bool):
        payload = stage_pretext_private_raw(
            self,
            category="jobs",
            paper_dataset_name="Jobs",
            train_size=PRETEXT_PRIVATE_TRAIN_SIZE,
            eval_size=PRETEXT_PRIVATE_EVAL_SIZE,
            force=force,
            provenance_note=(
                "The paper describes Jobs as the first 11k C4 rows from jobs sites. This downloader recreates that "
                "idea with deterministic URL heuristics and writes PrE-Text-compatible train/eval JSON files."
            ),
        )
        payload["metadata"]["paper_target_counts"] = {
            "train": PRETEXT_PRIVATE_TRAIN_SIZE,
            "eval": PRETEXT_PRIVATE_EVAL_SIZE,
        }
        return payload
