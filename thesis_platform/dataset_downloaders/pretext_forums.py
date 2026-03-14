from __future__ import annotations

from .base import BaseDatasetDownloader
from .pretext_utils import PRETEXT_PRIVATE_EVAL_SIZE, PRETEXT_PRIVATE_TRAIN_SIZE, stage_pretext_private_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextForumsDownloader(BaseDatasetDownloader):
    """Approximate the PrE-Text Forums dataset from C4 forum pages."""

    name = "pretext_forums"
    description = "Approximate the PrE-Text Forums dataset from c4-en forum/community pages."
    formatter_name = "pretext_json"
    optional = True
    pretext_c4_category = "forums"
    pretext_dataset_kind = "private_dataset"
    pretext_output_prefix = "forums"

    def perform_download_raw(self, force: bool):
        payload = stage_pretext_private_raw(
            self,
            category="forums",
            paper_dataset_name="Forums",
            train_size=PRETEXT_PRIVATE_TRAIN_SIZE,
            eval_size=PRETEXT_PRIVATE_EVAL_SIZE,
            force=force,
            provenance_note=(
                "The paper describes Forums as 11k C4 rows from forum websites. This downloader recreates that "
                "construction with deterministic forum/community URL heuristics."
            ),
        )
        payload["metadata"]["paper_target_counts"] = {
            "train": PRETEXT_PRIVATE_TRAIN_SIZE,
            "eval": PRETEXT_PRIVATE_EVAL_SIZE,
        }
        return payload
