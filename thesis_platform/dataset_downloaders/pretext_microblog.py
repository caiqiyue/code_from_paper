from __future__ import annotations

from .base import BaseDatasetDownloader
from .pretext_utils import PRETEXT_PRIVATE_EVAL_SIZE, PRETEXT_PRIVATE_TRAIN_SIZE, stage_pretext_private_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextMicroblogDownloader(BaseDatasetDownloader):
    """Approximate the PrE-Text Microblog dataset from C4 microblog pages."""

    name = "pretext_microblog"
    description = "Approximate the PrE-Text Microblog dataset from c4-en microblog sources."
    formatter_name = "pretext_json"
    optional = True
    pretext_c4_category = "microblog"
    pretext_dataset_kind = "private_dataset"
    pretext_output_prefix = "microblog"

    def perform_download_raw(self, force: bool):
        payload = stage_pretext_private_raw(
            self,
            category="microblog",
            paper_dataset_name="Microblog",
            train_size=PRETEXT_PRIVATE_TRAIN_SIZE,
            eval_size=PRETEXT_PRIVATE_EVAL_SIZE,
            force=force,
            provenance_note=(
                "The paper describes Microblog as 11k C4 rows from microblogging sites. This downloader recreates "
                "that pool with deterministic microblog URL heuristics over the C4 English train split."
            ),
        )
        payload["metadata"]["paper_target_counts"] = {
            "train": PRETEXT_PRIVATE_TRAIN_SIZE,
            "eval": PRETEXT_PRIVATE_EVAL_SIZE,
        }
        return payload
