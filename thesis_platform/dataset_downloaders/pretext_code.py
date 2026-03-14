from __future__ import annotations

from .base import BaseDatasetDownloader
from .pretext_utils import PRETEXT_CODE_EVAL_SIZE, PRETEXT_CODE_TRAIN_SIZE, stage_pretext_private_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextCodeDownloader(BaseDatasetDownloader):
    """Approximate the PrE-Text Code dataset from technical Q&A pages."""

    name = "pretext_code"
    description = "Approximate the PrE-Text Code dataset from c4-en technical Q&A pages."
    formatter_name = "pretext_json"
    optional = True
    pretext_c4_category = "code"
    pretext_dataset_kind = "private_dataset"
    pretext_output_prefix = "code"

    def perform_download_raw(self, force: bool):
        payload = stage_pretext_private_raw(
            self,
            category="code",
            paper_dataset_name="Code",
            train_size=PRETEXT_CODE_TRAIN_SIZE,
            eval_size=PRETEXT_CODE_EVAL_SIZE,
            force=force,
            provenance_note=(
                "The paper's Code dataset comes from a user-partitioned coding/technical Q&A corpus that is not "
                "released with the repository. This downloader stages a deterministic C4 approximation built from "
                "technical Q&A domains such as Stack Overflow and related Stack Exchange sites."
            ),
        )
        payload["metadata"]["paper_target_counts"] = {
            "train": PRETEXT_CODE_TRAIN_SIZE,
            "eval": PRETEXT_CODE_EVAL_SIZE,
        }
        payload["metadata"]["paper_alignment"]["approximation"] = True
        return payload
