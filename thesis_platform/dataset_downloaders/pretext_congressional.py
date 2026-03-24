from __future__ import annotations

from .base import BaseDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextCongressionalDownloader(BaseDatasetDownloader):
    """Download the Congressional dataset for PrE-Text experiments.

    This dataset is used in the original PrE-Text platform as an alternative
    to the paper's Jobs/Forums/Microblog/Code datasets. It contains text from
    congressional records.
    """

    name = "congressional"
    description = "Congressional records dataset for PrE-Text experiments"
    formatter_name = "pretext_json"
    optional = True

    HF_DATASET_NAME = "allenai/c4"
    HF_CONFIG = "en"
    HF_SPLIT = "train"
    TARGET_SIZE = 11000

    CONGRESSIONAL_HOST_FRAGMENTS = (
        "congress.gov",
        "senate.gov",
        "house.gov",
        "govtrack.us",
        "cspan.org",
    )
    CONGRESSIONAL_PATH_FRAGMENTS = (
        "/congress/",
        "/legislative",
        "/hearing",
        "/transcript",
    )

    def _matches_congressional(self, url: str) -> bool:
        """Check if URL is from congressional sources."""
        url_lower = url.lower()
        for fragment in self.CONGRESSIONAL_HOST_FRAGMENTS:
            if fragment in url_lower:
                return True
        for fragment in self.CONGRESSIONAL_PATH_FRAGMENTS:
            if fragment in url_lower:
                return True
        return False

    def perform_download_raw(self, force: bool):
        """Download Congressional dataset from C4-en filtered by congressional URLs."""

        from datasets import load_dataset
        from .common import ensure_dir, remove_path
        from .pretext_utils import PRETEXT_C4_RANDOM_SEED, write_jsonl

        target = self.raw_path()
        if target is None:
            raise ValueError(f"{self.name} must define a raw path.")

        remove_path(target)
        ensure_dir(target)

        print(f"Downloading C4-en and filtering for congressional content...")
        try:
            dataset = load_dataset(
                self.HF_DATASET_NAME,
                self.HF_CONFIG,
                split=self.HF_SPLIT,
                streaming=False,
                download_mode="reuse_cache_if_exists",
            )

            rows = []
            for item in dataset:
                url = str(item.get("url", ""))
                if self._matches_congressional(url):
                    text = str(item.get("text", ""))
                    if len(text.split()) >= 20:
                        rows.append({"text": text, "url": url})
                        if len(rows) >= self.TARGET_SIZE:
                            break

            # Random sample to TARGET_SIZE if we got more
            import random
            if len(rows) > self.TARGET_SIZE:
                random.seed(PRETEXT_C4_RANDOM_SEED)
                rows = random.sample(rows, self.TARGET_SIZE)

            write_jsonl(target / "train.jsonl", rows[:10000])
            write_jsonl(target / "eval.jsonl", rows[10000:11000])
            print(f"Downloaded and filtered {len(rows)} congressional texts")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download/filter congressional data from {self.HF_DATASET_NAME}. Error: {exc}"
            ) from exc

        return {
            "message": "Downloaded Congressional dataset",
            "metadata": {
                "source_type": "huggingface_dataset",
                "source_dataset": self.HF_DATASET_NAME,
                "raw_format": "jsonl",
                "paper_alignment": {
                    "paper": "PrE-Text",
                    "dataset": "Congressional",
                    "approximation": True,
                },
                "source_row_count": len(rows),
            },
        }
