from __future__ import annotations

from .base import BaseDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextBioarxivDownloader(BaseDatasetDownloader):
    """Download the Bioarxiv dataset for PrE-Text experiments.

    This dataset is used in the original PrE-Text platform as an alternative
    to the paper's Jobs/Forums/Microblog/Code datasets. It contains text from
    bioRxiv/medRxiv preprints.
    """

    name = "bioarxiv"
    description = "BioRxiv/MedRxiv preprint dataset for PrE-Text experiments"
    formatter_name = "pretext_json"
    optional = True

    HF_DATASET_NAME = "allenai/c4"
    HF_CONFIG = "en"
    HF_SPLIT = "train"
    TARGET_SIZE = 11000

    BIOARXIV_HOST_FRAGMENTS = (
        "biorxiv.org",
        "medrxiv.org",
        "arxiv.org",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "plos.org",
        "peerj.com",
        "frontiersin.org",
    )
    BIOARXIV_PATH_FRAGMENTS = (
        "/content/",
        "/doi/",
        "/article",
        "/abstract",
    )

    def _matches_bioarxiv(self, url: str) -> bool:
        """Check if URL is from bioarxiv/medical sources."""
        url_lower = url.lower()
        for fragment in self.BIOARXIV_HOST_FRAGMENTS:
            if fragment in url_lower:
                return True
        for fragment in self.BIOARXIV_PATH_FRAGMENTS:
            if fragment in url_lower:
                return True
        return False

    def perform_download_raw(self, force: bool):
        """Download Bioarxiv dataset from C4-en filtered by academic/medical URLs."""

        from datasets import load_dataset
        from .common import ensure_dir, remove_path
        from .pretext_utils import PRETEXT_C4_RANDOM_SEED, write_jsonl

        target = self.raw_path()
        if target is None:
            raise ValueError(f"{self.name} must define a raw path.")

        remove_path(target)
        ensure_dir(target)

        print(f"Downloading C4-en and filtering for bioarxiv content...")
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
                if self._matches_bioarxiv(url):
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
            print(f"Downloaded and filtered {len(rows)} bioarxiv texts")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download/filter bioarxiv data from {self.HF_DATASET_NAME}. Error: {exc}"
            ) from exc

        return {
            "message": "Downloaded Bioarxiv dataset",
            "metadata": {
                "source_type": "huggingface_dataset",
                "source_dataset": self.HF_DATASET_NAME,
                "raw_format": "jsonl",
                "paper_alignment": {
                    "paper": "PrE-Text",
                    "dataset": "Bioarxiv",
                    "approximation": True,
                },
                "source_row_count": len(rows),
            },
        }
