from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    tqdm = None

from thesis_platform.core.io_utils import ensure_dir, write_json, write_jsonl

from .common import datasets_root, move_path, remove_path, utc_timestamp

# HuggingFace token for authenticated downloads (improves stability, avoids rate limits)
# Set via HF_TOKEN environment variable or pass directly
HF_TOKEN = os.environ.get("HF_TOKEN", None)

PRETEXT_C4_SOURCE_DATASET = "allenai/c4"
PRETEXT_C4_SOURCE_CONFIG = "en"
PRETEXT_C4_SOURCE_SPLIT = "train"
PRETEXT_C4_RANDOM_SEED = 20240629
PRETEXT_MIN_WORDS = 20

PRETEXT_JOBS_POOL_SIZE = 11_000
PRETEXT_FORUMS_POOL_SIZE = 11_000
PRETEXT_MICROBLOG_POOL_SIZE = 11_000
PRETEXT_CODE_POOL_SIZE = 22_000
PRETEXT_INITIALIZATION_POOL_SIZE = 87_000

PRETEXT_PRIVATE_TRAIN_SIZE = 10_000
PRETEXT_PRIVATE_EVAL_SIZE = 1_000
PRETEXT_CODE_TRAIN_SIZE = 20_000
PRETEXT_CODE_EVAL_SIZE = 2_000

_PRETEXT_C4_TARGETS = {
    "jobs": PRETEXT_JOBS_POOL_SIZE,
    "forums": PRETEXT_FORUMS_POOL_SIZE,
    "microblog": PRETEXT_MICROBLOG_POOL_SIZE,
    "code": PRETEXT_CODE_POOL_SIZE,
    "initialization": PRETEXT_INITIALIZATION_POOL_SIZE,
}

_JOB_HOST_FRAGMENTS = (
    "indeed.",
    "glassdoor.",
    "monster.",
    "ziprecruiter.",
    "jobvite.",
    "myworkdayjobs.",
    "greenhouse.",
    "lever.co",
)
_JOB_PATH_FRAGMENTS = (
    "/jobs",
    "/careers",
    "/career",
    "/vacancies",
    "/vacancy",
    "/hiring",
    "/recruit",
)
_FORUM_HOST_FRAGMENTS = (
    "forum.",
    "forums.",
    "community.",
    "communities.",
    "discourse.",
)
_FORUM_PATH_FRAGMENTS = (
    "/forum",
    "/forums",
    "/board",
    "/boards",
    "/thread",
    "/threads",
    "/community",
)
_MICROBLOG_HOST_FRAGMENTS = (
    "twitter.com",
    "x.com",
    "mastodon",
    "weibo.com",
    "tumblr.com",
    "micro.blog",
)
_MICROBLOG_PATH_FRAGMENTS = (
    "/status/",
    "/statuses/",
    "/posts/",
    "/tweet",
    "/tweets",
)
_CODE_HOST_FRAGMENTS = (
    "stackoverflow.com",
    "stackexchange.com",
    "superuser.com",
    "serverfault.com",
    "askubuntu.com",
    "mathoverflow.net",
)
_CODE_PATH_FRAGMENTS = (
    "/questions/",
    "/q/",
    "/answers/",
)


def pretext_c4_cache_root() -> Path:
    """Return the shared C4 cache root used by PrE-Text-aligned dataset downloaders."""

    return datasets_root() / "_pretext_c4_cache"


def _matches_any(value: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in value for fragment in fragments)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _valid_text(text: str) -> bool:
    return len(text.split()) >= PRETEXT_MIN_WORDS


def classify_c4_url(url: str) -> str | None:
    """Map one C4 URL to the closest paper-aligned PrE-Text domain bucket."""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    host_and_path = f"{host}{path}"
    if _matches_any(host, _CODE_HOST_FRAGMENTS) or _matches_any(path, _CODE_PATH_FRAGMENTS):
        return "code"
    if _matches_any(host, _MICROBLOG_HOST_FRAGMENTS) or _matches_any(path, _MICROBLOG_PATH_FRAGMENTS):
        return "microblog"
    if _matches_any(host, _JOB_HOST_FRAGMENTS) or _matches_any(host_and_path, _JOB_PATH_FRAGMENTS):
        return "jobs"
    if _matches_any(host, _FORUM_HOST_FRAGMENTS) or _matches_any(host_and_path, _FORUM_PATH_FRAGMENTS):
        return "forums"
    return None


def _cache_file(category: str) -> Path:
    return pretext_c4_cache_root() / f"{category}.jsonl"


def _normalize_required_categories(required_categories: Iterable[str] | None) -> list[str]:
    if required_categories is None:
        categories = list(_PRETEXT_C4_TARGETS)
    else:
        categories = list(dict.fromkeys(required_categories))
    unknown = [category for category in categories if category not in _PRETEXT_C4_TARGETS]
    if unknown:
        raise ValueError(f"Unknown PrE-Text C4 cache categories: {', '.join(sorted(unknown))}")
    return categories


def _format_target_summary(categories: list[str], counts: dict[str, int], targets: dict[str, int]) -> str:
    return ", ".join(f"{category}={counts[category]}/{targets[category]}" for category in categories)


def _write_pretext_cache_metadata(
    cache_root: Path,
    *,
    required_categories: list[str],
    built_categories: list[str],
) -> None:
    """Persist cache metadata after one full or partial cache build."""

    write_json(
        cache_root / "metadata.json",
        {
            "source_dataset": PRETEXT_C4_SOURCE_DATASET,
            "source_config": PRETEXT_C4_SOURCE_CONFIG,
            "source_split": PRETEXT_C4_SOURCE_SPLIT,
            "generated_at": utc_timestamp(),
            "built_categories": built_categories,
            "available_categories": sorted(path.stem for path in cache_root.glob("*.jsonl")),
            "targets": {category: _PRETEXT_C4_TARGETS[category] for category in required_categories},
            "min_words": PRETEXT_MIN_WORDS,
            "seed": PRETEXT_C4_RANDOM_SEED,
            "heuristic_note": (
                "This cache approximates the PrE-Text paper datasets with URL-domain heuristics because the "
                "original curated dataset files are not bundled with the paper repository."
            ),
        },
    )


def ensure_pretext_c4_cache(
    required_categories: Iterable[str] | None = None,
    force: bool = False,
) -> Path:
    """Build or reuse only the requested PrE-Text C4 cache categories."""

    required = _normalize_required_categories(required_categories)
    cache_root = pretext_c4_cache_root()
    ensure_dir(cache_root)
    cached_categories = [category for category in required if _cache_file(category).exists() and not force]
    missing_categories = [category for category in required if category not in cached_categories]
    if not missing_categories:
        print(
            "PrE-Text C4 cache | reusing cached categories: "
            + ", ".join(required)
        )
        return cache_root

    from datasets import load_dataset

    # Download ALL of C4-en once (not streaming), then iterate from local cache.
    # This avoids thousands of fragile HTTP requests during streaming.
    # The datasets library handles retries and caching internally.
    print("PrE-Text C4 cache | downloading full C4-en dataset (this takes a while)...")
    try:
        full_dataset = load_dataset(
            PRETEXT_C4_SOURCE_DATASET,
            PRETEXT_C4_SOURCE_CONFIG,
            split=PRETEXT_C4_SOURCE_SPLIT,
            streaming=False,
            download_mode="reuse_cache_if_exists",
        )
        row_count = len(full_dataset) if hasattr(full_dataset, "__len__") else None
        if row_count is not None:
            print(f"PrE-Text C4 cache | loaded {row_count} rows from cache, now filtering...")
        else:
            print("PrE-Text C4 cache | loaded cached dataset, now filtering...")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download/load C4 dataset. Error: {exc}. "
            "Note: HF_TOKEN is set but the dataset library may need additional config. "
            "Try setting HF_TOKEN in environment before running."
        ) from exc

    temp_root = datasets_root() / "_pretext_c4_cache_build"
    remove_path(temp_root)
    ensure_dir(temp_root)
    file_handles = {
        category: (temp_root / f"{category}.jsonl").open("w", encoding="utf-8")
        for category in missing_categories
    }
    targets = {category: _PRETEXT_C4_TARGETS[category] for category in missing_categories}
    counts = {category: 0 for category in missing_categories}
    seen_hashes: set[bytes] = set()
    total_samples = sum(targets.values())
    completed_categories: list[str] = []

    def persist_completed_category(category: str) -> None:
        """Flush one completed category to the shared cache immediately."""

        if category in completed_categories:
            return
        handle = file_handles.pop(category, None)
        if handle is not None:
            handle.flush()
            handle.close()
        source = temp_root / f"{category}.jsonl"
        if not source.exists():
            return
        destination = _cache_file(category)
        if destination.exists():
            remove_path(destination)
        move_path(source, destination)
        completed_categories.append(category)
        _write_pretext_cache_metadata(
            cache_root,
            required_categories=required,
            built_categories=completed_categories,
        )

    if cached_categories:
        print(
            "PrE-Text C4 cache | reusing cached categories: "
            + ", ".join(cached_categories)
        )
    print(
        "PrE-Text C4 cache | building categories: "
        + ", ".join(missing_categories)
    )
    print(
        "PrE-Text C4 cache | targets: "
        + ", ".join(f"{category}={targets[category]}" for category in missing_categories)
    )
    progress = tqdm(total=total_samples, desc="PrE-Text C4 cache", unit="sample") if tqdm else None
    iteration_error: Exception | None = None

    # Iterate from local data (no network needed) - just filter and classify
    try:
        for row in full_dataset:
            text = _normalize_text(str(row.get("text") or ""))
            if not text or not _valid_text(text):
                continue

            text_hash = hashlib.sha1(text.encode("utf-8")).digest()
            if text_hash in seen_hashes:
                continue

            url = str(row.get("url") or "")
            category = classify_c4_url(url) or "initialization"
            if category not in targets:
                continue
            if counts[category] >= targets[category]:
                continue

            file_handles[category].write(
                json.dumps({"text": text, "url": url}, ensure_ascii=False) + "\n"
            )
            counts[category] += 1
            seen_hashes.add(text_hash)
            if progress:
                progress.update(1)
                if progress.n == 1 or progress.n % 1000 == 0 or counts[category] == targets[category]:
                    progress.set_postfix_str(_format_target_summary(missing_categories, counts, targets))
            if counts[category] == targets[category]:
                print(f"PrE-Text C4 cache | completed {category}: {targets[category]} samples")
                persist_completed_category(category)
            if all(counts[name] >= targets[name] for name in missing_categories):
                break
    except Exception as exc:
        iteration_error = exc

    # Always close file handles when done (success or failure)
    if progress:
        progress.close()
    for handle in list(file_handles.values()):
        handle.close()

    missing = {
        category: target - counts[category]
        for category, target in targets.items()
        if counts[category] < target
    }
    if missing or iteration_error is not None:
        remove_path(temp_root)
        missing_text = ", ".join(f"{category}={count}" for category, count in sorted(missing.items()))
        completed_text = ", ".join(completed_categories)
        completed_suffix = (
            f" Completed categories were preserved in the shared cache: {completed_text}."
            if completed_categories
            else ""
        )
        if iteration_error is not None:
            if missing_text:
                missing_text = f" Missing counts: {missing_text}."
            raise RuntimeError(
                "PrE-Text C4 cache build was interrupted before all requested categories completed."
                f"{completed_suffix}{missing_text} Original error: {iteration_error}"
            ) from iteration_error
        raise RuntimeError(
            "Unable to collect enough C4 rows for the requested PrE-Text cache. "
            f"Missing counts: {missing_text}."
        )

    for category in missing_categories:
        if category in completed_categories:
            continue
        destination = _cache_file(category)
        if destination.exists():
            remove_path(destination)
        move_path(temp_root / f"{category}.jsonl", destination)
    remove_path(temp_root)
    _write_pretext_cache_metadata(
        cache_root,
        required_categories=required,
        built_categories=missing_categories,
    )
    return cache_root


def _read_jsonl_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_pretext_c4_rows(category: str, force: bool = False) -> list[dict[str, str]]:
    """Return cached C4 rows for one paper-aligned PrE-Text domain bucket."""

    cache_root = ensure_pretext_c4_cache(required_categories=[category], force=force)
    return _read_jsonl_rows(cache_root / f"{category}.jsonl")


def split_pretext_private_rows(
    rows: list[dict[str, str]],
    train_size: int,
    eval_size: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split one cached row list into deterministic train/eval subsets."""

    if len(rows) < train_size + eval_size:
        raise ValueError(
            f"Need at least {train_size + eval_size} rows, but only collected {len(rows)}."
        )
    shuffled = list(rows)
    random.Random(PRETEXT_C4_RANDOM_SEED).shuffle(shuffled)
    train_rows = shuffled[:train_size]
    eval_rows = shuffled[train_size : train_size + eval_size]
    return train_rows, eval_rows


def stage_pretext_initialization_raw(downloader, force: bool = False) -> dict[str, object]:
    """Materialize the public PrE-Text initialization pool into one dataset raw directory."""

    rows = load_pretext_c4_rows("initialization", force=force)
    target = downloader.raw_path()
    if target is None:
        raise ValueError(f"{downloader.name} must define a raw path.")
    remove_path(target)
    ensure_dir(target)
    write_jsonl(target / "initialization.jsonl", rows)
    return {
        "message": "Collected the public C4-derived initialization pool used by PrE-Text.",
        "metadata": {
            "raw_format": "jsonl",
            "source_type": "huggingface_dataset_streaming",
            "source_dataset": PRETEXT_C4_SOURCE_DATASET,
            "source_config": PRETEXT_C4_SOURCE_CONFIG,
            "source_split": PRETEXT_C4_SOURCE_SPLIT,
            "paper_alignment": {
                "paper": "PrE-Text",
                "dataset": "Initialization",
                "intended_use": "public_seed_pool",
            },
            "source_row_count": len(rows),
            "provenance_note": (
                "PrE-Text's original initialization.json is not distributed with the paper repository, so this "
                "downloader stages a C4-derived approximation that excludes the paper's private-domain URL buckets."
            ),
        },
    }


def stage_pretext_private_raw(
    downloader,
    *,
    category: str,
    paper_dataset_name: str,
    train_size: int,
    eval_size: int,
    force: bool = False,
    provenance_note: str,
) -> dict[str, object]:
    """Materialize one private PrE-Text train/eval dataset pair into raw JSONL files."""

    rows = load_pretext_c4_rows(category, force=force)
    train_rows, eval_rows = split_pretext_private_rows(rows, train_size=train_size, eval_size=eval_size)
    target = downloader.raw_path()
    if target is None:
        raise ValueError(f"{downloader.name} must define a raw path.")
    remove_path(target)
    ensure_dir(target)
    write_jsonl(target / "train.jsonl", train_rows)
    write_jsonl(target / "eval.jsonl", eval_rows)
    return {
        "message": "Collected the paper-aligned PrE-Text raw train/eval corpora.",
        "metadata": {
            "raw_format": "jsonl",
            "source_type": "huggingface_dataset_streaming",
            "source_dataset": PRETEXT_C4_SOURCE_DATASET,
            "source_config": PRETEXT_C4_SOURCE_CONFIG,
            "source_split": PRETEXT_C4_SOURCE_SPLIT,
            "paper_alignment": {
                "paper": "PrE-Text",
                "dataset": paper_dataset_name,
            },
            "source_row_count": len(rows),
            "raw_split_sizes": {"train": len(train_rows), "eval": len(eval_rows)},
            "source_category": category,
            "sampling_seed": PRETEXT_C4_RANDOM_SEED,
            "provenance_note": provenance_note,
        },
    }
