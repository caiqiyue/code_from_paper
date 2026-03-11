from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

from .common import remove_path


def download_bbh_task_raw(task_name: str, target: Path) -> dict[str, object]:
    """Download one BBH task JSON file into the raw dataset directory."""

    if target.exists():
        remove_path(target)
    target.mkdir(parents=True, exist_ok=True)

    source_url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{task_name}.json"
    raw_json_path = target / f"{task_name}.json"
    urlretrieve(source_url, raw_json_path)

    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))

    return {
        "source_url": source_url,
        "task_name": task_name,
        "num_examples": len(payload["examples"]),
        "raw_format": "json",
    }
