from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def extract_baseline_training_text(raw_text: str) -> str:
    """Mirror the minimal cleaning that PrE-Text small eval applies before training."""

    split_samples = re.split("Orig", str(raw_text))
    candidate = split_samples[0].strip().strip("\n")
    if len(candidate.split(" ")) <= 3:
        return ""
    return candidate.replace("\n\n", " ").replace("\n", " ")


def load_stage2_json(path: Path) -> list[Any]:
    import json

    return list(json.loads(path.read_text(encoding="utf-8")))
