from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..eval_bridge import run_eval


def normalize_direct_synthetic_texts(texts: list[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_text in texts:
        text = str(raw_text).strip()
        if len(text.split()) < 2:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def load_direct_synthetic_summary(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("skip_bootstrap", False)):
        raise ValueError(
            f"External baseline summary must set skip_bootstrap=true: {path}"
        )
    texts = payload.get("direct_synthetic_texts")
    if not isinstance(texts, list):
        raise ValueError(
            f"External baseline summary must contain direct_synthetic_texts list: {path}"
        )
    payload["direct_synthetic_texts"] = normalize_direct_synthetic_texts(texts)
    return payload


def run_external_stage1_summary_eval(
    *,
    summary_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_direct_synthetic_summary(summary_path)
    eval_output_dir = Path(output_dir) if output_dir is not None else None
    return run_eval(
        synthetic_texts=list(payload["direct_synthetic_texts"]),
        config_path=config_path,
        output_dir=eval_output_dir,
    )
