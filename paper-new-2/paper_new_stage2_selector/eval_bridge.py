from __future__ import annotations

import json
from pathlib import Path

from .thesis_bridge import run_eval_selected_texts


def write_selected_stage2_dir(selected_texts: list[str], *, output_dir: Path) -> Path:
    stage2_dir = output_dir / "stage2_selected"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = stage2_dir / "llama7b_text_syn.json"
    corpus_path.write_text(json.dumps(list(selected_texts), ensure_ascii=False, indent=2), encoding="utf-8")
    return stage2_dir


def run_eval_from_stage2_dir(config_path: str | Path, *, stage2_dir: Path, output_dir: Path) -> dict[str, Any]:
    corpus_path = stage2_dir / "llama7b_text_syn.json"
    selected_texts = json.loads(corpus_path.read_text(encoding="utf-8"))
    return run_eval_selected_texts(config_path, selected_texts=list(selected_texts), output_dir=output_dir)
