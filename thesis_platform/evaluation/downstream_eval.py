from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from thesis_platform.core.io_utils import ensure_dir, to_jsonable, write_json


def export_synthetic_corpus(
    synthetic_texts: list[str],
    *,
    output_dir: Path,
    filename: str = "llama7b_text_syn.json",
) -> Path:
    """Write the final synthetic corpus in the format expected by pretext large-eval."""

    output_dir = ensure_dir(output_dir)
    corpus_path = output_dir / filename
    deduped: list[str] = []
    seen: set[str] = set()
    for text in synthetic_texts:
        cleaned = str(text).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    corpus_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
    return corpus_path


def _ensure_pretext_import(repo_root: Path) -> None:
    pretext_root = (repo_root / "PrE-Text").resolve()
    if str(pretext_root) not in sys.path:
        sys.path.insert(0, str(pretext_root))


def run_pretext_large_eval(thesis_config, *, stage2_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the PrE-Text large-model downstream evaluation in-process."""

    repo_root = thesis_config.repo_root()
    _ensure_pretext_import(repo_root)

    from pretext_platform.core.config import ExperimentConfig as PretextExperimentConfig
    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle
    from pretext_platform.evaluation.llama2_eval import run_llama2_eval

    downstream_cfg = thesis_config.downstream_eval
    pretext_raw = {
        "meta": {
            "experiment_id": f"{thesis_config.meta.get('experiment_id', 'experiment')}_pretext_large_eval",
            "seed": int(thesis_config.meta.get("seed", 42)),
        },
        "paths": {
            "repo_root": str(repo_root),
            "output_root": str(output_dir),
            "dataset_root": str(
                thesis_config.resolve_path(downstream_cfg.get("dataset_root", "thesis_platform/datasets"))
            ),
            "model_root": str(
                thesis_config.resolve_path(downstream_cfg.get("model_root", "thesis_platform/open_model"))
            ),
        },
        "data": {
            "dataset_name": str(thesis_config.data.get("dataset_name", "jobs")),
            "train_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get("train_path", thesis_config.data.get("train_path", ""))
                )
            ),
            "eval_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get("eval_path", thesis_config.data.get("eval_path", ""))
                )
            ),
            "initialization_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get(
                        "initialization_path",
                        thesis_config.data.get(
                            "initialization_path",
                            "thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json",
                        ),
                    )
                )
            ),
            "max_samples_per_client": int(thesis_config.data.get("max_samples_per_client", 8)),
            "initialization_min_words": int(thesis_config.data.get("initialization_min_words", 20)),
        },
        "models": {
            "minilm_path": str(
                thesis_config.resolve_path(downstream_cfg.get("minilm_path", "thesis_platform/open_model/all_minilm_l6_v2"))
            ),
            "roberta_large_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get("roberta_large_path", "thesis_platform/open_model/roberta_large")
                )
            ),
            "llama2_7b_path": str(
                thesis_config.resolve_path(downstream_cfg.get("llama2_7b_path", "thesis_platform/open_model/llama_2_7b_hf"))
            ),
            "distilgpt2_path": str(
                thesis_config.resolve_path(downstream_cfg.get("distilgpt2_path", "thesis_platform/open_model/distilgpt2"))
            ),
            "c4_checkpoint_path": downstream_cfg.get("c4_checkpoint_path", ""),
        },
        "stage1": {"enabled": False, "rounds": 1},
        "bootstrap": {"enabled": False},
        "eval_small": {"enabled": False},
        "eval_large": {
            "enabled": True,
            "cutoff_len": int(downstream_cfg.get("cutoff_len", 64)),
            "grad_accum_steps": int(downstream_cfg.get("grad_accum_steps", 16)),
            "epochs": int(downstream_cfg.get("epochs", 1)),
            "batch_size": int(downstream_cfg.get("batch_size", 8)),
            "eval_batch_size": int(downstream_cfg.get("eval_batch_size", 2)),
            "learning_rate": float(downstream_cfg.get("learning_rate", 0.0002)),
            "num_proc": int(downstream_cfg.get("num_proc", 1)),
            "lora_rank": int(downstream_cfg.get("lora_rank", 4)),
            "lora_alpha": int(downstream_cfg.get("lora_alpha", 8)),
            "lora_dropout": float(downstream_cfg.get("lora_dropout", 0.0)),
        },
        "runtime": {
            "device": str(thesis_config.runtime.get("device", "cuda")),
        },
    }
    pretext_config = PretextExperimentConfig.from_mapping(pretext_raw, base_dir=repo_root, name="thesis_v3_pretext_eval.yaml")
    dataset_bundle = load_dataset_bundle(pretext_config)
    model_paths = resolve_model_paths(pretext_config)
    summary = run_llama2_eval(pretext_config, dataset_bundle, model_paths, stage2_dir, output_dir)
    return to_jsonable(summary)


def collect_baseline_summaries(repo_root: Path, summary_paths: list[str], *, output_dir: Path) -> dict[str, Any]:
    """Collect existing baseline summary files into one normalized payload."""

    resolved: dict[str, Any] = {}
    for raw_path in summary_paths:
        path = (repo_root / raw_path).resolve()
        if not path.exists():
            resolved[raw_path] = {"missing": True}
            continue
        with path.open("r", encoding="utf-8") as handle:
            resolved[raw_path] = json.load(handle)
    write_json(ensure_dir(output_dir) / "baseline_summaries.json", resolved)
    return resolved
