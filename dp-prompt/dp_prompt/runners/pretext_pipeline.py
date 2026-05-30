from __future__ import annotations

from pathlib import Path
from typing import Any

from dp_prompt.decoding.privacy import build_privacy_controls_summary
from dp_prompt.pretext_bridge import (
    build_thesis_eval_config,
    ensure_thesis_platform_importable,
    export_and_run_small_eval,
    resolve_repo_root,
)
from dp_prompt.prompting.templates import render_document_prompt
from dp_prompt.utils.io import ensure_dir, write_json


def _load_pretext_texts(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    ensure_thesis_platform_importable(resolve_repo_root())
    from thesis_platform.data.loaders import load_texts

    train_path = Path(config["data"]["train_path"]).expanduser()
    eval_path = Path(config["data"]["eval_path"]).expanduser()
    return load_texts(train_path), load_texts(eval_path)


def _build_vllm_backend(config: dict[str, Any]) -> Any:
    ensure_thesis_platform_importable(resolve_repo_root())
    from thesis_platform.models.backends import build_text_backend

    llm_server_cfg = dict(config["llm"]["server"])
    return build_text_backend(llm_server_cfg, repo_root=resolve_repo_root())


def _generate_sanitized_texts(train_texts: list[str], config: dict[str, Any], backend: Any) -> tuple[list[dict[str, Any]], list[str]]:
    generation_cfg = config["generation"]
    privacy_cfg = config["privacy"]
    prompt_cfg = config.get("prompting", {})
    limit = int(config["data"].get("train_limit") or generation_cfg.get("num_documents") or len(train_texts))
    selected = train_texts[:limit]
    prompts = [
        render_document_prompt(text, template_name=prompt_cfg.get("template_name", "document_paraphrase"))
        for text in selected
    ]
    outputs = backend.generate_batch(
        prompts,
        max_new_tokens=int(privacy_cfg.get("max_generated_tokens", generation_cfg.get("max_new_tokens", 128))),
        temperature=float(privacy_cfg.get("temperature", generation_cfg.get("temperature", 0.2))),
    )
    records = []
    synthetic_texts: list[str] = []
    for idx, (source_text, prompt, generated_text) in enumerate(zip(selected, prompts, outputs, strict=True)):
        cleaned = str(generated_text).strip()
        synthetic_texts.append(cleaned)
        records.append(
            {
                "sample_id": f"pretext_style_{idx}",
                "original_text": source_text,
                "prompt": prompt,
                "sanitized_text": cleaned,
            }
        )
    return records, synthetic_texts


def run_pretext_style_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(config["experiment"]["id"])
    output_dir = ensure_dir(Path(config["runtime"]["output_root"]).expanduser() / experiment_id)
    stage2_dir = ensure_dir(output_dir / "stage2")
    eval_dir = ensure_dir(output_dir / "eval")

    train_texts, eval_texts = _load_pretext_texts(config)
    backend = _build_vllm_backend(config)
    sanitized_records, synthetic_texts = _generate_sanitized_texts(train_texts, config, backend)

    write_json(output_dir / "sanitized_corpus.json", sanitized_records)
    write_json(
        output_dir / "privacy_controls_summary.json",
        build_privacy_controls_summary(config.get("privacy", {})),
    )

    # Release vLLM GPU memory eagerly; torch.cuda.empty_cache() frees PyTorch-managed
    # tensors but NOT vLLM's raw cuMalloc allocations. Process isolation (the caller
    # runs eval in a separate subprocess after this process exits) is the only
    # guarantee. We still call release() as a best-effort.
    _backend_release = getattr(backend, "release", None)
    if callable(_backend_release):
        _backend_release()
    del backend, _backend_release

    # skip_eval=True means the caller (run_e1_dpprompt_baseline.py) will run eval
    # in a fresh subprocess after this process exits and fully frees GPU memory.
    skip_eval = bool(config.get("runtime", {}).get("skip_eval", False))

    eval_summary: dict[str, Any] = {}
    if not skip_eval:
        thesis_eval_config = build_thesis_eval_config(
            experiment_id=experiment_id,
            dataset_name=str(config["data"]["dataset_name"]),
            train_path=str(config["data"]["train_path"]),
            eval_path=str(config["data"]["eval_path"]),
            initialization_path=str(config["data"]["initialization_path"]),
            output_root=str(output_dir),
            train_limit=config["data"].get("train_limit"),
            eval_limit=config["data"].get("eval_limit"),
            initialization_limit=config["data"].get("initialization_limit"),
            max_samples_per_client=int(config["data"].get("max_samples_per_client", 8)),
            linux_small_eval_mode=str(config.get("evaluation", {}).get("linux_small_eval_mode", "distilgpt2")),
        )
        eval_summary = export_and_run_small_eval(
            thesis_eval_config,
            synthetic_texts=synthetic_texts,
            stage2_dir=stage2_dir,
            eval_dir=eval_dir,
        )
        write_json(eval_dir / "eval_small_summary.json", eval_summary)

    summary = {
        "experiment_id": experiment_id,
        "pipeline_mode": "pretext_style",
        "dataset_name": config["data"]["dataset_name"],
        "train_source_count": len(train_texts),
        "eval_source_count": len(eval_texts),
        "synthetic_count": len(synthetic_texts),
        "llm_backend": config["llm"]["server"],
        "generation": config["generation"],
        "privacy_controls": build_privacy_controls_summary(config.get("privacy", {})),
        "artifacts": {
            "sanitized_corpus": str(output_dir / "sanitized_corpus.json"),
            "stage2_dir": str(stage2_dir),
            "eval_dir": str(eval_dir),
        },
        "evaluation": eval_summary,
        "skip_eval": skip_eval,
    }
    write_json(output_dir / "experiment_summary.json", summary)
    return summary
