from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .thesis_bridge import load_yaml_config, resolve_repo_root


def _ensure_pretext_importable(repo_root: Path) -> None:
    pretext_root = (repo_root / "PrE-Text").resolve()
    if str(pretext_root) not in sys.path:
        sys.path.insert(0, str(pretext_root))


def resolve_bootstrap_model_path(models_root: str | Path, model_name: str) -> Path:
    repo_root = resolve_repo_root()
    root = Path(models_root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    mapping = {
        "llama2_7b": "llama_2_7b_hf",
        "distilgpt2": "distilgpt2",
    }
    resolved = root / mapping.get(model_name, model_name)
    return resolved.resolve()


def prepare_bootstrap_runtime(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    repo_root = resolve_repo_root()
    _ensure_pretext_importable(repo_root)

    from pretext_platform.algorithms.bootstrap import (
        build_bootstrap_prompts,
        generate_bootstrapped_samples,
        generate_bootstrapped_samples_hf,
        generate_bootstrapped_samples_vllm,
    )

    backend = str(config["bootstrap"]["generator_backend"]).strip().lower()
    if backend not in {"vllm", "huggingface"}:
        raise ValueError(
            f"bootstrap.generator_backend must be 'vllm' or 'huggingface', got '{backend}'."
        )
    generator_fn = generate_bootstrapped_samples if backend == "vllm" else generate_bootstrapped_samples_hf
    if backend == "vllm":
        generator_fn = generate_bootstrapped_samples_vllm

    bootstrap_cfg = {
        "num_prompts": int(config["bootstrap"]["num_prompts"]),
        "generator_backend": backend,
        "generator_model": str(config["bootstrap"]["generator_model"]),
        "max_tokens": int(config["bootstrap"].get("max_tokens", 85)),
        "temperature": float(config["bootstrap"].get("temperature", 1.0)),
        "top_p": float(config["bootstrap"].get("top_p", 1.0)),
        "max_model_len": int(config["bootstrap"].get("max_model_len", 512)),
        "tensor_parallel_size": int(config["bootstrap"].get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(config["bootstrap"].get("gpu_memory_utilization", 0.55)),
        "enforce_eager": bool(config["bootstrap"].get("enforce_eager", True)),
        "device": str(config["bootstrap"].get("device", "cuda")),
        "batch_size": int(config["bootstrap"].get("batch_size", 1)),
    }
    model_path = resolve_bootstrap_model_path(
        str(config["paths"]["models_root"]),
        str(config["bootstrap"]["generator_model"]),
    )
    return {
        "build_bootstrap_prompts": build_bootstrap_prompts,
        "generate_bootstrapped_samples": generator_fn,
        "bootstrap_cfg": bootstrap_cfg,
        "model_path": model_path,
    }
