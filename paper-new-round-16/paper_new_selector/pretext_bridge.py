from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

from .thesis_bridge import load_yaml_config, resolve_repo_root


def _ensure_pretext_importable(repo_root: Path) -> None:
    pretext_root = (repo_root / "PrE-Text").resolve()
    if str(pretext_root) not in sys.path:
        sys.path.insert(0, str(pretext_root))


def _resolve_vllm_host_ip() -> str:
    for env_key in ("VLLM_HOST_IP", "HOST_IP", "MASTER_ADDR"):
        configured = str(os.environ.get(env_key, "")).strip()
        if configured:
            return configured
    return "127.0.0.1"


def _patch_vllm_network_host_ip() -> str:
    host_ip = _resolve_vllm_host_ip()
    os.environ["VLLM_HOST_IP"] = host_ip
    os.environ["HOST_IP"] = host_ip
    os.environ["MASTER_ADDR"] = host_ip
    try:
        import vllm.utils as vllm_utils

        vllm_utils.get_ip = lambda: host_ip
    except Exception:
        pass
    try:
        import vllm.engine.llm_engine as llm_engine

        llm_engine.get_ip = lambda: host_ip
    except Exception:
        pass
    return host_ip


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


def generate_with_shared_vllm_session(
    prompt_list: list[str],
    *,
    shared_session: dict[str, Any] | Any,
    bootstrap_cfg: dict[str, Any],
) -> list[str]:
    """Generate Stage 2 outputs with the already-loaded shared Stage 1 vLLM backend."""

    if isinstance(shared_session, dict):
        backend = shared_session.get("backend")
    else:
        backend = getattr(shared_session, "backend", None)
    if backend is None:
        raise ValueError(
            "shared_session must expose a backend for Stage 2 shared generation."
        )
    return backend.generate_batch(
        prompt_list,
        max_new_tokens=int(bootstrap_cfg.get("max_tokens", 85)),
        temperature=float(bootstrap_cfg.get("temperature", 1.0)),
    )


def prepare_bootstrap_runtime(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)

    _dataset_name = str(config.get("data", {}).get("dataset_name", ""))
    if _dataset_name == "forums":
        _bootstrap_overrides = [
            ("_forums_max_tokens", "max_tokens"),
        ]
        for _src_key, _tgt_key in _bootstrap_overrides:
            if _src_key in config.get("selector", {}):
                if "bootstrap" not in config:
                    config["bootstrap"] = {}
                config["bootstrap"][_tgt_key] = float(config["selector"][_src_key])

    repo_root = resolve_repo_root()
    _ensure_pretext_importable(repo_root)

    from pretext_platform.algorithms.bootstrap import (
        build_bootstrap_prompts,
        generate_bootstrapped_samples_vllm,
    )

    backend = str(config["bootstrap"]["generator_backend"]).strip().lower()
    if backend != "vllm":
        raise ValueError(
            "paper-new Stage 2 bootstrap requires bootstrap.generator_backend='vllm'. "
            "Non-vLLM backends are rejected to keep formal/test runs aligned with PrE-Text."
        )

    def generator_fn(prompt_list, model_path, bootstrap_cfg):
        _patch_vllm_network_host_ip()
        return generate_bootstrapped_samples_vllm(
            prompt_list, model_path, bootstrap_cfg
        )

    bootstrap_cfg = {
        "num_prompts": int(config["bootstrap"]["num_prompts"]),
        "generator_backend": "vllm",
        "generator_model": str(config["bootstrap"]["generator_model"]),
        "max_tokens": int(config["bootstrap"].get("max_tokens", 85)),
        "temperature": float(config["bootstrap"].get("temperature", 1.0)),
        "top_p": float(config["bootstrap"].get("top_p", 1.0)),
        "max_model_len": int(config["bootstrap"].get("max_model_len", 512)),
        "tensor_parallel_size": int(config["bootstrap"].get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(
            config["bootstrap"].get("gpu_memory_utilization", 0.35)
        ),
        "startup_required_free_gb": float(
            config["bootstrap"].get("startup_required_free_gb", 2)
        ),
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
        "generate_with_shared_session": generate_with_shared_vllm_session,
        "bootstrap_cfg": bootstrap_cfg,
        "model_path": model_path,
    }
