from __future__ import annotations

import importlib.util
from pathlib import Path


def _module_available(module_name: str) -> bool:
    """Return true when one Python module can be imported in the active environment."""

    return importlib.util.find_spec(module_name) is not None


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def validate_preflight(config) -> None:
    """Validate dependencies and local assets before a v3 experiment starts."""

    dependency_errors: list[str] = []
    asset_errors: list[str] = []

    scorer_name = str(config.scorer.get("name", "")).lower()
    prototype_name = str(config.prototype.get("name", "")).lower()
    routing_enabled = bool(config.routing.get("enabled", False))
    downstream_eval = config.downstream_eval

    if scorer_name in {"datainf_real", "gradmm_real"}:
        for module_name, package_name in {
            "numpy": "numpy",
            "sklearn": "scikit-learn",
            "torch": "torch",
            "transformers": "transformers",
        }.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    if downstream_eval.get("enabled"):
        for module_name, package_name in {
            "accelerate": "accelerate",
            "datasets": "datasets",
            "peft": "peft",
            "transformers": "transformers",
        }.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    if config.data.get("dataset_name") == "jobs":
        required_paths = {
            "jobs train dataset": config.resolve_path(
                config.data.get("train_path", "thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json")
            ),
            "jobs eval dataset": config.resolve_path(
                config.data.get("eval_path", "thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json")
            ),
            "jobs initialization dataset": config.resolve_path(
                config.data.get(
                    "initialization_path",
                    "thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json",
                )
            ),
        }
        for label, path in required_paths.items():
            if path is None or not path.exists():
                asset_errors.append(f"missing {label}: {_display_path(path or Path('<unset>'))}")

    prototype_model = config.prototype.get("embedding_model")
    if routing_enabled and prototype_name == "minilm_mean" and prototype_model:
        prototype_model_path = config.resolve_path(prototype_model)
        if prototype_model_path is None or not prototype_model_path.exists():
            asset_errors.append(f"missing prototype model: {_display_path(prototype_model_path or Path('<unset>'))}")

    scorer_model = config.scorer.get("feature_model")
    if scorer_name in {"datainf_real", "gradmm_real"} and scorer_model:
        scorer_model_path = config.resolve_path(scorer_model)
        if scorer_model_path is None or not scorer_model_path.exists():
            asset_errors.append(f"missing scorer feature model: {_display_path(scorer_model_path or Path('<unset>'))}")

    llm_cfg = config.llm
    for role in ("client", "server"):
        role_cfg = dict(llm_cfg.get(role, {}))
        if str(role_cfg.get("engine", "")).lower() != "transformers":
            continue
        model_path = config.resolve_path(role_cfg.get("model_name_or_path"))
        if model_path is None or not model_path.exists():
            asset_errors.append(f"missing {role} text backend model: {_display_path(model_path or Path('<unset>'))}")

    if downstream_eval.get("enabled") and downstream_eval.get("kind") == "pretext_large_eval":
        model_root = config.resolve_path(downstream_eval.get("model_root", "thesis_platform/open_model"))
        llama_path = config.resolve_path(
            downstream_eval.get("llama2_7b_path", "thesis_platform/open_model/llama_2_7b_hf")
        )
        if model_root is None or not model_root.exists():
            asset_errors.append(f"missing downstream eval model root: {_display_path(model_root or Path('<unset>'))}")
        if llama_path is None or not llama_path.exists():
            asset_errors.append(f"missing downstream eval llama2_7b model: {_display_path(llama_path or Path('<unset>'))}")

    if dependency_errors or asset_errors:
        message = ["Preflight validation failed."]
        if dependency_errors:
            message.append("Dependencies:")
            message.extend(f"- {item}" for item in dependency_errors)
        if asset_errors:
            message.append("Assets:")
            message.extend(f"- {item}" for item in asset_errors)
        raise ValueError("\n".join(message))
