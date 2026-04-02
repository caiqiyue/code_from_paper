from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from thesis_platform.core.privacy import PrivacyPolicy
from thesis_platform.data.loaders import load_samples
from thesis_platform.evaluation.downstream_eval import (
    resolve_large_eval_mode,
    resolve_small_eval_mode,
)

try:
    from thesis_platform.core.lora_gradients import resolve_model_name_or_path
except Exception:  # pragma: no cover - fallback for dependency-light environments
    def resolve_model_name_or_path(model_name_or_path, repo_root=None):
        raw = str(model_name_or_path or "").strip()
        if not raw:
            return raw
        candidate = Path(raw)
        if candidate.is_absolute():
            return str(candidate.resolve()) if candidate.exists() else raw
        if repo_root is not None:
            repo_candidate = Path(repo_root) / raw.replace("\\", "/")
            if repo_candidate.exists():
                return str(repo_candidate.resolve())
        return raw


def _module_available(module_name: str) -> bool:
    """Return true when one Python module can be imported in the active environment."""

    return importlib.util.find_spec(module_name) is not None


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _looks_like_local_asset(value: str | Path | None) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    normalized = raw.replace("\\", "/")
    path = Path(raw)
    return (
        path.is_absolute()
        or normalized.startswith("./")
        or normalized.startswith("../")
        or normalized.startswith("thesis_platform/")
        or normalized.startswith("open_model/")
        or normalized.startswith("datasets/")
    )


def _validate_configured_dataset_paths(config, asset_errors: list[str]) -> None:
    """Validate all explicitly configured dataset paths, regardless of dataset name."""

    dataset_name = str(config.data.get("dataset_name", "dataset"))
    configured_paths = {
        f"{dataset_name} train dataset": config.data.get("train_path"),
        f"{dataset_name} eval dataset": config.data.get("eval_path"),
        f"{dataset_name} initialization dataset": config.data.get("initialization_path"),
        f"{dataset_name} public seed dataset": config.data.get("public_seed_path"),
    }
    for label, raw_path in configured_paths.items():
        if raw_path in (None, ""):
            continue
        path = config.resolve_path(raw_path)
        if path is None or not path.exists():
            asset_errors.append(f"missing {label}: {_display_path(path or Path('<unset>'))}")


def _validate_partition_inputs(config, asset_errors: list[str]) -> None:
    """Validate dataset partition assumptions before large models are loaded."""

    strategy = str(config.data.get("partition_strategy", "shuffle_round_robin")).strip().lower()
    if strategy != "preserve_buckets":
        return

    train_path = config.resolve_path(config.data.get("train_path"))
    if train_path is None or not train_path.exists():
        return

    try:
        samples = load_samples(
            train_path,
            dataset_name=str(config.data.get("dataset_name", "dataset")),
            source="real",
            task_type=str(config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="raw",
            prefix="preflight",
            sample_format=str(config.data.get("sample_format", "raw_text")),
            limit=(
                int(config.data.get("train_limit"))
                if config.data.get("train_limit") not in (None, "")
                else None
            ),
        )
    except Exception as exc:
        asset_errors.append(f"failed to inspect training data for preserve_buckets: {exc}")
        return

    bucket_ids = {
        str(sample.meta.get("bucket_id")).strip()
        for sample in samples
        if str(sample.meta.get("bucket_id", "")).strip()
    }
    required_clients = int(config.data.get("num_clients", 3))
    if len(bucket_ids) < required_clients:
        asset_errors.append(
            "partition_strategy='preserve_buckets' requires at least "
            f"{required_clients} distinct bucket_id/source_domain groups after dataset loading, "
            f"but only found {len(bucket_ids)} in {_display_path(train_path)}. "
            "Provide bucket metadata or keep the PrE-Text raw URL sidecar so source domains can be recovered."
        )


def _cross_domain_eval_runtime_cfg(config) -> dict[str, object]:
    """Resolve the effective downstream-eval settings used by cross-domain transfer runs."""

    cross_domain_eval = config.cross_domain_eval
    if not bool(cross_domain_eval.get("enabled", False)):
        return {}

    derived = dict(config.downstream_eval)
    derived["enabled"] = True
    derived["kind"] = str(cross_domain_eval.get("kind", "pretext_large_eval"))
    derived["run_large_eval"] = bool(cross_domain_eval.get("run_large_eval", True))
    derived["run_small_eval"] = bool(cross_domain_eval.get("run_small_eval", False))
    for key in (
        "large_eval_mode",
        "windows_large_eval_mode",
        "linux_large_eval_mode",
        "small_eval_mode",
        "windows_small_eval_mode",
        "linux_small_eval_mode",
        "model_root",
        "distilgpt2_path",
        "gpt2_xl_path",
        "llama2_7b_path",
        "llama_3_2_3b_instruct_path",
        "c4_checkpoint_path",
        "batch_size",
        "eval_batch_size",
        "grad_accum_steps",
        "epochs",
        "learning_rate",
        "num_proc",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "small_batch_size",
        "small_eval_batch_size",
        "small_grad_accum_steps",
        "small_epochs",
        "small_learning_rate",
        "small_num_proc",
    ):
        if key in cross_domain_eval:
            derived[key] = cross_domain_eval[key]
    return derived


def _validate_pretext_eval_requirements(
    config,
    eval_cfg: dict[str, object],
    dependency_errors: list[str],
    asset_errors: list[str],
    *,
    label_prefix: str,
) -> None:
    """Validate shared pretext-eval dependencies/assets for downstream and transfer runs."""

    if not bool(eval_cfg.get("enabled")):
        return

    for module_name, package_name in {
        "accelerate": "accelerate",
        "datasets": "datasets",
        "peft": "peft",
        "transformers": "transformers",
    }.items():
        if not _module_available(module_name):
            dependency_errors.append(f"missing Python package: {package_name}")

    run_large_eval = bool(eval_cfg.get("run_large_eval"))
    if run_large_eval:
        large_eval_mode = resolve_large_eval_mode(eval_cfg, platform_name=sys.platform)
        model_root = config.resolve_path(eval_cfg.get("model_root", "thesis_platform/open_model"))
        if model_root is None or not model_root.exists():
            asset_errors.append(f"missing {label_prefix} model root: {_display_path(model_root or Path('<unset>'))}")
        if large_eval_mode == "full_finetune":
            llama32_path = config.resolve_path(
                eval_cfg.get("llama_3_2_3b_instruct_path", "thesis_platform/open_model/llama_3_2_3b_instruct")
            )
            if llama32_path is None or not llama32_path.exists():
                asset_errors.append(
                    f"missing {label_prefix} llama_3_2_3b_instruct model: {_display_path(llama32_path or Path('<unset>'))}"
                )
        elif large_eval_mode == "gpt2_xl":
            gpt2_xl_path = config.resolve_path(eval_cfg.get("gpt2_xl_path", "thesis_platform/open_model/gpt2_xl"))
            distilgpt2_path = config.resolve_path(eval_cfg.get("distilgpt2_path", "thesis_platform/open_model/distilgpt2"))
            if (gpt2_xl_path is None or not gpt2_xl_path.exists()) and (
                distilgpt2_path is None or not distilgpt2_path.exists()
            ):
                asset_errors.append(
                    f"missing {label_prefix} gpt2_xl model and distilgpt2 fallback: "
                    f"{_display_path(gpt2_xl_path or Path('<unset>'))}, {_display_path(distilgpt2_path or Path('<unset>'))}"
                )
        else:
            llama_path = config.resolve_path(eval_cfg.get("llama2_7b_path", "thesis_platform/open_model/llama_2_7b_hf"))
            if llama_path is None or not llama_path.exists():
                asset_errors.append(f"missing {label_prefix} llama2_7b model: {_display_path(llama_path or Path('<unset>'))}")

    run_small_eval = bool(eval_cfg.get("run_small_eval"))
    if run_small_eval:
        small_eval_mode = resolve_small_eval_mode(eval_cfg, platform_name=sys.platform)
        distilgpt2_path = config.resolve_path(eval_cfg.get("distilgpt2_path", "thesis_platform/open_model/distilgpt2"))
        if distilgpt2_path is None or not distilgpt2_path.exists():
            asset_errors.append(f"missing {label_prefix} distilgpt2 model: {_display_path(distilgpt2_path or Path('<unset>'))}")
        if small_eval_mode != "gpt2":
            checkpoint_path = config.resolve_path(eval_cfg.get("c4_checkpoint_path"))
            if checkpoint_path is None or not checkpoint_path.exists():
                asset_errors.append(f"missing {label_prefix} c4 checkpoint: {_display_path(checkpoint_path or Path('<unset>'))}")


def validate_preflight(config) -> None:
    """Validate dependencies and local assets before a v3 experiment starts."""

    dependency_errors: list[str] = []
    asset_errors: list[str] = []

    scorer_name = str(config.scorer.get("name", "")).lower()
    aggregator_name = str(config.aggregator.get("name", "")).lower()
    prototype_name = str(config.prototype.get("name", "")).lower()
    routing_enabled = bool(config.routing.get("enabled", False))
    downstream_eval = config.downstream_eval
    cross_domain_eval = config.cross_domain_eval
    privacy_policy = PrivacyPolicy.from_config(config.privacy)

    try:
        privacy_policy.validate()
    except ValueError as exc:
        asset_errors.append(str(exc))

    _validate_partition_inputs(config, asset_errors)

    if scorer_name in {"datainf", "gradmm"}:
        for module_name, package_name in {
            "numpy": "numpy",
        }.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    if scorer_name in {"datainf_real", "gradmm_real"}:
        for module_name, package_name in {
            "numpy": "numpy",
            "sklearn": "scikit-learn",
            "torch": "torch",
            "transformers": "transformers",
        }.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    if scorer_name in {"datainf_paper", "gradmm_paper"}:
        required_packages = {
            "torch": "torch",
        }
        if bool(config.scorer.get("use_real_gradients", True)):
            required_packages["transformers"] = "transformers"
            required_packages["peft"] = "peft"
        for module_name, package_name in required_packages.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    if scorer_name in {"datainf_lora", "gradmm_lora"}:
        required_packages = {
            "torch": "torch",
            "transformers": "transformers",
        }
        if bool(config.scorer.get("use_real_gradients", True)):
            required_packages["peft"] = "peft"
        for module_name, package_name in required_packages.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    if aggregator_name in {"dbscan_attn", "dbscan_attn_tsgdm", "uid_llm"}:
        for module_name, package_name in {
            "numpy": "numpy",
            "sklearn": "scikit-learn",
        }.items():
            if not _module_available(module_name):
                dependency_errors.append(f"missing Python package: {package_name}")

    _validate_pretext_eval_requirements(
        config,
        downstream_eval,
        dependency_errors,
        asset_errors,
        label_prefix="downstream eval",
    )
    _validate_pretext_eval_requirements(
        config,
        _cross_domain_eval_runtime_cfg(config),
        dependency_errors,
        asset_errors,
        label_prefix="cross-domain eval",
    )

    _validate_configured_dataset_paths(config, asset_errors)
    if bool(cross_domain_eval.get("enabled", False)):
        target_dataset = str(cross_domain_eval.get("target_dataset", "target"))
        target_train_path = config.resolve_path(cross_domain_eval.get("target_train_path"))
        if target_train_path is None or not target_train_path.exists():
            asset_errors.append(
                f"missing cross-domain {target_dataset} train dataset: {_display_path(target_train_path or Path('<unset>'))}"
            )
        raw_target_eval_path = cross_domain_eval.get("target_eval_path")
        if raw_target_eval_path not in (None, ""):
            target_eval_path = config.resolve_path(raw_target_eval_path)
            if target_eval_path is None or not target_eval_path.exists():
                asset_errors.append(
                    f"missing cross-domain {target_dataset} eval dataset: {_display_path(target_eval_path or Path('<unset>'))}"
                )

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

    if scorer_name in {"datainf_lora", "gradmm_lora", "datainf_paper", "gradmm_paper"}:
        if bool(config.scorer.get("use_real_gradients", True)):
            raw_model_name = config.scorer.get("model_name")
            if _looks_like_local_asset(raw_model_name):
                raw_model_path = Path(str(raw_model_name).replace("\\", "/"))
                resolved_model_path = (
                    raw_model_path
                    if raw_model_path.is_absolute()
                    else (config.repo_root() / raw_model_path).resolve()
                )
                if not resolved_model_path.exists():
                    missing_label = (
                        "missing scorer LoRA base model"
                        if scorer_name in {"datainf_lora", "gradmm_lora"}
                        else "missing scorer base model"
                    )
                    asset_errors.append(
                        f"{missing_label}: {_display_path(resolved_model_path)}"
                    )
        elif scorer_model:
            scorer_model_path = config.resolve_path(scorer_model)
            if scorer_model_path is None or not scorer_model_path.exists():
                asset_errors.append(
                    f"missing scorer feature model: {_display_path(scorer_model_path or Path('<unset>'))}"
                )

    llm_cfg = config.llm
    for role in ("client", "server"):
        role_cfg = dict(llm_cfg.get(role, {}))
        if str(role_cfg.get("engine", "")).lower() != "transformers":
            continue
        model_path = config.resolve_path(role_cfg.get("model_name_or_path"))
        if model_path is None or not model_path.exists():
            asset_errors.append(f"missing {role} text backend model: {_display_path(model_path or Path('<unset>'))}")

    if dependency_errors or asset_errors:
        message = ["Preflight validation failed."]
        if dependency_errors:
            message.append("Dependencies:")
            message.extend(f"- {item}" for item in dependency_errors)
        if asset_errors:
            message.append("Assets:")
            message.extend(f"- {item}" for item in asset_errors)
        raise ValueError("\n".join(message))



