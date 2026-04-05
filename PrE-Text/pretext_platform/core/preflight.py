from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import platform
from pathlib import Path
from typing import Any

from pretext_platform.core.config import ExperimentConfig, load_experiment_config
from pretext_platform.core.models import resolve_model_paths


@dataclass(slots=True)
class PreflightIssue:
    severity: str
    category: str
    message: str


@dataclass(slots=True)
class PreflightReport:
    experiment_id: str
    config_path: str
    enabled_stages: list[str]
    ready: bool
    errors: list[PreflightIssue]
    warnings: list[PreflightIssue]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _add_issue(issues: list[PreflightIssue], *, severity: str, category: str, message: str) -> None:
    issues.append(PreflightIssue(severity=severity, category=category, message=message))


def enabled_stage_names(config: ExperimentConfig, *, with_glue: bool = False) -> list[str]:
    stages: list[str] = []
    if bool(config.stage1.get("enabled", True)):
        stages.append("stage1")
    if bool(config.bootstrap.get("enabled", True)):
        stages.append("stage2")
    if bool(config.eval_small.get("enabled", False)):
        stages.append("eval_small")
    if bool(config.eval_large.get("enabled", False)):
        stages.append("eval_large")
    if bool(config.eval_glue.get("enabled", False)) or with_glue:
        stages.append("eval_glue")
    return stages


def _required_modules(config: ExperimentConfig, *, with_glue: bool) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}

    if bool(config.stage1.get("enabled", True)):
        required["stage1"] = {
            "accelerate",
            "faiss",
            "numpy",
            "opacus",
            "sentence_transformers",
            "torch",
            "transformers",
        }

    if bool(config.bootstrap.get("enabled", True)):
        backend = str(config.bootstrap.get("generator_backend", "auto"))
        stage2_modules = {"torch", "transformers"}
        generator_model = str(config.bootstrap.get("generator_model", "llama2_7b"))
        # Support both llama2_7b and distilgpt2 for testing
        if generator_model == "llama2_7b":
            stage2_modules.add("sentencepiece")
        elif generator_model != "distilgpt2":
            raise ValueError(
                "bootstrap.generator_model must be 'llama2_7b' or 'distilgpt2'."
            )
        if backend == "vllm":
            stage2_modules.add("vllm")
        required["stage2"] = stage2_modules

    if bool(config.eval_small.get("enabled", False)):
        required["eval_small"] = {"accelerate", "datasets", "torch", "transformers"}

    if bool(config.eval_large.get("enabled", False)):
        eval_mode = str(config.eval_large.get("eval_mode", "peft_lora"))
        eval_large_modules = {"datasets", "torch", "transformers"}
        if eval_mode == "peft_lora":
            eval_large_modules.update({"accelerate", "peft", "sentencepiece"})
        else:
            raise ValueError(
                "eval_large.eval_mode must be 'peft_lora' on the fixed Linux server."
            )
        required["eval_large"] = eval_large_modules

    if bool(config.eval_glue.get("enabled", False)) or with_glue:
        required["eval_glue"] = {"datasets", "torch", "transformers"}

    return required


def _check_python_modules(
    config: ExperimentConfig,
    *,
    with_glue: bool,
    errors: list[PreflightIssue],
) -> None:
    for stage_name, modules in _required_modules(config, with_glue=with_glue).items():
        missing = sorted(module_name for module_name in modules if not _module_available(module_name))
        if missing:
            _add_issue(
                errors,
                severity="error",
                category="dependency",
                message=f"{stage_name} requires missing Python modules: {', '.join(missing)}",
            )


def _configured_dataset_paths(config: ExperimentConfig) -> dict[str, Path]:
    data_cfg = config.data
    dataset_name = str(data_cfg.get("dataset_name", "dataset"))
    dataset_root = config.dataset_root()
    return {
        "train": config.resolve_path(data_cfg.get("train_path")) or (dataset_root / f"{dataset_name}_train.json").resolve(),
        "eval": config.resolve_path(data_cfg.get("eval_path")) or (dataset_root / f"{dataset_name}_eval.json").resolve(),
        "initialization": config.resolve_path(data_cfg.get("initialization_path"))
        or (dataset_root / "initial_set.json").resolve(),
    }


def _check_dataset_files(config: ExperimentConfig, *, errors: list[PreflightIssue]) -> None:
    needs_bundle = any(
        (
            bool(config.stage1.get("enabled", True)),
            bool(config.eval_small.get("enabled", False)),
            bool(config.eval_large.get("enabled", False)),
        )
    )
    if not needs_bundle:
        return

    for logical_name, path in _configured_dataset_paths(config).items():
        if not path.exists():
            _add_issue(
                errors,
                severity="error",
                category="data",
                message=f"Configured {logical_name} dataset path does not exist: {path}",
            )


def _check_glue_datasets(
    config: ExperimentConfig,
    *,
    with_glue: bool,
    errors: list[PreflightIssue],
) -> None:
    if not (bool(config.eval_glue.get("enabled", False)) or with_glue):
        return

    from pretext_platform.evaluation.glue_classification_eval import SUPPORTED_TASKS, validate_local_glue_datasets

    validation = validate_local_glue_datasets(config.dataset_root())
    requested_tasks = config.eval_glue.get("tasks", list(SUPPORTED_TASKS))
    if with_glue and not config.eval_glue.get("tasks"):
        requested_tasks = list(SUPPORTED_TASKS)

    for task_name in requested_tasks:
        task_info = validation["tasks"].get(task_name)
        if task_info is None or not task_info.get("available", False):
            reason = task_info["reason"] if task_info is not None else "No local validation entry found."
            _add_issue(
                errors,
                severity="error",
                category="data",
                message=f"eval_glue task '{task_name}' is not available locally: {reason}",
            )


def _check_model_paths(
    config: ExperimentConfig,
    *,
    with_glue: bool,
    errors: list[PreflightIssue],
    warnings: list[PreflightIssue],
) -> None:
    model_paths = resolve_model_paths(config)

    if bool(config.stage1.get("enabled", True)):
        required_stage1_paths = {
            "minilm": model_paths.minilm,
            "roberta_large": model_paths.roberta_large,
        }
        for logical_name, path in required_stage1_paths.items():
            if not path.exists():
                _add_issue(
                    errors,
                    severity="error",
                    category="model",
                    message=f"Stage 1 requires local model path '{logical_name}' to exist: {path}",
                )

    if bool(config.bootstrap.get("enabled", True)):
        generator_model = str(config.bootstrap.get("generator_model", "llama2_7b"))
        # Support both llama2_7b and distilgpt2 for testing
        if generator_model == "llama2_7b":
            bootstrap_model_path = model_paths.llama2_7b
        elif generator_model == "distilgpt2":
            bootstrap_model_path = model_paths.distilgpt2
        else:
            _add_issue(
                errors,
                severity="error",
                category="model",
                message="Stage 2 bootstrap only supports generator_model='llama2_7b' or 'distilgpt2'.",
            )
            bootstrap_model_path = None
        if bootstrap_model_path is None or not bootstrap_model_path.exists():
            _add_issue(
                errors,
                severity="error",
                category="model",
                message=f"Stage 2 requires local bootstrap model '{generator_model}' at: {bootstrap_model_path}",
            )

    if bool(config.eval_small.get("enabled", False)):
        eval_mode = str(config.eval_small.get("eval_mode", "gpt2"))
        if not model_paths.distilgpt2.exists():
            _add_issue(
                errors,
                severity="error",
                category="model",
                message=f"eval_small requires a local DistilGPT2/GPT2 model directory at: {model_paths.distilgpt2}",
            )
        if eval_mode != "gpt2" and (model_paths.c4_checkpoint is None or not model_paths.c4_checkpoint.exists()):
            _add_issue(
                errors,
                severity="error",
                category="model",
                message="eval_small with distilgpt2 warm-start requires models.c4_checkpoint_path to exist.",
            )

    if bool(config.eval_large.get("enabled", False)):
        eval_mode = str(config.eval_large.get("eval_mode", "peft_lora"))
        if eval_mode == "peft_lora":
            if not model_paths.llama2_7b.exists():
                _add_issue(
                    errors,
                    severity="error",
                    category="model",
                    message=f"eval_large peft_lora requires local LLaMA-2-7B weights at: {model_paths.llama2_7b}",
                )
        else:
            _add_issue(
                errors,
                severity="error",
                category="model",
                message="eval_large only supports eval_mode='peft_lora'.",
            )

    if bool(config.eval_glue.get("enabled", False)) or with_glue:
        if not model_paths.distilgpt2.exists():
            _add_issue(
                errors,
                severity="error",
                category="model",
                message=f"eval_glue requires a local DistilGPT2 directory at: {model_paths.distilgpt2}",
            )


def _check_stage_dependencies(
    config: ExperimentConfig,
    *,
    with_glue: bool,
    errors: list[PreflightIssue],
) -> None:
    experiment_dir = config.output_root() / config.experiment_id()

    if not bool(config.stage1.get("enabled", True)) and bool(config.bootstrap.get("enabled", True)):
        rounds = int(config.stage1.get("rounds", 11))
        stage1_dir = experiment_dir / "stage1"
        missing = [stage1_dir / f"surviving_text_it{round_idx}.json" for round_idx in range(rounds) if not (stage1_dir / f"surviving_text_it{round_idx}.json").exists()]
        if missing:
            _add_issue(
                errors,
                severity="error",
                category="artifact",
                message=f"Bootstrap-only run requires existing Stage 1 survivors under {stage1_dir}. First missing file: {missing[0]}",
            )

    need_stage2_artifact = any(
        (
            bool(config.eval_small.get("enabled", False)),
            bool(config.eval_large.get("enabled", False)),
            bool(config.eval_glue.get("enabled", False)),
            with_glue,
        )
    )
    if need_stage2_artifact and not bool(config.bootstrap.get("enabled", True)):
        synthetic_path = experiment_dir / "stage2" / "llama7b_text_syn.json"
        if not synthetic_path.exists():
            _add_issue(
                errors,
                severity="error",
                category="artifact",
                message=f"Evaluation-only run requires existing Stage 2 synthetic corpus: {synthetic_path}",
            )


def _check_platform_risks(config: ExperimentConfig, *, warnings: list[PreflightIssue]) -> None:
    system_name = platform.system()
    if system_name == "Windows":
        if bool(config.bootstrap.get("enabled", True)):
            generator_backend = str(config.bootstrap.get("generator_backend", "auto"))
            if generator_backend == "auto":
                _add_issue(
                    warnings,
                    severity="warning",
                    category="platform",
                    message="Windows cannot use vLLM; Stage 2 will fall back to local Transformers generation and run much slower.",
                )
            elif generator_backend == "vllm":
                _add_issue(
                    warnings,
                    severity="warning",
                    category="platform",
                    message="bootstrap.generator_backend=vllm is not supported on Windows. Switch to huggingface or auto.",
                )

        eval_mode = str(config.eval_large.get("eval_mode", "peft_lora"))
        if bool(config.eval_large.get("enabled", False)) and eval_mode == "peft_lora":
            _add_issue(
                warnings,
                severity="warning",
                category="platform",
                message="eval_large peft_lora requires more careful GPU memory management on Windows than on Linux.",
            )


def run_preflight(
    config_or_path: ExperimentConfig | str | Path,
    *,
    with_glue: bool = False,
) -> PreflightReport:
    config = config_or_path if isinstance(config_or_path, ExperimentConfig) else load_experiment_config(config_or_path)
    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []

    _check_python_modules(config, with_glue=with_glue, errors=errors)
    _check_dataset_files(config, errors=errors)
    _check_glue_datasets(config, with_glue=with_glue, errors=errors)
    _check_model_paths(config, with_glue=with_glue, errors=errors, warnings=warnings)
    _check_stage_dependencies(config, with_glue=with_glue, errors=errors)
    _check_platform_risks(config, warnings=warnings)

    return PreflightReport(
        experiment_id=config.experiment_id(),
        config_path=str(config.path),
        enabled_stages=enabled_stage_names(config, with_glue=with_glue),
        ready=not errors,
        errors=errors,
        warnings=warnings,
    )


def format_preflight_report(report: PreflightReport) -> str:
    lines = [
        "=" * 70,
        f"PrE-Text Preflight: {'READY' if report.ready else 'BLOCKED'}",
        f"Experiment: {report.experiment_id}",
        f"Config: {report.config_path}",
        f"Enabled stages: {', '.join(report.enabled_stages) if report.enabled_stages else '(none)'}",
    ]
    if report.errors:
        lines.append("Errors:")
        for issue in report.errors:
            lines.append(f"  - [{issue.category}] {issue.message}")
    if report.warnings:
        lines.append("Warnings:")
        for issue in report.warnings:
            lines.append(f"  - [{issue.category}] {issue.message}")
    lines.append("=" * 70)
    return "\n".join(lines)
