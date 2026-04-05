from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from pretext_platform.core.config import ExperimentConfig, load_experiment_config
from pretext_platform.core.io_utils import ensure_dir, write_json
from pretext_platform.core.resource_cleanup import release_gpu_memory
from pretext_platform.core.types import StageSummary


def _convert_paths(obj):
    """Recursively convert Path objects to strings for JSON serialization."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, dict)):
        return [_convert_paths(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    elif hasattr(obj, "__fspath__"):
        return str(obj)
    return obj


def _experiment_dir(config: ExperimentConfig) -> Path:
    """Return the output directory for one experiment id."""

    return ensure_dir(config.output_root() / config.experiment_id())


def _write_stage_summary(experiment_dir: Path, filename: str, summary: StageSummary) -> None:
    """Persist one stage summary as JSON."""

    write_json(experiment_dir / filename, summary)


def run_stage1(config: ExperimentConfig) -> StageSummary:
    """Run only the Stage 1 Private Evolution generation flow."""

    from pretext_platform.algorithms.stage1 import run_private_evolution_stage
    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle

    dataset_bundle = load_dataset_bundle(config)
    model_paths = resolve_model_paths(config)
    experiment_dir = _experiment_dir(config)
    summary = run_private_evolution_stage(config, dataset_bundle, model_paths, experiment_dir / "stage1")
    _write_stage_summary(experiment_dir, "stage1_summary.json", summary)
    write_json(experiment_dir / "resolved_config.json", config.raw)
    return summary


def run_bootstrap(config: ExperimentConfig) -> StageSummary:
    """Run only the Stage 2 bootstrap flow."""

    from pretext_platform.algorithms.bootstrap import run_bootstrap_stage
    from pretext_platform.core.models import resolve_model_paths

    model_paths = resolve_model_paths(config)
    experiment_dir = _experiment_dir(config)
    summary = run_bootstrap_stage(config, model_paths, experiment_dir / "stage1", experiment_dir / "stage2")
    _write_stage_summary(experiment_dir, "stage2_summary.json", summary)
    write_json(experiment_dir / "resolved_config.json", config.raw)
    return summary


def run_eval_small(config: ExperimentConfig) -> StageSummary:
    """Run only the small-model downstream evaluation.

    Supports two modes:
    - "distilgpt2": DistilGPT2 with warm-start checkpoint (requires c4_checkpoint.pth)
    - "gpt2": Base GPT-2 from HuggingFace (no checkpoint needed, cross-platform)
    """

    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle

    dataset_bundle = load_dataset_bundle(config)
    model_paths = resolve_model_paths(config)
    experiment_dir = _experiment_dir(config)

    eval_cfg = config.eval_small
    eval_mode = eval_cfg.get("eval_mode", "gpt2")

    if eval_mode == "gpt2":
        from pretext_platform.evaluation.gpt2_eval import run_gpt2_eval
        summary = run_gpt2_eval(config, dataset_bundle, model_paths, experiment_dir / "stage2", experiment_dir / "eval_small")
    else:
        from pretext_platform.evaluation.distilgpt2_eval import run_distilgpt2_eval
        summary = run_distilgpt2_eval(config, dataset_bundle, model_paths, experiment_dir / "stage2", experiment_dir / "eval_small")

    _write_stage_summary(experiment_dir, "eval_small_summary.json", summary)
    write_json(experiment_dir / "resolved_config.json", config.raw)
    return summary


def run_eval_large(config: ExperimentConfig) -> StageSummary:
    """Run only the large-model downstream evaluation.

    The fixed Linux experiment environment supports only one large-eval mode:
    - "peft_lora": LLaMA-2-7B + PEFT LoRA
    """

    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle

    dataset_bundle = load_dataset_bundle(config)
    model_paths = resolve_model_paths(config)
    experiment_dir = _experiment_dir(config)

    eval_cfg = config.eval_large
    eval_mode = eval_cfg.get("eval_mode", "peft_lora")

    if eval_mode != "peft_lora":
        raise ValueError(
            "eval_large only supports eval_mode='peft_lora' on the fixed Linux server."
        )

    from pretext_platform.evaluation.llama2_eval import run_llama2_eval

    summary = run_llama2_eval(
        config, dataset_bundle, model_paths, experiment_dir / "stage2", experiment_dir / "eval_large"
    )

    _write_stage_summary(experiment_dir, "eval_large_summary.json", summary)
    write_json(experiment_dir / "resolved_config.json", config.raw)
    return summary


def run_glue_eval(config: ExperimentConfig) -> dict[str, StageSummary]:
    """Run GLUE downstream evaluation on synthetic data.

    Supports multiple GLUE classification tasks:
    - sst2: Binary sentiment classification (SST-2)
    - qqp: Quora Question Pairs
    - qnli: Question-answering NLI
    - imdb: Binary sentiment classification (IMDB)
    - rotten_tomatoes: Sentiment classification

    Reads synthetic corpus from Stage2 output (llama7b_text_syn.json).
    """
    from pretext_platform.evaluation.glue_classification_eval import run_glue_classification_eval
    from pretext_platform.core.models import resolve_model_paths

    model_paths = resolve_model_paths(config)
    experiment_dir = _experiment_dir(config)
    stage2_dir = experiment_dir / "stage2"

    eval_cfg = config.eval_glue
    summaries = {}

    # Get list of tasks to evaluate
    tasks = eval_cfg.get("tasks", ["sst2"])

    for task in tasks:
        summary = run_glue_classification_eval(
            config=config,
            model_paths=model_paths,
            stage2_dir=stage2_dir,
            output_dir=experiment_dir / "eval_glue",
            task_name=task,
        )
        summaries[f"glue_{task}"] = summary
        _write_stage_summary(experiment_dir, f"glue_{task}_summary.json", summary)

    glue_summary_payload = {
        task: {
            "best_accuracy": summaries[f"glue_{task}"].metrics.get("best_accuracy", 0.0),
            "correct": summaries[f"glue_{task}"].metrics.get("correct", 0),
            "total": summaries[f"glue_{task}"].metrics.get("total", 0),
            "data_source": summaries[f"glue_{task}"].metrics.get("data_source", "unknown"),
        }
        for task in tasks
        if f"glue_{task}" in summaries
    }
    write_json(ensure_dir(experiment_dir / "eval_glue") / "glue_summary.json", glue_summary_payload)
    write_json(experiment_dir / "resolved_config.json", config.raw)
    return summaries


def run_pipeline(config_or_path: ExperimentConfig | str | Path) -> dict[str, Any]:
    """Run the enabled stages for one experiment config."""

    config = config_or_path if isinstance(config_or_path, ExperimentConfig) else load_experiment_config(config_or_path)
    experiment_dir = _experiment_dir(config)
    write_json(experiment_dir / "resolved_config.json", config.raw)

    summaries: dict[str, StageSummary] = {}
    stage1_cfg = config.stage1
    bootstrap_cfg = config.bootstrap
    eval_small_cfg = config.eval_small
    eval_large_cfg = config.eval_large
    eval_glue_cfg = config.eval_glue

    if bool(stage1_cfg.get("enabled", True)):
        summaries["stage1"] = run_stage1(config)
        _write_stage_summary(experiment_dir, "stage1_summary.json", summaries["stage1"])
        release_gpu_memory()

    if bool(bootstrap_cfg.get("enabled", True)):
        summaries["stage2"] = run_bootstrap(config)
        _write_stage_summary(experiment_dir, "stage2_summary.json", summaries["stage2"])
        release_gpu_memory()

    if bool(eval_small_cfg.get("enabled", False)):
        summaries["eval_small"] = run_eval_small(config)
        _write_stage_summary(experiment_dir, "eval_small_summary.json", summaries["eval_small"])
        release_gpu_memory()

    if bool(eval_large_cfg.get("enabled", False)):
        summaries["eval_large"] = run_eval_large(config)
        _write_stage_summary(experiment_dir, "eval_large_summary.json", summaries["eval_large"])
        release_gpu_memory()

    if bool(eval_glue_cfg.get("enabled", False)):
        glue_summaries = run_glue_eval(config)
        summaries.update(glue_summaries)
        release_gpu_memory()

    summary_payload = {
        "experiment_id": config.experiment_id(),
        "experiment_dir": str(experiment_dir),
        "status": "SUCCESS",
        "stages": {k: _convert_paths(asdict(v)) if hasattr(v, "__dataclass_fields__") else v for k, v in summaries.items()},
    }
    write_json(experiment_dir / "metrics_summary.json", summary_payload)
    return summary_payload
