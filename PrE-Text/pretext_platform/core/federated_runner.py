from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.federated_artifacts import (
    client_dir,
    ensure_experiment_dir,
    server_stage2_dir,
    write_metrics_summary,
    write_privacy_ledger,
    write_stage_summary,
)
from pretext_platform.core.federated_partition import build_federated_client_partitions
from pretext_platform.core.federated_privacy import build_privacy_summary, make_round_privacy_record
from pretext_platform.core.io_utils import write_json
from pretext_platform.core.resource_cleanup import release_gpu_memory
from pretext_platform.core.run_state import write_failure_artifacts, write_run_state
from pretext_platform.core.types import StageSummary


Stage1Runner = Callable[..., tuple[StageSummary, list[str]]]
BootstrapRunner = Callable[..., StageSummary]
Stage2Runner = Callable[..., StageSummary]
PartitionFn = Callable[[ExperimentConfig], dict[str, dict[str, list[str]]]]


def resolve_model_paths(config: ExperimentConfig):
    from pretext_platform.core.models import resolve_model_paths as _resolve_model_paths

    return _resolve_model_paths(config)


def load_initialization_texts(path: Path, *, min_words: int) -> list[str]:
    from pretext_platform.data.loaders import load_initialization_texts as _load_initialization_texts

    return _load_initialization_texts(path, min_words=min_words)


def run_private_evolution_stage(*args, **kwargs):
    from pretext_platform.algorithms.stage1 import run_private_evolution_stage as _run_private_evolution_stage

    return _run_private_evolution_stage(*args, **kwargs)


def build_bootstrap_prompts(seed_texts: list[str], *, num_prompts: int, seed: int) -> list[str]:
    from pretext_platform.algorithms.bootstrap import build_bootstrap_prompts as _build_bootstrap_prompts

    return _build_bootstrap_prompts(seed_texts, num_prompts=num_prompts, seed=seed)


def generate_bootstrapped_samples(prompt_list: list[str], model_path: Path, bootstrap_cfg: dict) -> list[str]:
    from pretext_platform.algorithms.bootstrap import (
        generate_bootstrapped_samples as _generate_bootstrapped_samples,
    )

    return _generate_bootstrapped_samples(prompt_list, model_path, bootstrap_cfg)


def generate_bootstrapped_samples_hf(prompt_list: list[str], model_path: Path, bootstrap_cfg: dict) -> list[str]:
    from pretext_platform.algorithms.bootstrap import (
        generate_bootstrapped_samples_hf as _generate_bootstrapped_samples_hf,
    )

    return _generate_bootstrapped_samples_hf(prompt_list, model_path, bootstrap_cfg)


def generate_bootstrapped_samples_vllm(prompt_list: list[str], model_path: Path, bootstrap_cfg: dict) -> list[str]:
    from pretext_platform.algorithms.bootstrap import (
        generate_bootstrapped_samples_vllm as _generate_bootstrapped_samples_vllm,
    )

    return _generate_bootstrapped_samples_vllm(prompt_list, model_path, bootstrap_cfg)


def _default_stage1_runner(
    *,
    config: ExperimentConfig,
    client_id: str,
    round_id: int,
    client_partition: dict[str, list[str]],
    output_dir: Path,
) -> tuple[StageSummary, list[str]]:
    from pretext_platform.core.types import DatasetBundle

    initialization_path = config.resolve_path(config.data.get("initialization_path"))
    if initialization_path is None:
        raise ValueError("data.initialization_path must be configured for federated Stage1.")
    initialization_texts = load_initialization_texts(
        initialization_path,
        min_words=int(config.data.get("initialization_min_words", 20)),
    )

    dataset_bundle = DatasetBundle(
        dataset_name=str(config.data.get("dataset_name", "dataset")),
        train_texts=list(client_partition.get("train_texts", [])),
        eval_texts=list(client_partition.get("eval_texts", [])),
        initialization_texts=initialization_texts,
        train_client_count=1,
        raw_train_sample_count=len(client_partition.get("train_texts", [])),
        sampled_train_sample_count=len(client_partition.get("train_texts", [])),
        max_samples_per_client=len(client_partition.get("train_texts", [])),
        metadata={"client_id": client_id, "round_id": round_id},
    )
    summary = run_private_evolution_stage(config, dataset_bundle, resolve_model_paths(config), output_dir)
    surviving_files = list(summary.artifacts.get("surviving_files", []))
    if not surviving_files:
        raise ValueError(f"Stage1 runner for {client_id} round {round_id} did not produce surviving_files.")
    surviving_path = Path(surviving_files[-1])
    surviving_texts = json.loads(surviving_path.read_text(encoding="utf-8"))
    summary.metrics["surviving_count"] = len(surviving_texts)
    return summary, surviving_texts


def _default_bootstrap_runner(
    *,
    config: ExperimentConfig,
    merged_surviving_texts: list[str],
    round_id: int,
    server_output_dir: Path,
) -> StageSummary:
    if len(merged_surviving_texts) < 3:
        raise ValueError("Federated bootstrap requires at least 3 surviving texts to build prompts.")

    bootstrap_inputs_path = server_output_dir / "bootstrap_inputs.json"
    write_json(bootstrap_inputs_path, merged_surviving_texts)

    prompt_list = build_bootstrap_prompts(
        merged_surviving_texts,
        num_prompts=int(config.bootstrap.get("num_prompts", 50000)),
        seed=int(config.meta.get("seed", 42)),
    )
    bootstrap_prompt_path = server_output_dir / "bootstrap_prompts.json"
    write_json(bootstrap_prompt_path, prompt_list)

    return StageSummary(
        stage_name="bootstrap",
        output_dir=server_output_dir,
        artifacts={
            "bootstrap_inputs_path": str(bootstrap_inputs_path),
            "bootstrap_prompt_path": str(bootstrap_prompt_path),
        },
        metrics={
            "round_id": round_id,
            "seed_text_count": len(merged_surviving_texts),
            "prompt_count": len(prompt_list),
        },
    )


def _default_stage2_runner(
    *,
    config: ExperimentConfig,
    merged_surviving_texts: list[str],
    round_id: int,
    server_output_dir: Path,
) -> StageSummary:
    bootstrap_cfg = config.bootstrap
    model_paths = resolve_model_paths(config)
    bootstrap_model = str(bootstrap_cfg.get("generator_model", "llama2_7b"))
    if bootstrap_model == "llama2_7b":
        model_path = model_paths.llama2_7b
    elif bootstrap_model == "distilgpt2":
        model_path = model_paths.distilgpt2
    else:
        raise ValueError(
            f"Federated Stage2 only supports generator_model='llama2_7b' or 'distilgpt2', got '{bootstrap_model}'."
        )

    prompt_list = build_bootstrap_prompts(
        merged_surviving_texts,
        num_prompts=int(bootstrap_cfg.get("num_prompts", 50000)),
        seed=int(config.meta.get("seed", 42)),
    )
    backend = str(bootstrap_cfg.get("generator_backend", "huggingface")).strip().lower()
    if backend in {"hf", "huggingface"}:
        generated_outputs = generate_bootstrapped_samples_hf(prompt_list, model_path, bootstrap_cfg)
    elif backend == "vllm":
        generated_outputs = generate_bootstrapped_samples_vllm(prompt_list, model_path, bootstrap_cfg)
    else:
        generated_outputs = generate_bootstrapped_samples(prompt_list, model_path, bootstrap_cfg)

    output_path = server_output_dir / "llama7b_text_syn.json"
    write_json(output_path, generated_outputs)
    return StageSummary(
        stage_name="stage2",
        output_dir=server_output_dir,
        artifacts={"synthetic_corpus_path": str(output_path)},
        metrics={
            "round_id": round_id,
            "seed_text_count": len(merged_surviving_texts),
            "prompt_count": len(prompt_list),
            "generated_count": len(generated_outputs),
        },
    )


class FederatedPretextRunner:
    """Orchestrate the federated PrE-Text generation flow."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        partition_fn: PartitionFn = build_federated_client_partitions,
        stage1_runner: Stage1Runner = _default_stage1_runner,
        bootstrap_runner: BootstrapRunner = _default_bootstrap_runner,
        stage2_runner: Stage2Runner = _default_stage2_runner,
    ) -> None:
        self.config = config
        self.partition_fn = partition_fn
        self.stage1_runner = stage1_runner
        self.bootstrap_runner = bootstrap_runner
        self.stage2_runner = stage2_runner

    def run(self) -> dict[str, Any]:
        experiment_dir = ensure_experiment_dir(self.config.output_root(), self.config.experiment_id())
        write_json(experiment_dir / "resolved_config.json", self.config.raw)
        write_run_state(experiment_dir, self.config, status="running", phase="initializing")

        rounds = int(self.config.federation.get("rounds", 1))
        round_summaries: list[dict[str, Any]] = []
        privacy_rounds: list[dict[str, Any]] = []
        final_synthetic_corpus_path = ""
        final_synthetic_sample_count = 0
        current_phase = "initializing"

        try:
            current_phase = "partitioning"
            write_run_state(experiment_dir, self.config, status="running", phase=current_phase)
            partitions = self.partition_fn(self.config)

            for round_id in range(rounds):
                client_summaries: dict[str, StageSummary] = {}
                merged_surviving_texts: list[str] = []

                for client_id in sorted(partitions.keys()):
                    current_phase = f"round_{round_id}_client_stage1"
                    write_run_state(experiment_dir, self.config, status="running", phase=current_phase)
                    current_client_dir = client_dir(experiment_dir, round_id, client_id)
                    stage1_summary, surviving_texts = self.stage1_runner(
                        config=self.config,
                        client_id=client_id,
                        round_id=round_id,
                        client_partition=partitions[client_id],
                        output_dir=current_client_dir,
                    )
                    write_stage_summary(current_client_dir / "stage1_summary.json", stage1_summary)
                    client_summaries[client_id] = stage1_summary
                    merged_surviving_texts.extend(surviving_texts)
                    release_gpu_memory()

                current_phase = f"round_{round_id}_bootstrap"
                write_run_state(experiment_dir, self.config, status="running", phase=current_phase)
                current_server_dir = server_stage2_dir(experiment_dir, round_id)
                bootstrap_summary = self.bootstrap_runner(
                    config=self.config,
                    merged_surviving_texts=merged_surviving_texts,
                    round_id=round_id,
                    server_output_dir=current_server_dir,
                )
                release_gpu_memory()

                current_phase = f"round_{round_id}_stage2"
                write_run_state(experiment_dir, self.config, status="running", phase=current_phase)
                stage2_summary = self.stage2_runner(
                    config=self.config,
                    merged_surviving_texts=merged_surviving_texts,
                    round_id=round_id,
                    server_output_dir=current_server_dir,
                )
                write_stage_summary(current_server_dir / "bootstrap_summary.json", bootstrap_summary)
                write_stage_summary(current_server_dir / "stage2_summary.json", stage2_summary)
                release_gpu_memory()

                round_summary = {
                    "round_id": round_id,
                    "client_count": len(client_summaries),
                    "merged_surviving_count": len(merged_surviving_texts),
                    "server_stage2_sample_count": int(stage2_summary.metrics.get("generated_count", 0)),
                    "server_stage2_output": Path(
                        str(stage2_summary.artifacts.get("synthetic_corpus_path", ""))
                    ).as_posix(),
                }
                round_summaries.append(round_summary)

                privacy_rounds.append(
                    make_round_privacy_record(
                        round_id=round_id,
                        client_summaries=client_summaries,
                        merged_surviving_count=len(merged_surviving_texts),
                        server_stage2_sample_count=int(stage2_summary.metrics.get("generated_count", 0)),
                    )
                )

                final_synthetic_corpus_path = Path(
                    str(stage2_summary.artifacts.get("synthetic_corpus_path", ""))
                ).as_posix()
                final_synthetic_sample_count = int(stage2_summary.metrics.get("generated_count", 0))

            privacy_payload = {
                "schema_version": 1,
                "experiment_id": self.config.experiment_id(),
                "rounds": privacy_rounds,
                "final_privacy_summary": build_privacy_summary(privacy_rounds),
            }
            write_privacy_ledger(experiment_dir, privacy_payload)

            summary_payload = {
                "experiment_id": self.config.experiment_id(),
                "experiment_dir": str(experiment_dir),
                "status": "SUCCESS",
                "round_count": rounds,
                "completed_rounds": rounds,
                "final_synthetic_corpus_path": final_synthetic_corpus_path,
                "final_synthetic_sample_count": final_synthetic_sample_count,
                "round_summaries": round_summaries,
                "privacy_summary": privacy_payload["final_privacy_summary"],
            }
            if final_synthetic_corpus_path:
                final_stage2_path = Path(final_synthetic_corpus_path)
                compat_stage2_dir = experiment_dir / "stage2"
                compat_stage2_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final_stage2_path, compat_stage2_dir / "llama7b_text_syn.json")

                final_round_id = max(rounds - 1, 0)
                final_round_stage2_summary = server_stage2_dir(experiment_dir, final_round_id) / "stage2_summary.json"
                if final_round_stage2_summary.exists():
                    shutil.copy2(final_round_stage2_summary, experiment_dir / "stage2_summary.json")
            write_metrics_summary(experiment_dir, summary_payload)
            write_run_state(experiment_dir, self.config, status="completed", phase="finished")
            return summary_payload
        except Exception as exc:
            write_failure_artifacts(experiment_dir, self.config, exc, phase=current_phase)
            raise


def run_federated_pipeline(config: ExperimentConfig) -> dict[str, Any]:
    """Run one federated PrE-Text experiment."""

    return FederatedPretextRunner(config).run()
