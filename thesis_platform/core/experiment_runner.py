from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from thesis_platform.core.config import ExperimentConfig
from thesis_platform.core.context import ClientContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, write_json, write_text
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.registry import create
from thesis_platform.core.round_runner import RoundRunner
from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples
from thesis_platform.models.embedding import build_embedder

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is declared in requirements but keep a fallback
    tqdm = None


class ExperimentRunner:
    """Top-level experiment orchestrator for one resolved experiment config."""

    def __init__(self, config: ExperimentConfig):
        """Prepare stable paths, logging, and output directories for one run."""

        self.config = config
        self.logger = get_logger()
        self.repo_root = config.repo_root()
        self.output_root = ensure_dir(config.output_root())  # Ensure the shared output root exists early.
        self.experiment_id = str(config.meta.get("experiment_id", config.path.stem))
        self.experiment_dir = ensure_dir(self.output_root / self.experiment_id)

    def _load_public_seed_samples(self) -> list[Any]:
        """Load the public seed pool used by the server-side generator."""

        public_seed_path = self.config.resolve_path(self.config.data.get("public_seed_path"))
        if public_seed_path is None:
            self.logger.info("No public seed dataset configured.")
            return []
        samples = load_samples(
            public_seed_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="public_seed",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="server",
            prefix="seed",
        )
        self.logger.info("Loaded %d public seed samples from %s", len(samples), public_seed_path)
        return samples

    def _load_client_contexts(self) -> list[ClientContext]:
        """Load the dataset and partition it into per-client runtime contexts."""

        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        if train_path is None:
            raise ValueError("data.train_path must be configured.")
        all_samples = load_samples(
            train_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="real",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="raw",
            prefix="real",
        )
        self.logger.info("Loaded %d private training samples from %s", len(all_samples), train_path)
        partitions = partition_samples(  # Split one dataset into stable per-client buckets.
            all_samples,
            num_clients=int(self.config.data.get("num_clients", 3)),
            max_samples_per_client=int(self.config.data.get("max_samples_per_client", 16)),
            validation_ratio=float(self.config.data.get("validation_ratio", 0.1)),
            seed=int(self.config.meta.get("seed", 42)),
        )
        retriever_cfg = self.config.retriever
        embedder = build_embedder(retriever_cfg.get("embedding_model"), self.repo_root)  # Share one embedder across clients.
        self.logger.info(
            "Partitioned dataset into %d clients with max %d samples/client and validation_ratio=%.2f",
            len(partitions),
            int(self.config.data.get("max_samples_per_client", 16)),
            float(self.config.data.get("validation_ratio", 0.1)),
        )
        self.logger.info("Embedder backend: %s", type(embedder).__name__)
        contexts: list[ClientContext] = []
        for idx, bucket in enumerate(partitions):
            client_id = f"client_{idx}"
            contexts.append(
                ClientContext(
                    client_id=client_id,
                    train_samples=bucket["train"],
                    validation_samples=bucket["validation"],
                    all_samples=bucket["all"],
                    embedder=embedder,
                    config=self.config.raw,
                )
            )
        return contexts

    def run(self) -> dict[str, Any]:
        """Run the configured experiment end to end and return the summary payload."""

        from thesis_platform import adapters  # noqa: F401

        self.logger.info("Experiment %s | starting with config %s", self.experiment_id, self.config.path)
        public_seed_samples = self._load_public_seed_samples()
        client_contexts = self._load_client_contexts()

        generator = create("generator", str(self.config.generator.get("name")), self.config.generator, self.repo_root)
        scorer = create("scorer", str(self.config.scorer.get("name")), self.config.scorer, self.repo_root)
        retriever = create("retriever", str(self.config.retriever.get("name", "knn")), self.config.retriever, self.repo_root)
        critic = create("critic", str(self.config.critic.get("name", "none")), self.config.critic, self.repo_root)
        aggregator = create("aggregator", str(self.config.aggregator.get("name", "none")), self.config.aggregator, self.repo_root)  # Build adapters from the registry.
        self.logger.info(
            "Experiment %s | adapters generator=%s scorer=%s retriever=%s critic=%s aggregator=%s",
            self.experiment_id,
            type(generator).__name__,
            type(scorer).__name__,
            type(retriever).__name__,
            type(critic).__name__,
            type(aggregator).__name__,
        )
        round_runner = RoundRunner(
            generator=generator,
            scorer=scorer,
            retriever=retriever,
            critic=critic,
            aggregator=aggregator,
        )

        initial_prompt = str(
            self.config.generator.get(
                "initial_prompt",
                f"Generate text consistent with dataset {self.config.data.get('dataset_name', 'dataset')}.",
            )
        )
        server_ctx = ServerContext(
            experiment_id=self.experiment_id,
            prompt_text=initial_prompt,
            prompt_history=[initial_prompt],
            config=self.config.raw,
            output_dir=self.experiment_dir,
        )

        rounds = int(self.config.federation.get("rounds", 1))
        all_round_metrics: list[dict[str, Any]] = []
        self.logger.info("Experiment %s | executing %d rounds", self.experiment_id, rounds)
        progress = tqdm(range(rounds), total=rounds, desc=f"{self.experiment_id}", unit="round") if tqdm is not None else None
        round_iter = progress if progress is not None else range(rounds)
        for round_id in round_iter:
            self.logger.info("Round %d/%d | start", round_id + 1, rounds)
            round_dir = ensure_dir(self.experiment_dir / f"round_{round_id:03d}")
            artifacts = round_runner.run_round(  # Execute one full generator-to-aggregator loop.
                round_id=round_id,
                server_ctx=server_ctx,
                client_contexts=client_contexts,
                public_seed_samples=public_seed_samples,
                federation_cfg=self.config.federation,
                output_dir=round_dir,
            )
            server_ctx.prompt_text = artifacts.updated_prompt  # Feed the updated prompt into the next round.
            server_ctx.prompt_history.append(server_ctx.prompt_text)
            all_round_metrics.append({"round_id": round_id, **artifacts.round_metrics})
            if progress is not None:
                progress.set_postfix(
                    generated=artifacts.round_metrics.get("generated_count", 0),
                    critiques=artifacts.round_metrics.get("critique_count", 0),
                )
            self.logger.info(
                "Round %d/%d | complete | generated=%s selected_bad=%d critiques=%d output=%s",
                round_id + 1,
                rounds,
                artifacts.round_metrics.get("generated_count", 0),
                len(artifacts.selected_bad_samples),
                len(artifacts.critiques),
                round_dir,
            )
        if progress is not None:
            progress.close()

        summary = {
            "experiment_id": self.experiment_id,
            "round_count": rounds,
            "final_prompt": server_ctx.prompt_text,
            "round_metrics": all_round_metrics,
        }
        write_json(self.experiment_dir / "metrics_summary.json", summary)
        write_json(self.experiment_dir / "resolved_config.json", self.config.raw)
        write_text(self.experiment_dir / "config.yaml", json.dumps(self.config.raw, ensure_ascii=False, indent=2))
        self.logger.info("Experiment %s | finished | summary=%s", self.experiment_id, self.experiment_dir / "metrics_summary.json")
        return summary
