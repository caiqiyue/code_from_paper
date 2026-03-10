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


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger()
        self.repo_root = config.repo_root()
        self.output_root = ensure_dir(config.output_root())
        self.experiment_id = str(config.meta.get("experiment_id", config.path.stem))
        self.experiment_dir = ensure_dir(self.output_root / self.experiment_id)

    def _load_public_seed_samples(self) -> list[Any]:
        public_seed_path = self.config.resolve_path(self.config.data.get("public_seed_path"))
        if public_seed_path is None:
            return []
        return load_samples(
            public_seed_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="public_seed",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="server",
            prefix="seed",
        )

    def _load_client_contexts(self) -> list[ClientContext]:
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
        partitions = partition_samples(
            all_samples,
            num_clients=int(self.config.data.get("num_clients", 3)),
            max_samples_per_client=int(self.config.data.get("max_samples_per_client", 16)),
            validation_ratio=float(self.config.data.get("validation_ratio", 0.1)),
            seed=int(self.config.meta.get("seed", 42)),
        )
        retriever_cfg = self.config.retriever
        embedder = build_embedder(retriever_cfg.get("embedding_model"), self.repo_root)
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
        from thesis_platform import adapters  # noqa: F401

        public_seed_samples = self._load_public_seed_samples()
        client_contexts = self._load_client_contexts()

        generator = create("generator", str(self.config.generator.get("name")), self.config.generator, self.repo_root)
        scorer = create("scorer", str(self.config.scorer.get("name")), self.config.scorer, self.repo_root)
        retriever = create("retriever", str(self.config.retriever.get("name", "knn")), self.config.retriever, self.repo_root)
        critic = create("critic", str(self.config.critic.get("name", "none")), self.config.critic, self.repo_root)
        aggregator = create("aggregator", str(self.config.aggregator.get("name", "none")), self.config.aggregator, self.repo_root)
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
        for round_id in range(rounds):
            round_dir = ensure_dir(self.experiment_dir / f"round_{round_id:03d}")
            artifacts = round_runner.run_round(
                round_id=round_id,
                server_ctx=server_ctx,
                client_contexts=client_contexts,
                public_seed_samples=public_seed_samples,
                federation_cfg=self.config.federation,
                output_dir=round_dir,
            )
            server_ctx.prompt_text = artifacts.updated_prompt
            server_ctx.prompt_history.append(server_ctx.prompt_text)
            all_round_metrics.append({"round_id": round_id, **artifacts.round_metrics})

        summary = {
            "experiment_id": self.experiment_id,
            "round_count": rounds,
            "final_prompt": server_ctx.prompt_text,
            "round_metrics": all_round_metrics,
        }
        write_json(self.experiment_dir / "metrics_summary.json", summary)
        write_json(self.experiment_dir / "resolved_config.json", self.config.raw)
        write_text(self.experiment_dir / "config.yaml", json.dumps(self.config.raw, ensure_ascii=False, indent=2))
        return summary
