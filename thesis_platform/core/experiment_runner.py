from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

from thesis_platform.core.artifact_manifest import (
    ARTIFACT_SCHEMA_VERSION,
    build_experiment_manifest,
    build_round_manifest,
)
from thesis_platform.core.checkpoint import CheckpointManager
from thesis_platform.core.config import ExperimentConfig
from thesis_platform.core.context import ClientContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, write_json, write_text
from thesis_platform.core.logging_utils import get_logger, setup_experiment_file_logger
from thesis_platform.core.preflight import validate_preflight
from thesis_platform.core.privacy import PrivacyLedger, PrivacyPolicy
from thesis_platform.core.registry import create
from thesis_platform.core.round_runner import RoundRunner
from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples
from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager
from thesis_platform.models.backends import build_text_backend
from thesis_platform.models.embedding import build_embedder

try:
    from tqdm import tqdm
except (
    ModuleNotFoundError
):  # pragma: no cover - tqdm is declared in requirements but keep a fallback
    tqdm = None


class ExperimentRunner:
    """Top-level experiment orchestrator for one resolved experiment config."""

    def __init__(self, config: ExperimentConfig):
        """Prepare stable paths, logging, and output directories for one run."""

        self.config = config
        self.logger = get_logger()
        self.repo_root = config.repo_root()
        self.output_root = ensure_dir(config.output_root())
        self.experiment_id = str(config.meta.get("experiment_id", config.path.stem))
        # Append timestamp to experiment directory name for uniqueness
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = ensure_dir(self.output_root / f"{self.experiment_id}_{timestamp}")
        # Set up file logging for this experiment
        setup_experiment_file_logger(self.experiment_dir, name="thesis_platform")
        self.logger.info("Experiment output directory: %s", self.experiment_dir)

    def _build_text_backends(self) -> tuple[Any, Any]:
        """Build shared client/server text backends when configured."""

        llm_cfg = self.config.llm
        client_cfg = dict(llm_cfg.get("client", {}))
        server_cfg = dict(llm_cfg.get("server", {}))

        client_backend = (
            build_text_backend(
                {**client_cfg, "role": "client"}, repo_root=self.repo_root
            )
            if client_cfg
            else None
        )
        server_backend = (
            build_text_backend(
                {**server_cfg, "role": "server"}, repo_root=self.repo_root
            )
            if server_cfg
            else None
        )
        return client_backend, server_backend

    def _load_public_seed_samples(self) -> list[Any]:
        """Load the public seed pool used by the server-side generator."""

        public_seed_path = self.config.resolve_path(
            self.config.data.get("public_seed_path")
        )
        if public_seed_path is None:
            self.logger.info("No public seed dataset configured.")
            return []
        public_seed_sample_format = str(
            self.config.data.get(
                "public_seed_sample_format",
                self.config.data.get("sample_format", "raw_text"),
            )
        )
        samples = load_samples(
            public_seed_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="public_seed",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="server",
            prefix="seed",
            sample_format=public_seed_sample_format,
            limit=(
                int(self.config.data.get("max_public_seed_samples"))
                if self.config.data.get("max_public_seed_samples") not in (None, "")
                else None
            ),
        )
        self.logger.info(
            "Loaded %d public seed samples from %s", len(samples), public_seed_path
        )
        return samples

    def _load_client_contexts(self, *, client_backend: Any) -> list[ClientContext]:
        """Load the dataset and partition it into per-client runtime contexts."""

        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        if train_path is None:
            raise ValueError("data.train_path must be configured.")
        sample_format = str(self.config.data.get("sample_format", "raw_text"))
        all_samples = load_samples(
            train_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="real",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="raw",
            prefix="real",
            sample_format=sample_format,
            limit=(
                int(self.config.data.get("train_limit"))
                if self.config.data.get("train_limit") not in (None, "")
                else None
            ),
        )
        self.logger.info(
            "Loaded %d private training samples from %s", len(all_samples), train_path
        )
        partitions = partition_samples(
            all_samples,
            num_clients=int(self.config.data.get("num_clients", 3)),
            max_samples_per_client=int(
                self.config.data.get("max_samples_per_client", 16)
            ),
            validation_ratio=float(self.config.data.get("validation_ratio", 0.1)),
            seed=int(self.config.meta.get("seed", 42)),
            strategy=str(
                self.config.data.get("partition_strategy", "shuffle_round_robin")
            ),
        )
        retriever_cfg = self.config.retriever
        embedder = build_embedder(
            retriever_cfg.get("embedding_model"),
            self.repo_root,
            allow_fallback=bool(retriever_cfg.get("allow_hashing_fallback", True)),
        )
        self.logger.info(
            "Partitioned dataset into %d clients with max %d samples/client and validation_ratio=%.2f",
            len(partitions),
            int(self.config.data.get("max_samples_per_client", 16)),
            float(self.config.data.get("validation_ratio", 0.1)),
        )
        self.logger.info(
            "Embedder backend: %s",
            getattr(embedder, "backend_name", type(embedder).__name__),
        )

        all_partition_samples = [
            sample for partition in partitions for sample in partition["all"]
        ]
        contexts: list[ClientContext] = []
        objective_type = str(
            self.config.scorer.get(
                "objective",
                "pair_alignment"
                if str(self.config.scorer.get("name", "")).lower() == "ira"
                else "domain_probe",
            )
        )
        for idx, bucket in enumerate(partitions):
            client_id = f"client_{idx}"
            bucket_sample_ids = {sample.sample_id for sample in bucket["all"]}
            negative_samples = [
                sample
                for sample in all_partition_samples
                if sample.sample_id not in bucket_sample_ids
            ]
            contexts.append(
                ClientContext(
                    client_id=client_id,
                    train_samples=bucket["train"],
                    validation_samples=bucket["validation"],
                    all_samples=bucket["all"],
                    embedder=embedder,
                    config=self.config.raw,
                    negative_samples=negative_samples,
                    text_backend=client_backend,
                    objective_type=objective_type,
                )
            )
        return contexts

    def run(self, resume: bool = False) -> dict[str, Any]:
        """Run the configured experiment end to end and return the summary payload.

        Args:
            resume: If True, attempt to resume from the latest checkpoint.

        Returns:
            Summary dictionary with experiment results.
        """

        from thesis_platform import adapters  # noqa: F401

        self.logger.info(
            "Experiment %s | starting with config %s",
            self.experiment_id,
            self.config.path,
        )
        validate_preflight(self.config)
        privacy_policy = PrivacyPolicy.from_config(self.config.privacy)
        privacy_ledger = PrivacyLedger(policy=privacy_policy)
        client_backend, server_backend = self._build_text_backends()
        public_seed_samples = self._load_public_seed_samples()
        client_contexts = self._load_client_contexts(client_backend=client_backend)

        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(
            output_dir=self.experiment_dir,
            max_checkpoints=3,
            save_artifacts=True,
        )

        # Try to resume from checkpoint
        start_round = 0
        checkpoint_data = None
        restored_experiment_state = None
        privacy_ledger_data = None
        if resume:
            checkpoint = checkpoint_mgr.load_checkpoint()
            if checkpoint:
                start_round = checkpoint.get("round_id", -1) + 1
                checkpoint_data = checkpoint.get("server_ctx_data")
                restored_experiment_state = checkpoint.get("experiment_state", {})
                privacy_ledger_data = checkpoint.get("privacy_ledger_data")
                self.logger.info(
                    "Resuming experiment %s from round %d",
                    self.experiment_id,
                    start_round,
                )
            else:
                self.logger.info("No checkpoint found, starting from round 0")

        # Restore privacy ledger from checkpoint or create fresh
        if privacy_ledger_data is not None:
            privacy_ledger = PrivacyLedger.restore_from_report(privacy_ledger_data)
            self.logger.info("PrivacyLedger restored from checkpoint")
        else:
            privacy_ledger = PrivacyLedger(policy=privacy_policy)

        generator = create(
            "generator",
            str(self.config.generator.get("name")),
            self.config.generator,
            self.repo_root,
        )
        scorer = create(
            "scorer",
            str(self.config.scorer.get("name")),
            self.config.scorer,
            self.repo_root,
        )
        retriever = create(
            "retriever",
            str(self.config.retriever.get("name", "knn")),
            self.config.retriever,
            self.repo_root,
        )
        critic = create(
            "critic",
            str(self.config.critic.get("name", "none")),
            self.config.critic,
            self.repo_root,
        )
        aggregator = create(
            "aggregator",
            str(self.config.aggregator.get("name", "none")),
            self.config.aggregator,
            self.repo_root,
        )
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

        # Restore from checkpoint if resuming
        if checkpoint_data is not None:
            server_ctx = ServerContext.restore_from_checkpoint(
                checkpoint_data=checkpoint_data,
                config=self.config.raw,
                output_dir=self.experiment_dir,
                text_backend=server_backend,
            )
            self.logger.info("ServerContext restored from checkpoint")
        else:
            server_ctx = ServerContext(
                experiment_id=self.experiment_id,
                prompt_text=initial_prompt,
                prompt_history=[initial_prompt],
                config=self.config.raw,
                output_dir=self.experiment_dir,
                text_backend=server_backend,
                base_prompt=initial_prompt,
            )

        rounds = int(self.config.federation.get("rounds", 1))

        # Restore completed round data if resuming
        if restored_experiment_state is not None:
            all_round_metrics = list(restored_experiment_state.get("round_metrics", []))
            round_manifests = list(restored_experiment_state.get("round_manifests", []))
            self.logger.info(
                "Restored %d completed rounds from checkpoint",
                len(all_round_metrics),
            )
        else:
            all_round_metrics = []
            round_manifests = []

        last_artifacts = None
        self.logger.info(
            "Experiment %s | executing rounds %d to %d (of %d total)",
            self.experiment_id,
            start_round,
            rounds - 1,
            rounds,
        )
        # Build the round range - skip already completed rounds when resuming
        remaining_rounds = rounds - start_round
        progress = (
            tqdm(
                range(remaining_rounds),
                total=remaining_rounds,
                desc=f"{self.experiment_id} (resumed from r{start_round})" if start_round > 0 else f"{self.experiment_id}",
                unit="round",
            )
            if tqdm is not None
            else None
        )
        round_iter = range(start_round, rounds)
        for round_id in round_iter:
            self.logger.info("Round %d/%d | start", round_id + 1, rounds)
            round_dir = ensure_dir(self.experiment_dir / f"round_{round_id:03d}")
            artifacts = round_runner.run_round(
                round_id=round_id,
                server_ctx=server_ctx,
                client_contexts=client_contexts,
                public_seed_samples=public_seed_samples,
                federation_cfg=self.config.federation,
                output_dir=round_dir,
                prototype_cfg=self.config.prototype,
                routing_cfg=self.config.routing,
                privacy_ledger=privacy_ledger,
            )
            last_artifacts = artifacts
            server_ctx.generated_history.append(
                [sample.rendered_text() for sample in artifacts.generated_samples]
            )
            server_ctx.prompt_text = artifacts.updated_prompt
            server_ctx.prompt_history.append(server_ctx.prompt_text)
            all_round_metrics.append({"round_id": round_id, **artifacts.round_metrics})
            round_manifests.append(
                build_round_manifest(
                    experiment_id=self.experiment_id,
                    round_id=round_id,
                    round_dir=round_dir,
                )
            )
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

            # Save checkpoint after each round
            try:
                checkpoint_mgr.save_checkpoint(
                    round_id=round_id,
                    experiment_state={
                        "round_metrics": all_round_metrics,
                        "round_manifests": round_manifests,
                        "prompt_history": server_ctx.prompt_history,
                    },
                    server_ctx=server_ctx,
                    privacy_ledger=privacy_ledger,
                    config=self.config.raw,
                )
                self.logger.debug("Checkpoint saved for round %d", round_id)
            except Exception as e:
                self.logger.warning(
                    "Failed to save checkpoint for round %d: %s", round_id, e
                )

        if progress is not None:
            progress.close()

        resolved_config_path = self.experiment_dir / "resolved_config.json"
        metrics_summary_path = self.experiment_dir / "metrics_summary.json"
        privacy_ledger_path = self.experiment_dir / "privacy_ledger.json"
        artifact_manifest_path = self.experiment_dir / "artifact_manifest.json"
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "metrics_summary",
            "experiment_id": self.experiment_id,
            "round_count": rounds,
            "final_prompt": server_ctx.prompt_text,
            "round_metrics": all_round_metrics,
            "privacy": privacy_ledger.summary(),
            "artifacts": {
                "resolved_config_path": str(resolved_config_path),
                "privacy_ledger_path": str(privacy_ledger_path),
                "artifact_manifest_path": str(artifact_manifest_path),
            },
        }
        downstream_cfg = self.config.downstream_eval
        downstream_summary: dict[str, Any] | None = None
        if last_artifacts is not None and bool(downstream_cfg.get("enabled")):
            downstream_root = ensure_dir(self.experiment_dir / "downstream_eval")
            downstream_summary = DownstreamEvalManager(
                self.config,
                experiment_id=self.experiment_id,
                output_dir=downstream_root,
            ).run(
                [
                    sample.rendered_text()
                    for sample in last_artifacts.client_assigned_samples
                ]
            )
            summary["synthetic_corpus_path"] = downstream_summary.get(
                "synthetic_corpus_path"
            )
            summary["downstream_eval"] = downstream_summary
            summary["baseline_summaries"] = downstream_summary.get(
                "baseline_summaries", {}
            )
        else:
            summary["downstream_eval"] = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "downstream_eval_summary",
                "experiment_id": self.experiment_id,
                "enabled": False,
                "status": "disabled",
                "kind": downstream_cfg.get("kind", "none"),
            }

        write_json(
            resolved_config_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "resolved_config",
                "experiment_id": self.experiment_id,
                "config_path": str(self.config.path),
                "config": self.config.raw,
            },
        )
        write_json(
            privacy_ledger_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "privacy_ledger",
                "experiment_id": self.experiment_id,
                **privacy_ledger.report(),
            },
        )
        write_json(metrics_summary_path, summary)
        write_json(
            artifact_manifest_path,
            build_experiment_manifest(
                experiment_id=self.experiment_id,
                experiment_dir=self.experiment_dir,
                resolved_config_path=resolved_config_path,
                metrics_summary_path=metrics_summary_path,
                privacy_ledger_path=privacy_ledger_path,
                round_manifests=round_manifests,
                downstream_summary=downstream_summary,
            ),
        )
        write_text(
            self.experiment_dir / "config.yaml",
            json.dumps(self.config.raw, ensure_ascii=False, indent=2),
        )
        self.logger.info(
            "Experiment %s | finished | summary=%s",
            self.experiment_id,
            self.experiment_dir / "metrics_summary.json",
        )
        return summary
