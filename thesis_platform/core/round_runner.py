from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, write_json, write_jsonl, write_text
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.prompt_updater import apply_prompt_update
from thesis_platform.core.schemas import Critique, PromptUpdate, ScoredSample, Sample
from thesis_platform.core.selector import select_top_k
from thesis_platform.evaluation.metrics import compute_critique_metrics, compute_generation_metrics, compute_system_metrics


@dataclass(slots=True)
class RoundArtifacts:
    """Bundle of the main artifacts produced by one completed round."""

    generated_samples: list[Sample]
    scored_samples: list[ScoredSample]
    selected_bad_samples: list[ScoredSample]
    retrieved_pairs: list[Any]
    critiques: list[Critique]
    prompt_update: PromptUpdate | None
    updated_prompt: str
    round_metrics: dict[str, Any]


class RoundRunner:
    """Execute the per-round server/client orchestration logic."""

    def __init__(self, *, generator: Any, scorer: Any, retriever: Any, critic: Any, aggregator: Any):
        """Store the adapter instances that participate in one round."""

        self.generator = generator
        self.scorer = scorer
        self.retriever = retriever
        self.critic = critic
        self.aggregator = aggregator
        self.logger = get_logger()

    def run_round(
        self,
        *,
        round_id: int,
        server_ctx: ServerContext,
        client_contexts: list[ClientContext],
        public_seed_samples: list[Sample],
        federation_cfg: dict[str, Any],
        output_dir: Path,
    ) -> RoundArtifacts:
        """Run one full federation round and persist its artifacts to disk."""

        ensure_dir(output_dir)
        display_round = round_id + 1
        round_ctx = RoundContext(
            round_id=round_id,
            prompt_text=server_ctx.prompt_text,
            public_seed_samples=public_seed_samples,
            config=federation_cfg,
            output_dir=output_dir,
            text_backend=server_ctx.text_backend,
        )

        server_start = time.perf_counter()
        self.logger.info("Round %d | generation start", display_round)
        generated_samples = self.generator.generate(round_ctx)
        server_after_generation = time.perf_counter()
        self.logger.info("Round %d | generation complete | generated=%d", display_round, len(generated_samples))

        scored_samples: list[ScoredSample] = []
        critiques: list[Critique] = []
        selected_bad_samples: list[ScoredSample] = []
        retrieved_pairs: list[Any] = []
        client_latency_total = 0.0
        probe_metrics: dict[str, Any] = {}

        self.logger.info("Round %d | client scoring start across %d clients", display_round, len(client_contexts))
        for client_ctx in client_contexts:
            client_ctx.probe_state.pop("last_metrics", None)
            client_start = time.perf_counter()
            client_scored = self.scorer.score(generated_samples, client_ctx)
            scored_samples.extend(client_scored)
            client_selected = [
                item for item in select_top_k(client_scored, int(federation_cfg.get("top_k_bad", 10))) if item.client_id == client_ctx.client_id
            ]
            selected_bad_samples.extend(client_selected)
            paired_samples = self.retriever.retrieve(client_selected, client_ctx)
            retrieved_pairs.extend(paired_samples)
            client_critiques = self.critic.critique(paired_samples, client_ctx)
            critiques.extend(client_critiques)
            probe_metrics[client_ctx.client_id] = client_ctx.probe_state.get("last_metrics", {})
            client_latency_total += time.perf_counter() - client_start
        self.logger.info(
            "Round %d | client scoring complete | scored=%d selected_bad=%d retrieved_pairs=%d critiques=%d",
            display_round,
            len(scored_samples),
            len(selected_bad_samples),
            len(retrieved_pairs),
            len(critiques),
        )

        self.logger.info("Round %d | aggregation start", display_round)
        prompt_update = self.aggregator.aggregate(critiques, server_ctx)
        updated_prompt = server_ctx.prompt_text
        if prompt_update is not None:
            updated_prompt = apply_prompt_update(server_ctx.prompt_text, prompt_update)
            self.logger.info("Round %d | aggregation complete | prompt updated", display_round)
        else:
            self.logger.info("Round %d | aggregation complete | prompt unchanged", display_round)

        server_latency = time.perf_counter() - server_after_generation
        upload_tokens = sum(len(item.text.split()) for item in critiques)
        previous_generation_texts = server_ctx.generated_history[-1] if server_ctx.generated_history else None
        round_metrics: dict[str, Any] = {}
        round_metrics.update(
            compute_generation_metrics(
                generated_samples,
                prompt_text=server_ctx.prompt_text,
                previous_texts=previous_generation_texts,
            )
        )
        round_metrics.update(compute_critique_metrics(critiques))
        round_metrics.update(
            compute_system_metrics(
                client_latency_s=client_latency_total,
                server_latency_s=server_latency,
                upload_tokens=upload_tokens,
                prompt_text=updated_prompt,
                backend_names={
                    "embedder": getattr(client_contexts[0].embedder, "backend_name", type(client_contexts[0].embedder).__name__)
                    if client_contexts
                    else "none",
                    "client_llm": getattr(client_contexts[0].text_backend, "backend_name", "none") if client_contexts else "none",
                    "server_llm": getattr(server_ctx.text_backend, "backend_name", "none"),
                },
            )
        )
        round_metrics["selected_bad_avg_score"] = (
            sum(item.score for item in selected_bad_samples) / len(selected_bad_samples) if selected_bad_samples else 0.0
        )

        probe_entries = [metrics for metrics in probe_metrics.values() if metrics]
        if probe_entries:
            before = sum(float(metrics.get("val_loss_before", 0.0)) for metrics in probe_entries) / len(probe_entries)
            after = sum(float(metrics.get("val_loss_after", 0.0)) for metrics in probe_entries) / len(probe_entries)
            round_metrics["probe_val_loss_before_after"] = {"before": before, "after": after}

        if prompt_update is not None:
            round_metrics["critique_cluster_count"] = int(prompt_update.meta.get("cluster_count", 0))
            round_metrics["compression_ratio"] = float(prompt_update.meta.get("compression_ratio", 1.0))
        else:
            round_metrics["critique_cluster_count"] = 0
            round_metrics["compression_ratio"] = 1.0

        self.logger.info("Round %d | writing artifacts to %s", display_round, output_dir)
        write_text(output_dir / "server_prompt.txt", server_ctx.prompt_text)
        write_jsonl(output_dir / "generated_samples.jsonl", generated_samples)
        write_jsonl(output_dir / "scored_samples.jsonl", scored_samples)
        write_jsonl(output_dir / "selected_bad_samples.jsonl", selected_bad_samples)
        write_jsonl(output_dir / "retrieved_pairs.jsonl", retrieved_pairs)
        write_jsonl(output_dir / "client_critiques.jsonl", critiques)
        if round_ctx.runtime_artifacts.get("generation_requests"):
            write_jsonl(output_dir / "generation_requests.jsonl", round_ctx.runtime_artifacts["generation_requests"])
        if round_ctx.runtime_artifacts.get("generation_responses"):
            write_jsonl(output_dir / "generation_responses.jsonl", round_ctx.runtime_artifacts["generation_responses"])
        if prompt_update is not None:
            write_json(output_dir / "prompt_update.json", prompt_update)
            if prompt_update.meta.get("clusters"):
                write_json(output_dir / "aggregation_clusters.json", prompt_update.meta["clusters"])
        if probe_metrics:
            write_json(output_dir / "probe_metrics.json", probe_metrics)
        write_json(output_dir / "round_metrics.json", round_metrics)
        self.logger.info("Round %d | artifacts persisted", display_round)

        del server_start
        return RoundArtifacts(
            generated_samples=generated_samples,
            scored_samples=scored_samples,
            selected_bad_samples=selected_bad_samples,
            retrieved_pairs=retrieved_pairs,
            critiques=critiques,
            prompt_update=prompt_update,
            updated_prompt=updated_prompt,
            round_metrics=round_metrics,
        )
