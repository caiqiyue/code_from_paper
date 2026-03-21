from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_platform.algorithms.prototypes.minilm_mean import extract_minilm_mean_prototype
from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, write_json, write_jsonl, write_text
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.prompt_updater import apply_prompt_update, render_cluster_prompt
from thesis_platform.core.schemas import Critique, PromptUpdate, PrototypeFeedback, ScoredSample, Sample
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
    client_assigned_samples: list[Sample]
    prototype_feedbacks: list[PrototypeFeedback]
    cluster_prompts: dict[str, str]
    routing_summary: dict[str, Any]


class RoundRunner:
    """Execute the per-round server/client orchestration logic."""

    def __init__(self, *, generator: Any, scorer: Any, retriever: Any, critic: Any, aggregator: Any):
        self.generator = generator
        self.scorer = scorer
        self.retriever = retriever
        self.critic = critic
        self.aggregator = aggregator
        self.logger = get_logger()

    @staticmethod
    def _build_round_context(
        *,
        round_id: int,
        prompt_text: str,
        public_seed_samples: list[Sample],
        config: dict[str, Any],
        output_dir: Path,
        text_backend: Any,
        runtime_artifacts: dict[str, Any],
        prompt_scope: str,
        cluster_id: str | None,
        sample_id_prefix: str,
        sample_source: str,
    ) -> RoundContext:
        return RoundContext(
            round_id=round_id,
            prompt_text=prompt_text,
            public_seed_samples=public_seed_samples,
            config=config,
            output_dir=output_dir,
            text_backend=text_backend,
            prompt_scope=prompt_scope,
            cluster_id=cluster_id,
            sample_id_prefix=sample_id_prefix,
            sample_source=sample_source,
            runtime_artifacts=runtime_artifacts,
        )

    @staticmethod
    def _clone_assigned_sample(sample: Sample, *, client_id: str, assigned_index: int, source_pool: str) -> Sample:
        """Materialize one client-visible sample while preserving the synthetic source id."""

        meta = dict(sample.meta)
        meta.update(
            {
                "assigned_client_id": client_id,
                "assigned_index": assigned_index,
                "source_pool": source_pool,
            }
        )
        return Sample(
            sample_id=sample.sample_id,
            client_id=client_id,
            round_id=sample.round_id,
            source=sample.source,
            dataset_name=sample.dataset_name,
            task_type=sample.task_type,
            text=sample.text,
            instruction=sample.instruction,
            response=sample.response,
            label=sample.label,
            meta=meta,
        )

    @staticmethod
    def _take_rotated(pool: list[Sample], count: int, offset: int) -> list[Sample]:
        """Take a deterministic rotated slice from one pool."""

        if not pool or count <= 0:
            return []
        return [pool[(offset + index) % len(pool)] for index in range(count)]

    def _assign_client_samples(
        self,
        *,
        client_contexts: list[ClientContext],
        global_pool: list[Sample],
        cluster_pools: dict[str, list[Sample]],
        personalized_mix_ratio: float,
    ) -> dict[str, list[Sample]]:
        """Create one scoring pool per client from global and cluster-local candidates."""

        target_count = max(
            len(global_pool),
            max((len(pool) for pool in cluster_pools.values()), default=0),
            1,
        )
        personalized_mix_ratio = min(max(float(personalized_mix_ratio), 0.0), 1.0)
        assigned: dict[str, list[Sample]] = {}
        for client_index, client_ctx in enumerate(client_contexts):
            cluster_id = client_ctx.cluster_id or "global"
            cluster_pool = cluster_pools.get(cluster_id, [])
            local_count = int(round(target_count * personalized_mix_ratio)) if cluster_pool else 0
            global_count = max(target_count - local_count, 0)
            chosen = list(
                itertools.chain(
                    self._take_rotated(global_pool, global_count, client_index),
                    self._take_rotated(cluster_pool, local_count, client_index),
                )
            )
            if not chosen:
                chosen = self._take_rotated(global_pool, target_count, client_index)
            assigned[client_ctx.client_id] = [
                self._clone_assigned_sample(
                    sample,
                    client_id=client_ctx.client_id,
                    assigned_index=assigned_index,
                    source_pool=str(sample.meta.get("cluster_id") or sample.meta.get("prompt_scope", "global")),
                )
                for assigned_index, sample in enumerate(chosen)
            ]
        return assigned

    def _extract_client_prototypes(
        self,
        *,
        round_id: int,
        prototype_cfg: dict[str, Any],
        client_contexts: list[ClientContext],
        selected_bad_by_client: dict[str, list[ScoredSample]],
        retrieved_pairs: list[Any],
    ) -> list[PrototypeFeedback]:
        """Extract one prototype vector per client from retrieved real anchors."""

        prototype_name = str(prototype_cfg.get("name", "minilm_mean")).lower()
        if prototype_name != "minilm_mean":
            raise ValueError(f"Unsupported prototype extractor '{prototype_name}'.")

        feedbacks: list[PrototypeFeedback] = []
        pairs_by_client: dict[str, list[Any]] = defaultdict(list)
        for pair in retrieved_pairs:
            pairs_by_client[pair.client_id].append(pair)

        for client_ctx in client_contexts:
            selected_bad = selected_bad_by_client.get(client_ctx.client_id, [])
            pairs = pairs_by_client.get(client_ctx.client_id, [])
            real_samples: list[Sample] = []
            seen_ids: set[str] = set()
            for pair in pairs:
                for sample in pair.real_samples:
                    if sample.sample_id in seen_ids:
                        continue
                    seen_ids.add(sample.sample_id)
                    real_samples.append(sample)
            if not real_samples:
                real_samples = list(client_ctx.train_samples or client_ctx.all_samples)
            avg_bad_score = (
                sum(float(sample.score) for sample in selected_bad) / len(selected_bad) if selected_bad else 1.0
            )
            feedback = extract_minilm_mean_prototype(
                client_id=client_ctx.client_id,
                round_id=round_id,
                samples=real_samples,
                embedder=client_ctx.embedder,
                weight=avg_bad_score,
            )
            client_ctx.prototype_vector = list(feedback.prototype_vector)
            client_ctx.prototype_weight = float(feedback.weight)
            feedbacks.append(feedback)
        return feedbacks

    @staticmethod
    def _apply_prompt_update_to_contexts(
        *,
        current_prompt: str,
        prompt_update: PromptUpdate | None,
        server_ctx: ServerContext,
        client_contexts: list[ClientContext],
    ) -> dict[str, str]:
        """Update server and client cluster prompts after aggregation."""

        if prompt_update is None:
            return dict(server_ctx.cluster_prompts)

        server_ctx.client_cluster_map = dict(prompt_update.client_cluster_map)
        cluster_prompts = {
            cluster_id: render_cluster_prompt(current_prompt, prompt_update, cluster_id)
            for cluster_id in prompt_update.cluster_rules
        }
        server_ctx.cluster_prompts = cluster_prompts
        for client_ctx in client_contexts:
            client_ctx.cluster_id = prompt_update.client_cluster_map.get(client_ctx.client_id)
            if client_ctx.cluster_id is not None:
                client_ctx.cluster_prompt = cluster_prompts.get(client_ctx.cluster_id)
        return cluster_prompts

    def run_round(
        self,
        *,
        round_id: int,
        server_ctx: ServerContext,
        client_contexts: list[ClientContext],
        public_seed_samples: list[Sample],
        federation_cfg: dict[str, Any],
        output_dir: Path,
        prototype_cfg: dict[str, Any] | None = None,
        routing_cfg: dict[str, Any] | None = None,
    ) -> RoundArtifacts:
        """Run one full federation round and persist its artifacts to disk."""

        ensure_dir(output_dir)
        prototype_cfg = prototype_cfg or {}
        routing_cfg = routing_cfg or {}
        routing_enabled = bool(routing_cfg.get("enabled", False))
        personalized_mix_ratio = float(routing_cfg.get("personalized_mix_ratio", 0.7))
        server_ctx.routing_state["personalized_mix_ratio"] = personalized_mix_ratio

        display_round = round_id + 1
        runtime_artifacts: dict[str, Any] = {}

        server_start = time.perf_counter()
        self.logger.info("Round %d | generation start", display_round)
        global_round_ctx = self._build_round_context(
            round_id=round_id,
            prompt_text=server_ctx.prompt_text,
            public_seed_samples=public_seed_samples,
            config=federation_cfg,
            output_dir=output_dir,
            text_backend=server_ctx.text_backend,
            runtime_artifacts=runtime_artifacts,
            prompt_scope="global",
            cluster_id=None,
            sample_id_prefix="syn_global",
            sample_source="synthetic_global",
        )
        global_pool = self.generator.generate(global_round_ctx)
        generated_pools: dict[str, list[Sample]] = {"global": global_pool}
        generated_samples = list(global_pool)
        cluster_candidate_pools: dict[str, list[Sample]] = {}
        if routing_enabled and round_id > 0 and server_ctx.cluster_prompts:
            for cluster_id, cluster_prompt in sorted(server_ctx.cluster_prompts.items()):
                cluster_round_ctx = self._build_round_context(
                    round_id=round_id,
                    prompt_text=cluster_prompt,
                    public_seed_samples=public_seed_samples,
                    config=federation_cfg,
                    output_dir=output_dir,
                    text_backend=server_ctx.text_backend,
                    runtime_artifacts=runtime_artifacts,
                    prompt_scope="cluster",
                    cluster_id=cluster_id,
                    sample_id_prefix=f"syn_{cluster_id}",
                    sample_source="synthetic_cluster",
                )
                cluster_pool = self.generator.generate(cluster_round_ctx)
                cluster_candidate_pools[cluster_id] = cluster_pool
                generated_pools[cluster_id] = cluster_pool
                generated_samples.extend(cluster_pool)
        server_after_generation = time.perf_counter()
        self.logger.info("Round %d | generation complete | generated=%d", display_round, len(generated_samples))

        if routing_enabled:
            client_assigned_map = self._assign_client_samples(
                client_contexts=client_contexts,
                global_pool=global_pool,
                cluster_pools=cluster_candidate_pools,
                personalized_mix_ratio=personalized_mix_ratio,
            )
        else:
            client_assigned_map = {
                client_ctx.client_id: [
                    self._clone_assigned_sample(
                        sample,
                        client_id=client_ctx.client_id,
                        assigned_index=index,
                        source_pool="global",
                    )
                    for index, sample in enumerate(global_pool)
                ]
                for client_ctx in client_contexts
            }
        client_assigned_samples = [
            sample for client_id in sorted(client_assigned_map.keys()) for sample in client_assigned_map[client_id]
        ]

        scored_samples: list[ScoredSample] = []
        critiques: list[Critique] = []
        selected_bad_samples: list[ScoredSample] = []
        selected_bad_by_client: dict[str, list[ScoredSample]] = defaultdict(list)
        retrieved_pairs: list[Any] = []
        client_latency_total = 0.0
        probe_metrics: dict[str, Any] = {}

        self.logger.info("Round %d | client scoring start across %d clients", display_round, len(client_contexts))
        for client_ctx in client_contexts:
            client_ctx.probe_state.pop("last_metrics", None)
            client_start = time.perf_counter()
            client_samples = client_assigned_map.get(client_ctx.client_id, [])
            client_scored = self.scorer.score(client_samples, client_ctx)
            scored_samples.extend(client_scored)
            client_selected = [
                item
                for item in select_top_k(client_scored, int(federation_cfg.get("top_k_bad", 10)))
                if item.client_id == client_ctx.client_id
            ]
            selected_bad_samples.extend(client_selected)
            selected_bad_by_client[client_ctx.client_id].extend(client_selected)
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

        prototype_feedbacks: list[PrototypeFeedback] = []
        if routing_enabled:
            prototype_feedbacks = self._extract_client_prototypes(
                round_id=round_id,
                prototype_cfg=prototype_cfg,
                client_contexts=client_contexts,
                selected_bad_by_client=selected_bad_by_client,
                retrieved_pairs=retrieved_pairs,
            )
        server_ctx.prototype_feedbacks = prototype_feedbacks

        self.logger.info("Round %d | aggregation start", display_round)
        prompt_update = self.aggregator.aggregate(critiques, server_ctx)
        updated_prompt = server_ctx.prompt_text
        cluster_prompts = dict(server_ctx.cluster_prompts)
        if prompt_update is not None:
            updated_prompt = apply_prompt_update(server_ctx.prompt_text, prompt_update)
            cluster_prompts = self._apply_prompt_update_to_contexts(
                current_prompt=server_ctx.prompt_text,
                prompt_update=prompt_update,
                server_ctx=server_ctx,
                client_contexts=client_contexts,
            )
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
        round_metrics["assigned_generated_count"] = len(client_assigned_samples)
        round_metrics["prototype_count"] = len(prototype_feedbacks)
        round_metrics["personalized_mix_ratio"] = personalized_mix_ratio if routing_enabled else 0.0

        probe_entries = [metrics for metrics in probe_metrics.values() if metrics]
        if probe_entries:
            before = sum(float(metrics.get("val_loss_before", 0.0)) for metrics in probe_entries) / len(probe_entries)
            after = sum(float(metrics.get("val_loss_after", 0.0)) for metrics in probe_entries) / len(probe_entries)
            round_metrics["probe_val_loss_before_after"] = {"before": before, "after": after}

        if prompt_update is not None:
            round_metrics["critique_cluster_count"] = int(prompt_update.meta.get("cluster_count", 0))
            round_metrics["compression_ratio"] = float(prompt_update.meta.get("compression_ratio", 1.0))
            round_metrics["prototype_cluster_count"] = len(prompt_update.meta.get("prototype_clusters", []))
        else:
            round_metrics["critique_cluster_count"] = 0
            round_metrics["compression_ratio"] = 1.0
            round_metrics["prototype_cluster_count"] = 0

        routing_summary = {
            "enabled": routing_enabled,
            "personalized_mix_ratio": personalized_mix_ratio if routing_enabled else 0.0,
            "candidate_pool_counts": {key: len(value) for key, value in generated_pools.items()},
            "assigned_counts": {key: len(value) for key, value in client_assigned_map.items()},
            "prototype_count": len(prototype_feedbacks),
            "client_cluster_map": dict(server_ctx.client_cluster_map),
            "cluster_prompt_count": len(cluster_prompts),
        }

        self.logger.info("Round %d | writing artifacts to %s", display_round, output_dir)
        write_text(output_dir / "server_prompt.txt", server_ctx.prompt_text)
        write_jsonl(output_dir / "generated_samples.jsonl", generated_samples)
        write_jsonl(output_dir / "client_assigned_samples.jsonl", client_assigned_samples)
        write_jsonl(output_dir / "scored_samples.jsonl", scored_samples)
        write_jsonl(output_dir / "selected_bad_samples.jsonl", selected_bad_samples)
        write_jsonl(output_dir / "retrieved_pairs.jsonl", retrieved_pairs)
        write_jsonl(output_dir / "client_critiques.jsonl", critiques)
        if runtime_artifacts.get("generation_requests"):
            write_jsonl(output_dir / "generation_requests.jsonl", runtime_artifacts["generation_requests"])
        if runtime_artifacts.get("generation_responses"):
            write_jsonl(output_dir / "generation_responses.jsonl", runtime_artifacts["generation_responses"])
        if prototype_feedbacks:
            write_jsonl(output_dir / "client_prototypes.jsonl", prototype_feedbacks)
        if server_ctx.client_cluster_map:
            write_json(output_dir / "cluster_assignments.json", server_ctx.client_cluster_map)
        if cluster_prompts:
            write_json(output_dir / "cluster_prompts.json", cluster_prompts)
        write_json(output_dir / "routing_summary.json", routing_summary)
        if prompt_update is not None:
            write_json(output_dir / "prompt_update.json", prompt_update)
            if prompt_update.meta.get("clusters"):
                write_json(output_dir / "aggregation_clusters.json", prompt_update.meta["clusters"])
            if prompt_update.meta.get("prototype_clusters"):
                write_json(output_dir / "prototype_clusters.json", prompt_update.meta["prototype_clusters"])
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
            client_assigned_samples=client_assigned_samples,
            prototype_feedbacks=prototype_feedbacks,
            cluster_prompts=cluster_prompts,
            routing_summary=routing_summary,
        )
