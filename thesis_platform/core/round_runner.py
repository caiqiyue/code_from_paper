from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_platform.algorithms.prototypes.minilm_mean import extract_minilm_mean_prototype
from thesis_platform.core.artifact_manifest import ARTIFACT_SCHEMA_VERSION
from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_text
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.privacy import PrivacyLedger
from thesis_platform.core.prompt_updater import apply_prompt_update, render_cluster_prompt
from thesis_platform.core.schemas import Critique, PairedSample, PromptUpdate, PrototypeFeedback, ScoredSample, Sample
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
    privacy_summary: dict[str, Any]


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
        privatizer=None,
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
            if selected_bad:
                raw_influence_scores = [float(sample.meta.get("influence_score", sample.score)) for sample in selected_bad]
                min_inf = min(raw_influence_scores)
                max_inf = max(raw_influence_scores)
                if max_inf > min_inf:
                    r_k_utility = sum((x - min_inf) / (max_inf - min_inf) for x in raw_influence_scores) / len(raw_influence_scores)
                else:
                    r_k_utility = 0.5
            else:
                r_k_utility = 0.5
            feedback = extract_minilm_mean_prototype(
                client_id=client_ctx.client_id,
                round_id=round_id,
                samples=real_samples,
                embedder=client_ctx.embedder,
                weight=r_k_utility,
                privatizer=privatizer,
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

    @staticmethod
    def _stage_state_path(output_dir: Path) -> Path:
        return output_dir / "round_stage_state.json"

    @staticmethod
    def _round_privacy_path(output_dir: Path) -> Path:
        return output_dir / "round_privacy_ledger.json"

    @staticmethod
    def _sample_from_payload(payload: dict[str, Any]) -> Sample:
        return Sample(**payload)

    @classmethod
    def _scored_sample_from_payload(cls, payload: dict[str, Any]) -> ScoredSample:
        return ScoredSample(**payload)

    @classmethod
    def _paired_sample_from_payload(cls, payload: dict[str, Any]) -> PairedSample:
        return PairedSample(
            pair_id=str(payload.get("pair_id", "")),
            client_id=str(payload.get("client_id", "")),
            round_id=int(payload.get("round_id", 0)),
            bad_sample=cls._sample_from_payload(dict(payload.get("bad_sample", {}))),
            real_samples=[cls._sample_from_payload(dict(item)) for item in payload.get("real_samples", [])],
            meta=dict(payload.get("meta", {})),
        )

    @staticmethod
    def _critique_from_payload(payload: dict[str, Any]) -> Critique:
        return Critique(**payload)

    @staticmethod
    def _prototype_feedback_from_payload(payload: dict[str, Any]) -> PrototypeFeedback:
        return PrototypeFeedback(**payload)

    @staticmethod
    def _prompt_update_from_payload(payload: dict[str, Any]) -> PromptUpdate:
        return PromptUpdate(**payload)

    @staticmethod
    def _group_samples_by_client(samples: list[Sample]) -> dict[str, list[Sample]]:
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.client_id].append(sample)
        return dict(grouped)

    @staticmethod
    def _strip_round_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
        stripped = dict(payload)
        for key in ("schema_version", "artifact_type", "experiment_id", "round_id"):
            stripped.pop(key, None)
        return stripped

    def _load_stage_state(self, output_dir: Path) -> dict[str, Any]:
        path = self._stage_state_path(output_dir)
        if not path.exists():
            return {}
        return dict(read_json(path))

    def _write_stage_state(
        self,
        *,
        output_dir: Path,
        experiment_id: str,
        round_id: int,
        stage: str,
        status: str,
        completed_clients: list[str] | None = None,
        candidate_pool_counts: dict[str, int] | None = None,
        client_latency_total_s: float = 0.0,
        updated_prompt: str | None = None,
        routing_enabled: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "round_stage_state",
            "experiment_id": experiment_id,
            "round_id": round_id,
            "stage": stage,
            "status": status,
            "completed_clients": sorted(completed_clients or []),
            "candidate_pool_counts": dict(candidate_pool_counts or {}),
            "client_latency_total_s": round(float(client_latency_total_s), 6),
            "updated_prompt": updated_prompt,
            "routing_enabled": bool(routing_enabled),
            "updated_at": time.time(),
        }
        write_json(self._stage_state_path(output_dir), payload)
        return payload
    def _write_round_privacy_snapshot(self, output_dir: Path, privacy_ledger: PrivacyLedger | None) -> None:
        if privacy_ledger is None:
            return
        write_json(self._round_privacy_path(output_dir), privacy_ledger.report())

    def _restore_round_privacy_snapshot(self, output_dir: Path, privacy_ledger: PrivacyLedger | None) -> None:
        if privacy_ledger is None:
            return
        path = self._round_privacy_path(output_dir)
        if not path.exists():
            return
        report = dict(read_json(path))
        restored = PrivacyLedger.restore_from_report(report)
        privacy_ledger.entries = list(restored.entries)
        privacy_ledger.cumulative_spent = float(restored.cumulative_spent)
        if getattr(privacy_ledger, "_dp_privatizer", None) is not None:
            dp_state = report.get("dp_runtime_state")
            if dp_state:
                privacy_ledger._dp_privatizer.restore_state(dp_state)

    def _load_generation_artifacts(self, output_dir: Path) -> tuple[list[Sample], dict[str, list[Sample]], list[Sample]]:
        generated_samples = [self._sample_from_payload(dict(row)) for row in read_jsonl(output_dir / "generated_samples.jsonl")]
        client_assigned_samples = [self._sample_from_payload(dict(row)) for row in read_jsonl(output_dir / "client_assigned_samples.jsonl")]
        client_assigned_map = self._group_samples_by_client(client_assigned_samples)
        return generated_samples, client_assigned_map, client_assigned_samples

    def _load_client_analysis_artifacts(
        self,
        output_dir: Path,
    ) -> tuple[
        list[ScoredSample],
        list[ScoredSample],
        dict[str, list[ScoredSample]],
        list[PairedSample],
        list[Critique],
        list[PrototypeFeedback],
        dict[str, Any],
    ]:
        scored_samples = [self._scored_sample_from_payload(dict(row)) for row in read_jsonl(output_dir / "scored_samples.jsonl")]
        selected_bad_samples = [self._scored_sample_from_payload(dict(row)) for row in read_jsonl(output_dir / "selected_bad_samples.jsonl")]
        selected_bad_by_client: dict[str, list[ScoredSample]] = defaultdict(list)
        for sample in selected_bad_samples:
            selected_bad_by_client[sample.client_id].append(sample)
        retrieved_pairs = [self._paired_sample_from_payload(dict(row)) for row in read_jsonl(output_dir / "retrieved_pairs.jsonl")]
        critiques = [self._critique_from_payload(dict(row)) for row in read_jsonl(output_dir / "client_critiques.jsonl")]
        prototype_path = output_dir / "client_prototypes.jsonl"
        prototype_feedbacks = (
            [self._prototype_feedback_from_payload(dict(row)) for row in read_jsonl(prototype_path)]
            if prototype_path.exists()
            else []
        )
        probe_path = output_dir / "probe_metrics.json"
        probe_metrics = dict(read_json(probe_path)).get("clients", {}) if probe_path.exists() else {}
        return (
            scored_samples,
            selected_bad_samples,
            defaultdict(list, selected_bad_by_client),
            retrieved_pairs,
            critiques,
            prototype_feedbacks,
            probe_metrics,
        )

    def _load_completed_artifacts(
        self,
        *,
        output_dir: Path,
        server_ctx: ServerContext,
        client_contexts: list[ClientContext],
    ) -> RoundArtifacts:
        stage_state = self._load_stage_state(output_dir)
        generated_samples, _, client_assigned_samples = self._load_generation_artifacts(output_dir)
        (
            scored_samples,
            selected_bad_samples,
            _,
            retrieved_pairs,
            critiques,
            prototype_feedbacks,
            _,
        ) = self._load_client_analysis_artifacts(output_dir)

        prompt_update_path = output_dir / "prompt_update.json"
        prompt_update = (
            self._prompt_update_from_payload(dict(read_json(prompt_update_path)))
            if prompt_update_path.exists()
            else None
        )
        round_metrics = self._strip_round_metrics_payload(dict(read_json(output_dir / "round_metrics.json")))
        routing_summary = dict(read_json(output_dir / "routing_summary.json"))
        cluster_prompts_path = output_dir / "cluster_prompts.json"
        cluster_prompts = dict(read_json(cluster_prompts_path)) if cluster_prompts_path.exists() else {}
        cluster_assignments_path = output_dir / "cluster_assignments.json"
        if cluster_assignments_path.exists():
            server_ctx.client_cluster_map = dict(read_json(cluster_assignments_path))
        if prompt_update is not None:
            cluster_prompts = self._apply_prompt_update_to_contexts(
                current_prompt=server_ctx.prompt_text,
                prompt_update=prompt_update,
                server_ctx=server_ctx,
                client_contexts=client_contexts,
            )
        else:
            server_ctx.cluster_prompts = dict(cluster_prompts)
            for client_ctx in client_contexts:
                client_ctx.cluster_id = server_ctx.client_cluster_map.get(client_ctx.client_id)
                if client_ctx.cluster_id is not None:
                    client_ctx.cluster_prompt = cluster_prompts.get(client_ctx.cluster_id)
        server_ctx.prototype_feedbacks = prototype_feedbacks
        updated_prompt = str(stage_state.get("updated_prompt") or round_metrics.get("updated_prompt") or server_ctx.prompt_text)
        privacy_summary = dict(round_metrics.get("privacy", {}))
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
            privacy_summary=privacy_summary,
        )
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
        privacy_ledger: PrivacyLedger | None = None,
    ) -> RoundArtifacts:
        """Run one full federation round and persist its artifacts to disk."""

        ensure_dir(output_dir)
        prototype_cfg = prototype_cfg or {}
        routing_cfg = routing_cfg or {}
        routing_enabled = bool(routing_cfg.get("enabled", False))
        personalized_mix_ratio = float(routing_cfg.get("personalized_mix_ratio", 0.7))
        server_ctx.routing_state["personalized_mix_ratio"] = personalized_mix_ratio

        display_round = round_id + 1
        self._restore_round_privacy_snapshot(output_dir, privacy_ledger)
        stage_state = self._load_stage_state(output_dir)
        current_stage = str(stage_state.get("stage", "not_started"))
        candidate_pool_counts = {
            str(key): int(value) for key, value in dict(stage_state.get("candidate_pool_counts", {})).items()
        }
        client_latency_total = float(stage_state.get("client_latency_total_s", 0.0))

        if current_stage == "completed":
            self.logger.info("Round %d | resume from completed round artifacts in %s", display_round, output_dir)
            return self._load_completed_artifacts(
                output_dir=output_dir,
                server_ctx=server_ctx,
                client_contexts=client_contexts,
            )

        runtime_artifacts: dict[str, Any] = {}
        generation_stages = {"generation_completed", "client_analysis_in_progress", "client_analysis_completed"}
        if current_stage in generation_stages:
            self.logger.info("Round %d | resume from generation stage artifacts", display_round)
            generated_samples, client_assigned_map, client_assigned_samples = self._load_generation_artifacts(output_dir)
        else:
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
            candidate_pool_counts = {key: len(value) for key, value in generated_pools.items()}
            write_jsonl(output_dir / "generated_samples.jsonl", generated_samples)
            write_jsonl(output_dir / "client_assigned_samples.jsonl", client_assigned_samples)
            if runtime_artifacts.get("generation_requests"):
                write_jsonl(output_dir / "generation_requests.jsonl", runtime_artifacts["generation_requests"])
            if runtime_artifacts.get("generation_responses"):
                write_jsonl(output_dir / "generation_responses.jsonl", runtime_artifacts["generation_responses"])
            self._write_round_privacy_snapshot(output_dir, privacy_ledger)
            self._write_stage_state(
                output_dir=output_dir,
                experiment_id=server_ctx.experiment_id,
                round_id=round_id,
                stage="generation_completed",
                status="running",
                completed_clients=[],
                candidate_pool_counts=candidate_pool_counts,
                client_latency_total_s=0.0,
                routing_enabled=routing_enabled,
            )
        completed_clients = set(stage_state.get("completed_clients", [])) if current_stage == "client_analysis_in_progress" else set()

        if current_stage in {"client_analysis_in_progress", "client_analysis_completed"}:
            self.logger.info(
                "Round %d | resume from client analysis stage | completed_clients=%d",
                display_round,
                len(completed_clients),
            )
            (
                scored_samples,
                selected_bad_samples,
                selected_bad_by_client,
                retrieved_pairs,
                critiques,
                prototype_feedbacks,
                probe_metrics,
            ) = self._load_client_analysis_artifacts(output_dir)
            client_latency_total = float(stage_state.get("client_latency_total_s", 0.0))
        else:
            scored_samples = []
            critiques = []
            selected_bad_samples = []
            selected_bad_by_client = defaultdict(list)
            retrieved_pairs = []
            prototype_feedbacks = []
            probe_metrics = {}
            completed_clients = set()

        if current_stage not in {"client_analysis_completed"}:
            self.logger.info("Round %d | client scoring start across %d clients", display_round, len(client_contexts))
            for client_ctx in client_contexts:
                if client_ctx.client_id in completed_clients:
                    continue
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
                completed_clients.add(client_ctx.client_id)

                write_jsonl(output_dir / "scored_samples.jsonl", scored_samples)
                write_jsonl(output_dir / "selected_bad_samples.jsonl", selected_bad_samples)
                write_jsonl(output_dir / "retrieved_pairs.jsonl", retrieved_pairs)
                write_jsonl(output_dir / "client_critiques.jsonl", critiques)
                if probe_metrics:
                    write_json(
                        output_dir / "probe_metrics.json",
                        {
                            "schema_version": ARTIFACT_SCHEMA_VERSION,
                            "artifact_type": "probe_metrics",
                            "experiment_id": server_ctx.experiment_id,
                            "round_id": round_id,
                            "clients": probe_metrics,
                        },
                    )
                self._write_round_privacy_snapshot(output_dir, privacy_ledger)
                self._write_stage_state(
                    output_dir=output_dir,
                    experiment_id=server_ctx.experiment_id,
                    round_id=round_id,
                    stage="client_analysis_in_progress",
                    status="running",
                    completed_clients=sorted(completed_clients),
                    candidate_pool_counts=candidate_pool_counts,
                    client_latency_total_s=client_latency_total,
                    routing_enabled=routing_enabled,
                )

            self.logger.info(
                "Round %d | client scoring complete | scored=%d selected_bad=%d retrieved_pairs=%d critiques=%d",
                display_round,
                len(scored_samples),
                len(selected_bad_samples),
                len(retrieved_pairs),
                len(critiques),
            )

            if routing_enabled:
                prototype_feedbacks = self._extract_client_prototypes(
                    round_id=round_id,
                    prototype_cfg=prototype_cfg,
                    client_contexts=client_contexts,
                    selected_bad_by_client=selected_bad_by_client,
                    retrieved_pairs=retrieved_pairs,
                    privatizer=privacy_ledger,
                )
                write_jsonl(output_dir / "client_prototypes.jsonl", prototype_feedbacks)
            else:
                prototype_feedbacks = []
            self._write_round_privacy_snapshot(output_dir, privacy_ledger)
            self._write_stage_state(
                output_dir=output_dir,
                experiment_id=server_ctx.experiment_id,
                round_id=round_id,
                stage="client_analysis_completed",
                status="running",
                completed_clients=sorted(completed_clients),
                candidate_pool_counts=candidate_pool_counts,
                client_latency_total_s=client_latency_total,
                routing_enabled=routing_enabled,
            )

        server_ctx.prototype_feedbacks = prototype_feedbacks
        self.logger.info("Round %d | aggregation start", display_round)
        server_after_generation = time.perf_counter()
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
        privacy_summary = (
            privacy_ledger.record_round(
                round_id=round_id,
                sample_count=len(client_assigned_samples),
                critique_count=len(critiques),
                upload_token_count=upload_tokens,
            )
            if privacy_ledger is not None
            else {
                "round_id": round_id,
                "privacy_enabled": False,
                "privacy_mode": "disabled",
                "epsilon_budget": None,
                "delta_budget": None,
                "privacy_spent": 0.0,
                "privacy_spent_cumulative": 0.0,
                "privacy_budget_left": None,
                "privacy_budget_exceeded": False,
                "privacy_event_counts": {
                    "sample_count": len(client_assigned_samples),
                    "critique_count": len(critiques),
                    "upload_token_count": upload_tokens,
                },
                "privacy_spend_breakdown": {
                    "samples": 0.0,
                    "critiques": 0.0,
                    "upload_tokens": 0.0,
                },
            }
        )
        round_metrics.update(
            {
                "privacy_enabled": privacy_summary["privacy_enabled"],
                "privacy_mode": privacy_summary["privacy_mode"],
                "privacy_spent": privacy_summary["privacy_spent"],
                "privacy_spent_cumulative": privacy_summary["privacy_spent_cumulative"],
                "privacy_budget_left": privacy_summary["privacy_budget_left"],
                "privacy_budget_exceeded": privacy_summary["privacy_budget_exceeded"],
                "updated_prompt": updated_prompt,
            }
        )
        round_metrics["privacy"] = privacy_summary

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
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "routing_summary",
            "experiment_id": server_ctx.experiment_id,
            "round_id": round_id,
            "enabled": routing_enabled,
            "personalized_mix_ratio": personalized_mix_ratio if routing_enabled else 0.0,
            "candidate_pool_counts": candidate_pool_counts,
            "assigned_counts": {key: len(value) for key, value in client_assigned_map.items()},
            "prototype_count": len(prototype_feedbacks),
            "client_cluster_map": dict(server_ctx.client_cluster_map),
            "cluster_prompt_count": len(cluster_prompts),
            "dp_protected": any(fb.meta.get("dp_applied", False) for fb in prototype_feedbacks),
        }
        round_metrics_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "round_metrics",
            "experiment_id": server_ctx.experiment_id,
            "round_id": round_id,
            **round_metrics,
        }
        probe_metrics_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "probe_metrics",
            "experiment_id": server_ctx.experiment_id,
            "round_id": round_id,
            "clients": probe_metrics,
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
            write_json(output_dir / "probe_metrics.json", probe_metrics_payload)
        write_json(output_dir / "round_metrics.json", round_metrics_payload)
        self._write_round_privacy_snapshot(output_dir, privacy_ledger)
        self._write_stage_state(
            output_dir=output_dir,
            experiment_id=server_ctx.experiment_id,
            round_id=round_id,
            stage="completed",
            status="completed",
            completed_clients=sorted(completed_clients),
            candidate_pool_counts=candidate_pool_counts,
            client_latency_total_s=client_latency_total,
            updated_prompt=updated_prompt,
            routing_enabled=routing_enabled,
        )
        self.logger.info("Round %d | artifacts persisted", display_round)

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
            privacy_summary=privacy_summary,
        )

