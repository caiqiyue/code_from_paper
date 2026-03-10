from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, write_json, write_jsonl, write_text
from thesis_platform.core.prompt_updater import apply_prompt_update
from thesis_platform.core.schemas import Critique, PromptUpdate, ScoredSample, Sample
from thesis_platform.core.selector import select_top_k
from thesis_platform.evaluation.metrics import compute_critique_metrics, compute_generation_metrics, compute_system_metrics


@dataclass(slots=True)
class RoundArtifacts:
    generated_samples: list[Sample]
    scored_samples: list[ScoredSample]
    selected_bad_samples: list[ScoredSample]
    retrieved_pairs: list[Any]
    critiques: list[Critique]
    prompt_update: PromptUpdate | None
    updated_prompt: str
    round_metrics: dict[str, Any]


class RoundRunner:
    def __init__(self, *, generator: Any, scorer: Any, retriever: Any, critic: Any, aggregator: Any):
        self.generator = generator
        self.scorer = scorer
        self.retriever = retriever
        self.critic = critic
        self.aggregator = aggregator

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
        ensure_dir(output_dir)
        round_ctx = RoundContext(
            round_id=round_id,
            prompt_text=server_ctx.prompt_text,
            public_seed_samples=public_seed_samples,
            config=federation_cfg,
            output_dir=output_dir,
        )

        server_start = time.perf_counter()
        generated_samples = self.generator.generate(round_ctx)
        server_after_generation = time.perf_counter()

        scored_samples: list[ScoredSample] = []
        critiques: list[Critique] = []
        selected_bad_samples: list[ScoredSample] = []
        retrieved_pairs: list[Any] = []
        client_latency_total = 0.0

        for client_ctx in client_contexts:
            client_start = time.perf_counter()
            client_scored = self.scorer.score(generated_samples, client_ctx)
            scored_samples.extend(client_scored)
            client_selected = [
                item for item in select_top_k(client_scored, int(federation_cfg.get("top_k_bad", 10)))
                if item.client_id == client_ctx.client_id
            ]
            selected_bad_samples.extend(client_selected)
            paired_samples = self.retriever.retrieve(client_selected, client_ctx)
            retrieved_pairs.extend(paired_samples)
            client_critiques = self.critic.critique(paired_samples, client_ctx)
            critiques.extend(client_critiques)
            client_latency_total += time.perf_counter() - client_start

        prompt_update = self.aggregator.aggregate(critiques, server_ctx)
        updated_prompt = server_ctx.prompt_text
        if prompt_update is not None:
            updated_prompt = apply_prompt_update(server_ctx.prompt_text, prompt_update)

        server_latency = time.perf_counter() - server_after_generation
        upload_tokens = sum(len(item.text.split()) for item in critiques)
        round_metrics = {}
        round_metrics.update(compute_generation_metrics(generated_samples))
        round_metrics.update(compute_critique_metrics(critiques))
        round_metrics.update(
            compute_system_metrics(
                client_latency_s=client_latency_total,
                server_latency_s=server_latency,
                upload_tokens=upload_tokens,
                prompt_text=updated_prompt,
            )
        )

        write_text(output_dir / "server_prompt.txt", server_ctx.prompt_text)
        write_jsonl(output_dir / "generated_samples.jsonl", generated_samples)
        write_jsonl(output_dir / "scored_samples.jsonl", scored_samples)
        write_jsonl(output_dir / "selected_bad_samples.jsonl", selected_bad_samples)
        write_jsonl(output_dir / "retrieved_pairs.jsonl", retrieved_pairs)
        write_jsonl(output_dir / "client_critiques.jsonl", critiques)
        if prompt_update is not None:
            write_json(output_dir / "prompt_update.json", prompt_update)
        write_json(output_dir / "round_metrics.json", round_metrics)

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
