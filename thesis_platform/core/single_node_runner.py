"""Single-node orchestration engine for the fine branch."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.schemas import Critique, PairedSample, PromptUpdate, Sample, ScoredSample
from thesis_platform.models.embedding import build_embedder


class SingleNodeRunner:
    """Orchestrate the single-node fine workflow: Stage A → Stage B → Stage C → Evaluation."""

    def __init__(
        self,
        *,
        generator: Any,
        scorer: Any,
        retriever: Any,
        critic: Any,
        aggregator: Any,
        config: Any,
    ):
        """Initialize the runner with all components and configuration."""

        self.generator = generator
        self.scorer = scorer
        self.retriever = retriever
        self.critic = critic
        self.aggregator = aggregator
        self.config = config
        self.logger = get_logger()

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full fine workflow and return a summary dict."""

        output_dir = self._get_output_dir()
        ensure_dir(output_dir)

        self.logger.info("=" * 60)
        self.logger.info("Starting single-node fine workflow")
        self.logger.info("Output directory: %s", output_dir)
        self.logger.info("=" * 60)

        # Stage A: Large-scale generation and scoring
        stage_a_start = time.perf_counter()
        scored_samples = self.run_stage_a(output_dir)
        stage_a_elapsed = time.perf_counter() - stage_a_start
        self.logger.info("Stage A completed in %.1f seconds | scored_samples=%d", stage_a_elapsed, len(scored_samples))

        # Stage B: Critique-retrieval-aggregation loop
        stage_b_start = time.perf_counter()
        stage_b_result = self.run_stage_b(output_dir, scored_samples)
        stage_b_elapsed = time.perf_counter() - stage_b_start
        self.logger.info("Stage B completed in %.1f seconds", stage_b_elapsed)

        # Stage C: Bootstrap expansion
        stage_c_start = time.perf_counter()
        synthetic_texts = self.run_stage_c(output_dir, stage_b_result)
        stage_c_elapsed = time.perf_counter() - stage_c_start
        self.logger.info("Stage C completed in %.1f seconds | synthetic_samples=%d", stage_c_elapsed, len(synthetic_texts))

        # Evaluation
        eval_result = self.run_evaluation(output_dir, synthetic_texts)

        # Write final summary
        summary = {
            "experiment_id": self.config.meta.get("experiment_id"),
            "stage_a": {
                "scored_samples": len(scored_samples),
                "elapsed_seconds": stage_a_elapsed,
            },
            "stage_b": {
                "num_iterations": len(stage_b_result.get("history", [])),
                "final_prompt_length": len(stage_b_result.get("prompt", "")),
                "elapsed_seconds": stage_b_elapsed,
            },
            "stage_c": {
                "synthetic_samples": len(synthetic_texts),
                "elapsed_seconds": stage_c_elapsed,
            },
            "evaluation": eval_result,
        }
        write_json(output_dir / "metrics_summary.json", summary)
        self.logger.info("Metrics summary written to %s", output_dir / "metrics_summary.json")

        # Final GPU memory cleanup
        self.logger.info("Cleaning up GPU memory")
        from thesis_platform.core.resource_cleanup import release_component_resources
        release_component_resources()

        return summary

    # -------------------------------------------------------------------------
    # Stage A: Large-scale generation and scoring
    # -------------------------------------------------------------------------

    def run_stage_a(self, output_dir: Path) -> list[ScoredSample]:
        """Generate synthetic samples and score them using DataInf.

        Args:
            output_dir: Directory to write stage artifacts

        Returns:
            List of ScoredSample objects sorted by influence score (ascending)
        """

        stage_a_dir = ensure_dir(output_dir / "stage_a")
        self.logger.info("Stage A: Large-scale generation and scoring")

        # Check for cached results
        cached_path = stage_a_dir / "scored_samples.jsonl"
        if cached_path.exists():
            self.logger.info("Stage A: Loading cached scored samples from %s", cached_path)
            return [ScoredSample(**row) for row in read_jsonl(cached_path)]

        # Load seed corpus
        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        seed_corpus = self._load_seed_corpus(train_path)
        self.logger.info("Stage A: Loaded %d seed samples from %s", len(seed_corpus), train_path)

        # Get configuration
        generated_count = int(self.config.stage_a.get("generated_count", 10000))
        batch_size = int(self.config.stage_a.get("batch_size", 1000))
        select_top_k = int(self.config.stage_a.get("select_top_k", 8000))

        # Stage A generation (using PretextPromptLLMGenerator)
        all_generated = self._generate_batched(seed_corpus, generated_count)
        self.logger.info("Stage A: Generated %d samples", len(all_generated))

        # Build single-node client context (limit train samples for probe to avoid OOM)
        max_probe_samples = int(self.config.stage_a.get("max_probe_samples", 10000))
        train_for_probe = seed_corpus[:max_probe_samples]
        self.logger.info("Stage A: Using %d train samples for DataInf probe (of %d total)", len(train_for_probe), len(seed_corpus))
        client_ctx = self._build_client_context(train_for_probe, all_generated)

        # Score all generated samples
        scored_samples = self._score_batched(all_generated, client_ctx, batch_size)
        self.logger.info("Stage A: Scored %d samples", len(scored_samples))

        # Sort by score (descending: worst/largest first) and select top-k
        # DataInf: score_direction="larger_is_worse", so larger score = worse sample
        scored_samples.sort(key=lambda x: x.score, reverse=True)
        selected_samples = scored_samples[:select_top_k]
        self.logger.info("Stage A: Selected top %d worst samples for Stage B", len(selected_samples))

        # Save stage artifacts
        write_jsonl(stage_a_dir / "scored_samples.jsonl", selected_samples)
        write_json(stage_a_dir / "stage_config.json", {
            "generated_count": generated_count,
            "batch_size": batch_size,
            "select_top_k": select_top_k,
            "total_scored": len(scored_samples),
        })

        # Release Stage A resources (Generator + Scorer + ClientContext)
        # Note: client_ctx is only needed for scoring in Stage A; its embedder (MiniLM)
        # and text_backend (Qwen client) are no longer needed after scoring completes.
        self.logger.info("Stage A: Releasing Generator, Scorer, and ClientContext resources")
        from thesis_platform.core.resource_cleanup import release_component_resources
        release_component_resources(self.generator, self.scorer, client_ctx)
        self.generator = None
        self.scorer = None

        return selected_samples

    # -------------------------------------------------------------------------
    # Stage B: Critique-retrieval-aggregation loop
    # -------------------------------------------------------------------------

    def run_stage_b(self, output_dir: Path, scored_samples: list[ScoredSample]) -> dict[str, Any]:
        """Execute the critique-retrieval-aggregation loop.

        Args:
            output_dir: Directory to write stage artifacts
            scored_samples: Samples from Stage A to process

        Returns:
            Dict with 'prompt', 'history', and 'memory' keys
        """

        stage_b_dir = ensure_dir(output_dir / "stage_b")
        self.logger.info("Stage B: Critique-retrieval-aggregation loop")

        # Check for cached results
        cached_path = stage_b_dir / "prompt_update.json"
        if cached_path.exists():
            self.logger.info("Stage B: Loading cached prompt update from %s", cached_path)
            data = read_json(cached_path)
            return {
                "prompt": data.get("final_prompt", ""),
                "history": data.get("prompt_history", []),
                "memory": data.get("aggregation_memory", {}),
            }

        # Get configuration
        num_iterations = int(self.config.stage_b.get("num_iterations", 5))
        batch_size = int(self.config.stage_b.get("batch_size", 500))
        selection_mode = str(self.config.stage_b.get("selection_mode", "worst"))

        # Select samples based on mode
        # scored_samples from Stage A is sorted descending (worst/largest first)
        # For "worst": use as-is (worst-first)
        # For "best": reverse to get best-first (ascending)
        # For "all": use as-is
        if selection_mode == "worst":
            b_samples = list(scored_samples)
        elif selection_mode == "best":
            b_samples = list(reversed(scored_samples))
        else:  # "all"
            b_samples = list(scored_samples)

        # Build contexts
        client_ctx = self._build_client_context_for_stage_b(scored_samples)
        server_ctx = self._build_server_context()

        # Iterative critique-retrieval-aggregation
        all_critiques: list[Critique] = []
        for iteration in range(num_iterations):
            self.logger.info("Stage B iteration %d/%d", iteration + 1, num_iterations)

            # Select batch for this iteration with cycling
            # Use modulo to cycle through samples when batch_size > len(b_samples)
            start_idx = (iteration * batch_size) % len(b_samples)
            end_idx = start_idx + batch_size

            if end_idx <= len(b_samples):
                # No cycling needed - batch is contiguous
                batch = b_samples[start_idx:end_idx]
            else:
                # Cycling needed - wrap around to beginning
                batch = b_samples[start_idx:]
                remaining_needed = batch_size - len(batch)
                batch = batch + b_samples[:remaining_needed]

            # Retrieval
            paired_samples = self._retrieve_batched(batch, client_ctx)

            # Critique
            critiques = self._critique_batched(paired_samples, client_ctx)
            all_critiques.extend(critiques)

            # Aggregation (use aggregate_dbscan_critiques directly)
            from thesis_platform.algorithms.aggregators.dbscan_core import aggregate_dbscan_critiques

            prompt_update, new_memory = aggregate_dbscan_critiques(
                critiques=all_critiques,
                round_id=iteration,
                max_rules=int(self.config.aggregator.get("max_rules", 5)),
                embedder=client_ctx.embedder,
                text_backend=client_ctx.text_backend,
                eps=float(self.config.aggregator.get("cluster_eps", 0.35)),
                min_samples=int(self.config.aggregator.get("cluster_min_samples", 2)),
                use_memory=True,
                memory=server_ctx.aggregation_memory,
                momentum_beta=float(self.config.aggregator.get("momentum_beta", 0.7)),
                base_prompt=server_ctx.base_prompt,
                prototype_feedbacks=list(server_ctx.prototype_feedbacks),
                prototype_cluster_method="dbscan",
            )

            # Update prompt
            if prompt_update and prompt_update.rules:
                server_ctx.prompt_text = self._apply_prompt_update(server_ctx.prompt_text, prompt_update)
                server_ctx.prompt_history.append(server_ctx.prompt_text)

            server_ctx.aggregation_memory = new_memory

            # Save iteration artifacts
            iter_dir = ensure_dir(stage_b_dir / f"iteration_{iteration}")
            write_jsonl(iter_dir / "critiques.jsonl", critiques)
            if prompt_update:
                write_json(iter_dir / "prompt_update.json", prompt_update)

        # Save final Stage B artifacts
        final_result = {
            "final_prompt": server_ctx.prompt_text,
            "prompt_history": server_ctx.prompt_history,
            "num_iterations": num_iterations,
            "total_critiques": len(all_critiques),
            "aggregation_memory": server_ctx.aggregation_memory,
        }
        write_json(stage_b_dir / "prompt_update.json", final_result)
        write_jsonl(stage_b_dir / "critiques.jsonl", all_critiques)

        # Release Stage B resources (Retriever + Critic + Aggregator + ClientContext)
        # Note: client_ctx holds the embedder (MiniLM) and text_backend (Qwen client)
        # which are no longer needed after Stage B completes.
        self.logger.info("Stage B: Releasing Retriever, Critic, Aggregator, and ClientContext resources")
        from thesis_platform.core.resource_cleanup import release_component_resources
        release_component_resources(self.retriever, self.critic, self.aggregator, client_ctx, server_ctx)
        self.retriever = None
        self.critic = None
        self.aggregator = None

        return {
            "prompt": server_ctx.prompt_text,
            "history": server_ctx.prompt_history,
            "memory": server_ctx.aggregation_memory,
        }

    # -------------------------------------------------------------------------
    # Stage C: Bootstrap expansion
    # -------------------------------------------------------------------------

    def run_stage_c(self, output_dir: Path, stage_b_result: dict[str, Any]) -> list[str]:
        """Execute bootstrap expansion to generate final synthetic corpus.

        Args:
            output_dir: Directory to write stage artifacts
            stage_b_result: Output from Stage B containing the aggregated prompt.
                           Currently reserved for future rule-guided bootstrap generation.

        Returns:
            List of synthetic text strings
        """

        stage_c_dir = ensure_dir(output_dir / "stage_c")
        self.logger.info("Stage C: Bootstrap expansion")

        # Check for cached results
        cached_path = stage_c_dir / "llama7b_text_syn.json"
        if cached_path.exists():
            self.logger.info("Stage C: Loading cached synthetic texts from %s", cached_path)
            with open(cached_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Get configuration
        num_prompts = int(self.config.stage_c.get("num_prompts", 10000))
        generator_backend = str(self.config.stage_c.get("generator_backend", "huggingface"))
        generator_model = str(self.config.stage_c.get("generator_model", "distilgpt2"))
        max_tokens = int(self.config.stage_c.get("max_tokens", 64))
        temperature = float(self.config.stage_c.get("temperature", 1.0))
        top_p = float(self.config.stage_c.get("top_p", 1.0))

        # Load seed texts for bootstrap prompts
        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        seed_texts = self._load_seed_texts(train_path, limit=min(num_prompts, 10000))

        # Build bootstrap prompts (reuse PrE-Text's build_bootstrap_prompts)
        try:
            # Add PrE-Text to sys.path before importing
            from thesis_platform.evaluation.downstream_eval import _ensure_pretext_import
            repo_root = self.config.repo_root()
            _ensure_pretext_import(repo_root)

            from pretext_platform.algorithms.bootstrap import build_bootstrap_prompts
            prompts = build_bootstrap_prompts(
                seed_texts=seed_texts,
                num_prompts=num_prompts,
                seed=int(self.config.meta.get("seed", 42)),
            )
        except ImportError as e:
            self.logger.warning("PrE-Text bootstrap not available (%s), using simple prompts", e)
            prompts = self._build_simple_prompts(seed_texts, num_prompts, stage_b_result)

        # Generate with configured backend
        synthetic_texts = self._generate_bootstrap(prompts, generator_backend, generator_model, max_tokens, temperature, top_p)

        # Light filtering (keep ~95%)
        filtered = [t for t in synthetic_texts if self._is_valid_sample(t)]
        self.logger.info("Stage C: Generated %d, kept %d after filtering", len(synthetic_texts), len(filtered))

        # Save artifacts
        with open(cached_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        write_json(stage_c_dir / "stage_config.json", {
            "num_prompts": num_prompts,
            "generator_backend": generator_backend,
            "generator_model": generator_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "total_generated": len(synthetic_texts),
            "total_filtered": len(filtered),
        })

        return filtered

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def run_evaluation(self, output_dir: Path, synthetic_texts: list[str]) -> dict[str, Any]:
        """Run downstream GPT-2 evaluation on the synthetic corpus.

        Args:
            output_dir: Directory to write evaluation artifacts
            synthetic_texts: List of synthetic text strings

        Returns:
            Evaluation results dict
        """

        eval_dir = ensure_dir(output_dir / "eval")
        self.logger.info("Running downstream evaluation on %d synthetic samples", len(synthetic_texts))

        # Check if small evaluation is enabled (via downstream_eval section)
        if not bool(self.config.downstream_eval.get("run_small_eval", False)):
            self.logger.info("Small evaluation disabled, skipping")
            return {"status": "skipped"}

        try:
            # Ensure pretext_platform is importable (adds ../PrE-Text to sys.path)
            from thesis_platform.evaluation.downstream_eval import _ensure_pretext_import
            _ensure_pretext_import(self.config.repo_root())

            from thesis_platform.evaluation.downstream_eval import export_synthetic_corpus, run_pretext_small_eval

            # Export synthetic corpus to the format expected by PrE-Text evaluation
            stage2_dir = ensure_dir(eval_dir / "stage2")
            export_synthetic_corpus(
                synthetic_texts,
                output_dir=stage2_dir,
                filename="llama7b_text_syn.json",
            )

            # Run PrE-Text small model evaluation (GPT-2 or DistilGPT-2)
            eval_result = run_pretext_small_eval(
                self.config,
                stage2_dir=stage2_dir,
                output_dir=eval_dir,
            )
            write_json(eval_dir / "eval_small_summary.json", eval_result)
            return eval_result
        except Exception as e:
            self.logger.error("Evaluation failed: %s", e)
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _get_output_dir(self) -> Path:
        """Resolve the output directory for this experiment."""
        experiment_id = self.config.meta.get("experiment_id", "single_node_fine")
        output_root = self.config.output_root()
        return ensure_dir(output_root / experiment_id)

    def _load_seed_corpus(self, train_path: Path) -> list[Sample]:
        """Load training samples as Sample objects."""
        if not train_path.exists():
            raise FileNotFoundError(f"Train path not found: {train_path}")

        data = read_json(train_path)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("samples", data.get("data", [data]))
        else:
            items = [data]

        samples = []
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                samples.append(Sample(
                    sample_id=f"seed_{idx}",
                    client_id="single_node",
                    round_id=0,
                    source="seed",
                    dataset_name=self.config.data.get("dataset_name", "unknown"),
                    task_type=self.config.data.get("task_type", "instruction_tuning"),
                    text=item.get("text", item.get("content", str(item))),
                    instruction=item.get("instruction"),
                    response=item.get("response"),
                    label=item.get("label"),
                    meta=item,
                ))
            else:
                samples.append(Sample(
                    sample_id=f"seed_{idx}",
                    client_id="single_node",
                    round_id=0,
                    source="seed",
                    dataset_name=self.config.data.get("dataset_name", "unknown"),
                    task_type=self.config.data.get("task_type", "instruction_tuning"),
                    text=str(item),
                    meta={},
                ))
        return samples

    def _load_seed_texts(self, train_path: Path, limit: int = 10000) -> list[str]:
        """Load seed texts as plain strings for bootstrap prompts."""
        samples = self._load_seed_corpus(train_path)
        texts = [s.render_text() for s in samples[:limit]]
        return texts

    def _build_client_context(self, train_samples: list[Sample], all_samples: list[Sample]) -> ClientContext:
        """Build a single-node ClientContext with embedder and text backend."""
        embedder_cfg = self.config.retriever or {}
        embedding_model = embedder_cfg.get("embedding_model", "models/all-MiniLM-L6-v2")
        repo_root = self.config.repo_root()
        embedder = build_embedder(
            embedding_model,
            repo_root,
            allow_fallback=bool(embedder_cfg.get("allow_hashing_fallback", True)),
        )

        # Build client text backend (used for critique in Stage B)
        text_backend = self._build_text_backend(role="client")

        client_ctx = ClientContext(
            client_id="single_node_client",
            train_samples=train_samples,
            validation_samples=[],
            all_samples=all_samples,
            embedder=embedder,
            config=dict(self.config.raw),
            text_backend=text_backend,
        )
        return client_ctx

    def _build_client_context_for_stage_b(self, scored_samples: list[ScoredSample]) -> ClientContext:
        """Build ClientContext for Stage B processing."""
        # Convert ScoredSample back to Sample for the corpus
        train_samples = [Sample(
            sample_id=s.sample_id,
            client_id=s.client_id,
            round_id=s.round_id,
            source=s.source,
            dataset_name=s.dataset_name,
            task_type=s.task_type,
            text=s.text,
            instruction=s.instruction,
            response=s.response,
            label=s.label,
            meta=dict(s.meta),
        ) for s in scored_samples]

        all_samples = list(train_samples)
        return self._build_client_context(train_samples, all_samples)

    def _build_server_context(self) -> ServerContext:
        """Build a single-node ServerContext.

        Note: In single-node Stage B, server_ctx.text_backend is NOT used -
        all LLM calls go through client_ctx.text_backend instead.
        We pass None to avoid loading the server Qwen model unnecessarily.
        """
        generator_cfg = self.config.generator or {}
        initial_prompt = generator_cfg.get("initial_prompt", "Generate text that matches the target dataset style.")

        return ServerContext(
            experiment_id=str(self.config.meta.get("experiment_id", "single_node")),
            prompt_text=initial_prompt,
            prompt_history=[],
            config=dict(self.config.raw),
            output_dir=self._get_output_dir(),
            text_backend=None,  # Not used in single-node Stage B - client_ctx.text_backend is used instead
            aggregation_memory={"entries": []},
            generated_history=[],
            base_prompt=initial_prompt,
        )

    def _build_text_backend(self, role: str = "server"):
        """Build the LLM text backend from config.

        Args:
            role: Either "client" or "server" to select the appropriate backend config.

        Returns:
            A text backend instance or None if building fails.
        """
        try:
            from thesis_platform.models.backends import build_text_backend
            llm_cfg = dict(self.config.llm or {})
            role_cfg = dict(llm_cfg.get(role, {}))
            if not role_cfg:
                self.logger.warning("No %s LLM config found, text backend will be None", role)
                return None
            repo_root = self.config.repo_root()
            return build_text_backend({**role_cfg, "role": role}, repo_root=repo_root)
        except Exception as e:
            self.logger.warning("Could not build %s text backend: %s", role, e)
            return None

    def _generate_batched(self, seed_corpus: list[Sample], total_count: int) -> list[Sample]:
        """Generate synthetic samples by repeatedly calling the generator.

        The PretextPromptLLMGenerator.generate() produces `generated_per_round` samples
        per call. We loop until we have at least `total_count` samples.
        """
        all_generated = []
        generated_per_round = getattr(self.generator, 'generated_per_round', 16)
        num_full_rounds = total_count // generated_per_round
        remainder = total_count % generated_per_round

        round_id = 0
        sample_offset = 0

        # Pre-create text_backend once to avoid reloading model every round
        text_backend = self._build_text_backend()

        # Full rounds
        for round_idx in range(num_full_rounds):
            # Rotate seed samples for this round to increase diversity
            start_idx = (round_idx * generated_per_round) % len(seed_corpus)
            seed_subset = [
                seed_corpus[(start_idx + i) % len(seed_corpus)]
                for i in range(self.generator.exemplars_per_prompt)
            ]

            round_ctx = RoundContext(
                round_id=round_id,
                prompt_text=self.config.generator.get("initial_prompt", "Generate text that matches the target dataset style."),
                public_seed_samples=seed_subset,
                config=dict(self.config.raw),
                output_dir=self._get_output_dir(),
                text_backend=text_backend,
                sample_id_prefix=f"syn_single_r{round_id}",
                sample_source="synthetic_single_node",
            )

            batch_samples = self.generator.generate(round_ctx)
            all_generated.extend(batch_samples)
            round_id += 1
            sample_offset += len(batch_samples)

            if (round_idx + 1) % 50 == 0:
                self.logger.info("Stage A generation progress: %d / %d samples", len(all_generated), total_count)

        # Partial round if needed
        if remainder > 0 and remainder != generated_per_round:
            start_idx = (num_full_rounds * generated_per_round) % len(seed_corpus)
            seed_subset = [
                seed_corpus[(start_idx + i) % len(seed_corpus)]
                for i in range(self.generator.exemplars_per_prompt)
            ]

            round_ctx = RoundContext(
                round_id=round_id,
                prompt_text=self.config.generator.get("initial_prompt", "Generate text that matches the target dataset style."),
                public_seed_samples=seed_subset,
                config=dict(self.config.raw),
                output_dir=self._get_output_dir(),
                text_backend=text_backend,
                sample_id_prefix=f"syn_single_r{round_id}",
                sample_source="synthetic_single_node",
            )

            batch_samples = self.generator.generate(round_ctx)
            # Take only the remainder
            all_generated.extend(batch_samples[:remainder])

        self.logger.info("Stage A: Generated %d samples (target was %d)", len(all_generated), total_count)

        # Release text_backend used during generation
        if text_backend is not None:
            from thesis_platform.core.resource_cleanup import release_component_resources
            release_component_resources(text_backend)

        return all_generated

    def _score_batched(self, samples: list[Sample], client_ctx: ClientContext, batch_size: int) -> list[ScoredSample]:
        """Score samples in batches using the configured scorer."""
        all_scored = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            scored = self.scorer.score(batch, client_ctx)
            all_scored.extend(scored)
            self.logger.info("Stage A scoring batch %d-%d / %d: %d samples",
                           i, min(i + batch_size, len(samples)), len(samples), len(scored))
        return all_scored

    def _retrieve_batched(self, samples: list[Sample], client_ctx: ClientContext) -> list[PairedSample]:
        """Retrieve anchor samples for a batch."""
        return self.retriever.retrieve(samples, client_ctx)

    def _critique_batched(self, paired_samples: list[PairedSample], client_ctx: ClientContext) -> list[Critique]:
        """Generate critiques for a batch of paired samples."""
        return self.critic.critique(paired_samples, client_ctx)

    def _generate_bootstrap(
        self,
        prompts: list[str],
        backend: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate synthetic texts from bootstrap prompts."""
        if backend == "local":
            return self._generate_local(prompts, model, max_tokens, temperature, top_p)
        else:
            self.logger.warning("Unsupported backend %s, using local model", backend)
            return self._generate_local(prompts, model, max_tokens, temperature, top_p)

    def _generate_local(
        self,
        prompts: list[str],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate using a local HuggingFace model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            repo_root = self.config.repo_root()
            model_path = str(repo_root / model)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_instance = AutoModelForCausalLM.from_pretrained(model_path).to(device)

            outputs = []
            batch_size = 8
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model_instance.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                outputs.extend(batch_outputs)

            # Release bootstrap model resources
            del model_instance
            del tokenizer
            torch.cuda.empty_cache()

            return outputs
        except Exception as e:
            self.logger.error("Local model generation failed: %s", e)
            return []

    def _build_simple_prompts(
        self,
        seed_texts: list[str],
        num_prompts: int,
        stage_b_result: dict[str, Any],
    ) -> list[str]:
        """Build simple few-shot prompts when PrE-Text bootstrap is unavailable.

        Args:
            seed_texts: Seed texts to use as examples
            num_prompts: Number of prompts to generate
            stage_b_result: Output from Stage B containing aggregated prompt/rules
        """
        import random
        rng = random.Random(int(self.config.meta.get("seed", 42)))

        # Optionally incorporate Stage B aggregated prompt for guided generation
        guidance = stage_b_result.get("prompt", "") if stage_b_result else ""

        prompts = []
        for _ in range(num_prompts):
            examples = rng.sample(seed_texts, min(3, len(seed_texts)))
            example_text = "\n".join(examples[:3])
            if guidance:
                prompt = (
                    f"List of 3 diverse original text samples:\n{example_text}\n\n"
                    f"Guidance: {guidance[:200]}"
                )
            else:
                prompt = f"List of 3 diverse original text samples:\n{example_text}"
            prompts.append(prompt)
        return prompts

    def _is_valid_sample(self, text: str) -> bool:
        """Light validation to filter out invalid samples."""
        if not text or not isinstance(text, str):
            return False
        text = text.strip()
        if len(text) < 5:
            return False
        # Filter out obvious errors
        if text.count("<") > 5 or text.count(">") > 5:
            return False
        return True

    @staticmethod
    def _apply_prompt_update(current_prompt: str, prompt_update: PromptUpdate) -> str:
        """Apply a PromptUpdate to the current prompt."""
        if not prompt_update or not prompt_update.rules:
            return current_prompt

        rules_text = "\n".join(f"- {rule}" for rule in prompt_update.rules[:5])
        return f"{current_prompt}\n\nGenerated guidance:\n{rules_text}"
