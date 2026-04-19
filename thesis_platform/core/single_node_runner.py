"""Single-node orchestration engine for the fine branch."""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, read_json, write_json, write_jsonl
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.registry import create
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

        # Stage A: Prompt optimization through iterative critique and aggregation
        stage_a_start = time.perf_counter()
        stage_a_result = self.run_stage_a(output_dir)
        stage_a_elapsed = time.perf_counter() - stage_a_start
        self.logger.info("Stage A completed in %.1f seconds | iterations=%d | final_prompt_len=%d",
                        stage_a_elapsed, stage_a_result.get("iterations", 0), len(stage_a_result.get("optimized_prompt", "")))

        # Stage B: Generate final synthetic corpus with optimized prompt
        stage_b_start = time.perf_counter()
        synthetic_texts = self.run_stage_b(output_dir, stage_a_result)
        stage_b_elapsed = time.perf_counter() - stage_b_start
        self.logger.info("Stage B completed in %.1f seconds | synthetic_samples=%d", stage_b_elapsed, len(synthetic_texts))

        # Stage C: Downstream evaluation (GPT-2 fine-tuning and evaluation)
        stage_c_start = time.perf_counter()
        eval_result = self.run_evaluation(output_dir, synthetic_texts)
        stage_c_elapsed = time.perf_counter() - stage_c_start
        self.logger.info("Stage C completed in %.1f seconds", stage_c_elapsed)

        # Write final summary
        summary = {
            "experiment_id": self.config.meta.get("experiment_id"),
            "stage_a": {
                "iterations": stage_a_result.get("iterations", 0),
                "final_prompt": stage_a_result.get("optimized_prompt", ""),
                "prompt_history": stage_a_result.get("prompt_history", []),
                "elapsed_seconds": stage_a_elapsed,
            },
            "stage_b": {
                "synthetic_samples": len(synthetic_texts),
                "elapsed_seconds": stage_b_elapsed,
            },
            "stage_c": {
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
    # Stage A: Prompt optimization through iterative critique and aggregation
    # -------------------------------------------------------------------------

    def run_stage_a(self, output_dir: Path) -> dict[str, Any]:
        """Optimize prompt through iterative critique and aggregation.

        Process:
        1. Generate 100 samples with current prompt
        2. Score with DataInf, select worst 10
        3. Generate 10 critiques for worst samples
        4. Aggregate critiques, update prompt
        5. Repeat until convergence or max iterations

        Args:
            output_dir: Directory to write stage artifacts

        Returns:
            Dict with 'optimized_prompt', 'prompt_history', and 'iterations' keys
        """

        stage_a_dir = ensure_dir(output_dir / "stage_a")
        self.logger.info("Stage A: Prompt optimization through iterative critique")
        cache_signature = self._stage_a_cache_signature()

        # Check for cached results
        cached_path = stage_a_dir / "prompt_update.json"
        if cached_path.exists():
            data = read_json(cached_path)
            if data.get("cache_signature") == cache_signature:
                self.logger.info("Stage A: Loading cached prompt optimization result from %s", cached_path)
                return {
                    "optimized_prompt": data.get("final_prompt", ""),
                    "prompt_history": data.get("prompt_history", []),
                    "iterations": data.get("iterations", 0),
                }
            self.logger.info("Stage A: Ignoring stale cached prompt optimization result at %s", cached_path)

        # Load seed corpus
        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        seed_corpus = self._load_seed_corpus(train_path)
        self.logger.info("Stage A: Loaded %d seed samples from %s", len(seed_corpus), train_path)

        # Get configuration
        generated_count = int(self.config.stage_a.get("generated_count", 100))
        select_top_k = int(self.config.stage_a.get("select_top_k", 10))
        max_iterations = int(self.config.stage_a.get("max_iterations", 10))
        convergence_threshold = float(self.config.stage_a.get("convergence_threshold", 0.1))
        max_probe_samples = int(self.config.stage_a.get("max_probe_samples", 500))

        # Initial prompt from generator config
        current_prompt = self.config.generator.get("initial_prompt", "Generate text that matches the target dataset style.")
        prompt_history = [current_prompt]

        # Build client context once for all iterations (includes MiniLM embedder)
        train_for_probe = seed_corpus[:max_probe_samples]
        self.logger.info("Stage A: Using %d train samples for DataInf probe", len(train_for_probe))

        # Iterative prompt optimization
        all_critiques = []
        # Initialize context variables to None for safe cleanup at end
        client_ctx = None
        critique_ctx = None
        server_ctx = None
        stage_a_scorer, resolved_stage_a_scorer_name = self._resolve_stage_a_scorer()

        for iteration in range(max_iterations):
            self.logger.info("Stage A iteration %d/%d", iteration + 1, max_iterations)

            # Step 1: Generate samples with current prompt
            self.logger.info("Stage A: Generating %d samples with current prompt...", generated_count)
            generated_samples = self._generate_with_prompt(seed_corpus, generated_count, current_prompt)

            # Step 2: Score samples with DataInf
            all_generated = generated_samples  # For probe context
            client_ctx = self._build_client_context(
                train_for_probe,
                all_generated,
                objective_type="pair_alignment" if resolved_stage_a_scorer_name == "ira" else "domain_probe",
            )
            scored_samples = self._score_batched(
                generated_samples,
                client_ctx,
                batch_size=generated_count,
                scorer=stage_a_scorer,
            )

            worst_samples, selection_meta = self._select_stage_a_samples(
                scored_samples,
                select_top_k=select_top_k,
                iteration=iteration,
            )
            iter_dir = ensure_dir(stage_a_dir / f"iteration_{iteration}")
            selection_meta["scorer_name"] = resolved_stage_a_scorer_name
            selection_meta["aggregator_name"] = str(self.config.aggregator.get("name", ""))
            write_json(iter_dir / "selection_summary.json", selection_meta)
            write_jsonl(iter_dir / "worst_samples.jsonl", worst_samples)
            self.logger.info("Stage A: Selected top %d worst samples (worst score: %.4f)", len(worst_samples), worst_samples[0].score if worst_samples else 0)
            if selection_meta.get("selection_mode") == "random_fallback":
                self.logger.info("Stage A: Falling back to random selection (%s)", selection_meta.get("failure_reason", "unknown"))

            # Step 3: Check convergence - if worst score is below threshold, we've converged
            if worst_samples:
                current_worst_score = worst_samples[0].score
                if current_worst_score < convergence_threshold:
                    self.logger.info("Stage A: Converged! Worst score %.4f < threshold %.4f", current_worst_score, convergence_threshold)
                    break

            # Step 4: Generate critiques for worst samples
            self.logger.info("Stage A: Generating critiques for %d worst samples...", len(worst_samples))

            # Convert ScoredSample to Sample for retrieval
            samples_for_critique = [Sample(
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
            ) for s in worst_samples]

            # Build new context with worst samples for critique
            critique_ctx = self._build_client_context(
                train_for_probe,
                samples_for_critique,
                objective_type="pair_alignment" if resolved_stage_a_scorer_name == "ira" else "domain_probe",
            )

            # Retrieve anchor samples for each worst sample
            paired_samples = self._retrieve_batched(samples_for_critique, critique_ctx)

            # Generate critiques
            critiques = self._critique_batched(paired_samples, critique_ctx)
            all_critiques.extend(critiques)
            self.logger.info("Stage A: Generated %d critiques (total: %d)", len(critiques), len(all_critiques))

            # Step 5: Aggregate critiques and update prompt
            # Build a simple server context for aggregation
            server_ctx = self._build_server_context()
            server_ctx.prompt_text = current_prompt
            server_ctx.prompt_history = list(prompt_history)
            server_ctx.text_backend = critique_ctx.text_backend
            prompt_update = self.aggregator.aggregate(all_critiques, server_ctx)

            # Update prompt if we have rules
            if prompt_update and prompt_update.rules:
                new_prompt = self._apply_prompt_update(current_prompt, prompt_update)
                if new_prompt != current_prompt:
                    current_prompt = new_prompt
                    prompt_history.append(current_prompt)
                    self.logger.info("Stage A: Prompt updated (length: %d)", len(current_prompt))
                else:
                    self.logger.info("Stage A: Prompt unchanged, stopping")
                    break
            else:
                self.logger.info("Stage A: No rules from aggregation, stopping")
                break

            # Save iteration artifacts
            write_jsonl(iter_dir / "critiques.jsonl", critiques)
            if prompt_update:
                write_json(iter_dir / "prompt_update.json", prompt_update)

            # Release iteration resources (client_ctx for scoring, critique_ctx for retrieval/critique, server_ctx for aggregation)
            from thesis_platform.core.resource_cleanup import release_component_resources
            release_component_resources(client_ctx, critique_ctx, server_ctx)

        # Save final Stage A artifacts
        final_result = {
            "final_prompt": current_prompt,
            "prompt_history": prompt_history,
            "iterations": iteration + 1,
            "total_critiques": len(all_critiques),
            "cache_signature": cache_signature,
        }
        write_json(stage_a_dir / "prompt_update.json", final_result)
        write_jsonl(stage_a_dir / "critiques.jsonl", all_critiques)

        # Release Stage A resources
        self.logger.info("Stage A: Releasing resources")
        from thesis_platform.core.resource_cleanup import release_component_resources
        # Release generator, scorer, and all contexts (safe even if some were never created)
        release_component_resources(self.generator, self.scorer, client_ctx, critique_ctx, server_ctx)

        return {
            "optimized_prompt": current_prompt,
            "prompt_history": prompt_history,
            "iterations": iteration + 1,
        }

    # -------------------------------------------------------------------------
    # Stage B: Final synthetic corpus generation with optimized prompt
    # -------------------------------------------------------------------------

    def run_stage_b(self, output_dir: Path, stage_a_result: dict[str, Any]) -> list[str]:
        """Generate final synthetic corpus using the optimized prompt from Stage A.

        Args:
            output_dir: Directory to write stage artifacts
            stage_a_result: Output from Stage A containing the optimized prompt

        Returns:
            List of synthetic text strings
        """

        stage_b_dir = ensure_dir(output_dir / "stage_b")
        self.logger.info("Stage B: Final synthetic corpus generation")
        cache_signature = self._stage_b_cache_signature(stage_a_result)

        # Check for cached results
        cached_path = stage_b_dir / "llama7b_text_syn.json"
        stage_config_path = stage_b_dir / "stage_config.json"
        if cached_path.exists() and stage_config_path.exists():
            stage_config = read_json(stage_config_path)
            if stage_config.get("cache_signature") == cache_signature:
                self.logger.info("Stage B: Loading cached synthetic texts from %s", cached_path)
                with open(cached_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            self.logger.info("Stage B: Ignoring stale cached synthetic texts at %s", cached_path)

        # Get optimized prompt from Stage A
        optimized_prompt = stage_a_result.get("optimized_prompt", "")
        if not optimized_prompt:
            self.logger.warning("Stage B: No optimized prompt from Stage A, using initial prompt")
            optimized_prompt = self.config.generator.get("initial_prompt", "Generate text that matches the target dataset style.")

        # Get configuration
        generated_count = int(self.config.stage_b.get("generated_count", 1000))
        self.logger.info("Stage B: Generating %d samples with optimized prompt", generated_count)

        # Load seed corpus
        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        seed_corpus = self._load_seed_corpus(train_path)
        self.logger.info("Stage B: Loaded %d seed samples", len(seed_corpus))

        # Generate synthetic samples using the optimized prompt
        synthetic_samples = self._generate_with_prompt(seed_corpus, generated_count, optimized_prompt)
        synthetic_texts = [s.render_text() for s in synthetic_samples]

        # Light filtering
        filtered = [t for t in synthetic_texts if self._is_valid_sample(t)]
        self.logger.info("Stage B: Generated %d, kept %d after filtering", len(synthetic_texts), len(filtered))

        # Save artifacts
        with open(cached_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        write_json(stage_b_dir / "stage_config.json", {
            "generated_count": generated_count,
            "total_generated": len(synthetic_texts),
            "total_filtered": len(filtered),
            "optimized_prompt": optimized_prompt,
            "cache_signature": cache_signature,
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

    def _build_client_context(self, train_samples: list[Sample], all_samples: list[Sample], *, objective_type: str = "domain_probe") -> ClientContext:
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
            objective_type=objective_type,
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

    def _generate_with_prompt(self, seed_corpus: list[Sample], total_count: int, prompt_text: str) -> list[Sample]:
        """Generate synthetic samples using a custom prompt.

        Similar to _generate_batched but uses a custom prompt text instead of the config's initial prompt.

        Args:
            seed_corpus: Seed samples to use as few-shot examples
            total_count: Total number of samples to generate
            prompt_text: Custom prompt text to use for generation

        Returns:
            List of generated Sample objects
        """
        all_generated = []
        generated_per_round = getattr(self.generator, 'generated_per_round', 16)
        num_full_rounds = total_count // generated_per_round
        remainder = total_count % generated_per_round

        round_id = 0

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
                prompt_text=prompt_text,
                public_seed_samples=seed_subset,
                config=dict(self.config.raw),
                output_dir=self._get_output_dir(),
                text_backend=text_backend,
                sample_id_prefix=f"syn_opt_r{round_id}",
                sample_source="synthetic_optimized",
            )

            batch_samples = self.generator.generate(round_ctx)
            all_generated.extend(batch_samples)
            round_id += 1

            if (round_idx + 1) % 50 == 0:
                self.logger.info("Generation progress: %d / %d samples", len(all_generated), total_count)

        # Partial round if needed
        if remainder > 0 and remainder != generated_per_round:
            start_idx = (num_full_rounds * generated_per_round) % len(seed_corpus)
            seed_subset = [
                seed_corpus[(start_idx + i) % len(seed_corpus)]
                for i in range(self.generator.exemplars_per_prompt)
            ]

            round_ctx = RoundContext(
                round_id=round_id,
                prompt_text=prompt_text,
                public_seed_samples=seed_subset,
                config=dict(self.config.raw),
                output_dir=self._get_output_dir(),
                text_backend=text_backend,
                sample_id_prefix=f"syn_opt_r{round_id}",
                sample_source="synthetic_optimized",
            )

            batch_samples = self.generator.generate(round_ctx)
            # Take only the remainder
            all_generated.extend(batch_samples[:remainder])

        self.logger.info("Generated %d samples with custom prompt (target was %d)", len(all_generated), total_count)

        # Release text_backend used during generation
        if text_backend is not None:
            from thesis_platform.core.resource_cleanup import release_component_resources
            release_component_resources(text_backend)

        return all_generated

    def _score_batched(self, samples: list[Sample], client_ctx: ClientContext, batch_size: int, *, scorer: Any | None = None) -> list[ScoredSample]:
        """Score samples in batches using the configured scorer."""
        scorer = scorer or self.scorer
        all_scored = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            scored = scorer.score(batch, client_ctx)
            all_scored.extend(scored)
            self.logger.info("Stage A scoring batch %d-%d / %d: %d samples",
                           i, min(i + batch_size, len(samples)), len(samples), len(scored))
        return all_scored

    def _resolve_stage_a_scorer(self) -> tuple[Any, str]:
        """Return the scorer instance that Stage A should use."""

        import thesis_platform.adapters  # noqa: F401 - ensure registry is populated for direct runner use

        stage_a_scorer_name = str(self.config.stage_a.get("scorer", "") or "").strip()
        base_scorer_name = str(self.config.scorer.get("name", "") or "").strip()
        requested_name = stage_a_scorer_name or base_scorer_name
        sample_format = str(self.config.data.get("sample_format", "raw_text")).strip().lower()

        resolved_name = requested_name
        if sample_format != "instruction_response":
            if requested_name == "datainf":
                raise ValueError("Single-node raw_text Stage A should use scorer 'datainf_real' instead of 'datainf'.")
            elif requested_name == "gradmm":
                raise ValueError("Single-node raw_text Stage A should use scorer 'gradmm_real' instead of 'gradmm'.")

        scorer_config = dict(self.config.scorer)
        scorer_config.update(dict(self.config.stage_a.get("scorer_config", {})))
        scorer_config["name"] = resolved_name
        if resolved_name == base_scorer_name and not stage_a_scorer_name and not self.config.stage_a.get("scorer_config"):
            return self.scorer, resolved_name
        if resolved_name == base_scorer_name and requested_name == base_scorer_name and not self.config.stage_a.get("scorer_config"):
            return self.scorer, resolved_name
        return create("scorer", resolved_name, scorer_config, self.config.repo_root()), resolved_name

    def _stage_a_cache_signature(self) -> dict[str, Any]:
        """Return the cache signature for Stage A artifacts."""

        return {
            "stage_a": dict(self.config.stage_a),
            "scorer": dict(self.config.scorer),
            "aggregator": dict(self.config.aggregator),
            "data_sample_format": str(self.config.data.get("sample_format", "raw_text")),
        }

    def _stage_b_cache_signature(self, stage_a_result: dict[str, Any]) -> dict[str, Any]:
        """Return the cache signature for Stage B artifacts."""

        return {
            "stage_b": dict(self.config.stage_b),
            "optimized_prompt": str(stage_a_result.get("optimized_prompt", "")),
        }

    def _select_stage_a_samples(
        self,
        scored_samples: list[ScoredSample],
        *,
        select_top_k: int,
        iteration: int,
    ) -> tuple[list[ScoredSample], dict[str, Any]]:
        """Select Stage A bad samples, with random fallback when scoring has no signal."""

        scored_samples = list(scored_samples)
        scored_samples.sort(key=lambda x: x.score, reverse=True)
        failure_reason = self._detect_stage_a_failure(scored_samples, select_top_k=select_top_k)
        if failure_reason is None:
            return scored_samples[:select_top_k], {
                "selection_mode": "scored",
                "failure_reason": "",
                "selected_sample_ids": [sample.sample_id for sample in scored_samples[:select_top_k]],
                "scores": [float(sample.score) for sample in scored_samples],
            }

        seed = int(self.config.stage_a.get("random_fallback_seed", self.config.meta.get("seed", 42)))
        rng = random.Random(seed + iteration)
        selected = rng.sample(scored_samples, min(select_top_k, len(scored_samples)))
        return selected, {
            "selection_mode": "random_fallback",
            "failure_reason": failure_reason,
            "selected_sample_ids": [sample.sample_id for sample in selected],
            "scores": [float(sample.score) for sample in scored_samples],
        }

    def _detect_stage_a_failure(self, scored_samples: list[ScoredSample], *, select_top_k: int) -> str | None:
        """Return a failure reason when the Stage A ranking has no useful signal."""

        if not scored_samples:
            return "empty_scores"
        scores = [float(sample.score) for sample in scored_samples]
        epsilon = float(self.config.stage_a.get("failure_equal_epsilon", 1e-9))
        if max(scores) - min(scores) <= epsilon:
            return "scores_nearly_equal"
        if len(scores) >= 2:
            margin_threshold = float(self.config.stage_a.get("failure_margin_threshold", 0.0))
            top_region = scores[: max(1, min(select_top_k, len(scores)))]
            median_score = float(statistics.median(scores))
            top_region_score = float(statistics.mean(top_region))
            if top_region_score - median_score <= margin_threshold:
                return "weak_top_k_separation"
        return None

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
