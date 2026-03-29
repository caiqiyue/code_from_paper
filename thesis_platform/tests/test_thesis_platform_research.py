from __future__ import annotations

from collections import defaultdict
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_platform.adapters.aggregators.dbscan_attn_tsgdm import DBSCANAttnTSGDMAggregator
from thesis_platform.adapters.generators.pretext_prompt_generator import PretextPromptLLMGenerator
from thesis_platform.adapters.scorers.datainf_paper_scorer import DataInfPaperScorer
from thesis_platform.adapters.scorers.datainf_scorer import DataInfScorer
from thesis_platform.adapters.scorers.gradmm_paper_scorer import GradMMPaperScorer
from thesis_platform.adapters.scorers.gradmm_scorer import GradMMScorer
from thesis_platform.adapters.scorers.ira_scorer import IRAScorer
from thesis_platform.algorithms.aggregators.dbscan_core import aggregate_dbscan_critiques
from thesis_platform.algorithms.aggregators.dbscan_core import _is_usable_summary_rule
from thesis_platform.algorithms.critics.fedtextgrad_core import build_textual_gradient_critique
from thesis_platform.algorithms.critics.fedtextgrad_core import _is_usable_rule
from thesis_platform.algorithms.redaction import redact_rule_text
from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.experiment_runner import ExperimentRunner
from thesis_platform.core.schemas import Critique, PairedSample, Sample
from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples
from thesis_platform.models.backends import MockTextBackend
from thesis_platform.models.embedding import HashingEmbedder


class ResearchModeTests(unittest.TestCase):
    """Validate the new research-mode components and end-to-end behavior."""

    def test_instruction_response_loader_populates_structured_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pairs.json"
            path.write_text(
                json.dumps(
                    [
                        {"instruction": "Summarize the note", "response": "The note is concise.", "label": "ok"},
                        {"instruction": "Classify the text", "response": "positive", "label": "positive"},
                    ]
                ),
                encoding="utf-8",
            )
            samples = load_samples(
                path,
                dataset_name="tmp",
                source="real",
                task_type="instruction_tuning",
                round_id=0,
                client_id="raw",
                prefix="pair",
                sample_format="instruction_response",
            )
            self.assertEqual(samples[0].instruction, "Summarize the note")
            self.assertEqual(samples[0].response, "The note is concise.")
            self.assertEqual(samples[1].label, "positive")
            self.assertIn("Instruction:", samples[0].text)

    def test_preserve_buckets_partition_keeps_bucket_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.json"
            path.write_text(json.dumps({"0": ["alpha", "beta"], "1": ["gamma"], "2": ["delta"]}), encoding="utf-8")
            samples = load_samples(
                path,
                dataset_name="tmp",
                source="real",
                task_type="instruction_tuning",
                round_id=0,
                client_id="raw",
                prefix="real",
                sample_format="raw_text",
            )
            partitions = partition_samples(
                samples,
                num_clients=2,
                max_samples_per_client=4,
                validation_ratio=0.5,
                seed=7,
                strategy="preserve_buckets",
            )
            owners: dict[str, set[str]] = defaultdict(set)
            for partition in partitions:
                for sample in partition["all"]:
                    owners[str(sample.meta.get("bucket_id"))].add(sample.client_id)
            self.assertTrue(all(len(client_ids) == 1 for client_ids in owners.values()))

    def test_preserve_buckets_partition_rejects_insufficient_bucket_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.json"
            path.write_text(json.dumps(["alpha", "beta", "gamma"]), encoding="utf-8")
            samples = load_samples(
                path,
                dataset_name="tmp",
                source="real",
                task_type="instruction_tuning",
                round_id=0,
                client_id="raw",
                prefix="real",
                sample_format="raw_text",
            )
            with self.assertRaises(ValueError) as error:
                partition_samples(
                    samples,
                    num_clients=2,
                    max_samples_per_client=4,
                    validation_ratio=0.5,
                    seed=7,
                    strategy="preserve_buckets",
                )
            self.assertIn("requires at least as many distinct bucket_id values", str(error.exception))

    def test_prompt_llm_generator_is_prompt_sensitive(self) -> None:
        seeds = [
            Sample("seed_0", "server", 0, "public_seed", "tmp", "instruction_tuning", "formal policy guidance"),
            Sample("seed_1", "server", 0, "public_seed", "tmp", "instruction_tuning", "casual diary fragment"),
        ]
        generator = PretextPromptLLMGenerator({"generated_per_round": 2, "exemplars_per_prompt": 1}, None)
        backend = MockTextBackend(role="server")
        samples_a = generator.generate(RoundContext(0, "Prompt A", seeds, {}, None, backend))
        samples_b = generator.generate(RoundContext(0, "Prompt B", seeds, {}, None, backend))
        self.assertNotEqual([sample.text for sample in samples_a], [sample.text for sample in samples_b])

    def _build_domain_client_context(self) -> ClientContext:
        embedder = HashingEmbedder()
        positives = [
            Sample("real_0", "client_0", 0, "real", "demo", "instruction_tuning", "medical diagnosis treatment plan"),
            Sample("real_1", "client_0", 0, "real", "demo", "instruction_tuning", "clinical report with medication"),
        ]
        negatives = [
            Sample("neg_0", "client_1", 0, "real", "demo", "instruction_tuning", "travel diary about beaches"),
            Sample("neg_1", "client_1", 0, "real", "demo", "instruction_tuning", "holiday coffee notes"),
        ]
        return ClientContext(
            client_id="client_0",
            train_samples=positives,
            validation_samples=positives[:1],
            all_samples=positives,
            embedder=embedder,
            config={},
            negative_samples=negatives,
            objective_type="domain_probe",
        )

    def test_datainf_and_gradmm_rank_off_domain_samples_as_worse(self) -> None:
        client_ctx = self._build_domain_client_context()
        candidates = [
            Sample("syn_good", "server", 0, "synthetic", "demo", "instruction_tuning", "medical treatment summary"),
            Sample("syn_bad", "server", 0, "synthetic", "demo", "instruction_tuning", "vacation diary with coffee"),
        ]
        datainf_scores = DataInfScorer({"objective": "domain_probe", "probe_epochs": 60, "probe_lr": 0.2}, None).score(
            candidates, client_ctx
        )
        gradmm_scores = GradMMScorer({"objective": "domain_probe", "probe_epochs": 60, "probe_lr": 0.2}, None).score(
            candidates, client_ctx
        )
        self.assertGreater(datainf_scores[1].score, datainf_scores[0].score)
        self.assertGreater(gradmm_scores[1].score, gradmm_scores[0].score)

    def test_ira_scores_misaligned_pairs_as_worse(self) -> None:
        client_ctx = ClientContext(
            client_id="client_0",
            train_samples=[],
            validation_samples=[],
            all_samples=[],
            embedder=HashingEmbedder(),
            config={},
            text_backend=MockTextBackend(role="client"),
            objective_type="pair_alignment",
        )
        samples = [
            Sample(
                "pair_good",
                "server",
                0,
                "synthetic",
                "demo",
                "instruction_tuning",
                "Instruction: medical summary\nResponse: medical summary",
                instruction="medical summary",
                response="medical summary",
            ),
            Sample(
                "pair_bad",
                "server",
                0,
                "synthetic",
                "demo",
                "instruction_tuning",
                "Instruction: medical summary\nResponse: beach holiday coffee",
                instruction="medical summary",
                response="beach holiday coffee",
            ),
        ]
        scores = IRAScorer({}, None).score(samples, client_ctx)
        self.assertGreater(scores[1].score, scores[0].score)

    def test_datainf_paper_falls_back_cleanly_when_gradients_unavailable(self) -> None:
        client_ctx = self._build_domain_client_context()
        candidates = [
            Sample("syn_good", "server", 0, "synthetic", "demo", "instruction_tuning", "medical treatment summary"),
            Sample("syn_bad", "server", 0, "synthetic", "demo", "instruction_tuning", "vacation diary with coffee"),
        ]
        scorer = DataInfPaperScorer(
            {
                "use_real_gradients": True,
                "feature_model": "",
                "allow_hashing_fallback": True,
            },
            Path("."),
        )

        with patch.object(scorer, "_extract_per_sample_gradients", return_value=None), patch.object(
            scorer, "_compute_validation_gradients", return_value=None
        ):
            scores = scorer.score(candidates, client_ctx)

        self.assertEqual(len(scores), len(candidates))
        self.assertTrue(all(score.meta["use_real_gradients"] is False for score in scores))
        self.assertTrue(
            all(score.meta["fallback_reason"] == "LoRA gradients not available" for score in scores)
        )

    def test_gradmm_paper_respects_disabled_real_gradients_flag(self) -> None:
        client_ctx = self._build_domain_client_context()
        candidates = [
            Sample("syn_good", "server", 0, "synthetic", "demo", "instruction_tuning", "medical treatment summary"),
            Sample("syn_bad", "server", 0, "synthetic", "demo", "instruction_tuning", "vacation diary with coffee"),
        ]
        scorer = GradMMPaperScorer(
            {
                "use_real_gradients": False,
                "feature_model": "",
                "allow_hashing_fallback": True,
            },
            Path("."),
        )

        with patch.object(scorer, "_get_model_for_client", side_effect=AssertionError("should not load LoRA model")):
            scores = scorer.score(candidates, client_ctx)

        self.assertEqual(len(scores), len(candidates))
        self.assertTrue(all(score.meta["use_real_gradients"] is False for score in scores))
        self.assertTrue(
            all(score.meta["fallback_reason"] == "real_gradients_disabled" for score in scores)
        )

    def test_dbscan_tsgdm_updates_memory(self) -> None:
        repo_root = Path.cwd()
        aggregator = DBSCANAttnTSGDMAggregator(
            {
                "max_rules": 2,
                "embedding_model": "models/all-MiniLM-L6-v2",
                "allow_hashing_fallback": True,
                "cluster_eps": 0.5,
                "cluster_min_samples": 1,
                "momentum_beta": 0.6,
            },
            repo_root,
        )
        server_ctx = ServerContext(
            "exp",
            "Base prompt",
            ["Base prompt"],
            {},
            None,
            MockTextBackend(role="server"),
            base_prompt="Base prompt",
        )
        critiques_round_1 = [
            Critique("c1", "client_0", 0, "bad_0", ["real_0"], ["Add more clinical terminology."], "Add more clinical terminology.", {"source_score": 1.0}),
            Critique("c2", "client_1", 0, "bad_1", ["real_1"], ["Add more clinical terminology."], "Add more clinical terminology.", {"source_score": 1.2}),
        ]
        update_1 = aggregator.aggregate(critiques_round_1, server_ctx)
        self.assertTrue(update_1 is not None and update_1.rules)

        critiques_round_2 = [
            Critique("c3", "client_0", 1, "bad_2", ["real_2"], ["Add more clinical terminology."], "Add more clinical terminology.", {"source_score": 1.1}),
            Critique("c4", "client_1", 1, "bad_3", ["real_3"], ["Tighten overall medical specificity."], "Tighten overall medical specificity.", {"source_score": 0.9}),
        ]
        server_ctx.prompt_history.append(update_1.prompt_text)
        update_2 = aggregator.aggregate(critiques_round_2, server_ctx)
        self.assertTrue(server_ctx.aggregation_memory.get("entries"))
        self.assertTrue(update_2 is not None and update_2.meta.get("memory_rules"))

    def test_dbscan_skips_llm_summary_when_cluster_rules_already_fit_budget(self) -> None:
        class FakeEmbedder:
            backend_name = "fake"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                vectors: list[list[float]] = []
                for index, _ in enumerate(texts):
                    vectors.append([1.0 if i == index else 0.0 for i in range(len(texts))])
                return vectors

        critiques = [
            Critique(
                "c1",
                "client_0",
                0,
                "bad_0",
                ["real_0"],
                ["Add more concrete detail and domain-specific structure to match the retrieved real examples."],
                "",
                {"source_score": 1.0},
            ),
            Critique(
                "c2",
                "client_1",
                0,
                "bad_1",
                ["real_1"],
                ["Remove generic or off-domain wording so the text stays aligned with the retrieved real examples."],
                "",
                {"source_score": 0.9},
            ),
        ]

        with patch(
            "thesis_platform.algorithms.aggregators.dbscan_core.summarize_rules_with_llm",
            side_effect=AssertionError("llm summary should be skipped"),
        ):
            update, _ = aggregate_dbscan_critiques(
                critiques,
                round_id=0,
                max_rules=5,
                embedder=FakeEmbedder(),
                text_backend=MockTextBackend(role="server"),
                eps=0.01,
                min_samples=1,
            )

        self.assertIsNotNone(update)
        self.assertIn(
            "Add more concrete detail and domain-specific structure to match the retrieved real examples.",
            update.rules,
        )
        self.assertIn(
            "Remove generic or off-domain wording so the text stays aligned with the retrieved real examples.",
            update.rules,
        )

    def test_fedtextgrad_llm_falls_back_when_model_returns_low_signal_json_fragments(self) -> None:
        class BrokenCritiqueBackend:
            backend_name = "broken"

            def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
                del prompt, max_new_tokens, temperature
                return "```json\n[\n```"

        pair = PairedSample(
            pair_id="pair_0",
            client_id="client_0",
            round_id=0,
            bad_sample=Sample("bad_0", "server", 0, "synthetic", "tmp", "instruction_tuning", "generic short note"),
            real_samples=[
                Sample(
                    "real_0",
                    "client_0",
                    0,
                    "real",
                    "tmp",
                    "instruction_tuning",
                    "Detailed hiring guidance with concrete interview steps and structured recruiter language.",
                )
            ],
        )

        critique = build_textual_gradient_critique(
            pair,
            text_backend=BrokenCritiqueBackend(),
            max_rules=2,
            redact_enable=True,
        )

        self.assertTrue(critique.rules)
        self.assertNotIn("```json", " ".join(critique.rules))
        self.assertTrue(critique.meta["used_fallback"])
        self.assertEqual(critique.meta["fallback_reason"], "low_signal_llm_output")

    def test_fedtextgrad_llm_rejects_copy_like_rules_and_uses_fallback(self) -> None:
        class CopyLikeCritiqueBackend:
            backend_name = "copy_like"

            def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
                del prompt, max_new_tokens, temperature
                return json.dumps(
                    {
                        "rules": [
                            "Join our regional barbecue workshop on Saturday at the downtown training center from 10 AM to 4 PM for anyone preparing to compete in the local cooking contest."
                        ]
                    }
                )

        real_text = (
            "Join our regional barbecue workshop on Saturday at the downtown training "
            "center from 10 AM to 4 PM for anyone preparing to compete in the local cooking contest."
        )
        pair = PairedSample(
            pair_id="pair_copy",
            client_id="client_0",
            round_id=0,
            bad_sample=Sample("bad_copy", "server", 0, "synthetic", "tmp", "instruction_tuning", "generic short note"),
            real_samples=[
                Sample(
                    "real_copy",
                    "client_0",
                    0,
                    "real",
                    "tmp",
                    "instruction_tuning",
                    real_text,
                )
            ],
        )

        critique = build_textual_gradient_critique(
            pair,
            text_backend=CopyLikeCritiqueBackend(),
            max_rules=2,
            redact_enable=True,
        )

        self.assertTrue(critique.rules)
        self.assertTrue(critique.meta["used_fallback"])
        self.assertEqual(critique.meta["fallback_reason"], "copy_like_llm_output")
        self.assertFalse(any("barbecue workshop" in rule.lower() for rule in critique.rules))
        self.assertTrue(all(len(rule.split()) <= 16 for rule in critique.rules))

    def test_redaction_preserves_action_verbs_and_redacts_acronyms(self) -> None:
        redacted = redact_rule_text("Add ING leadership context without copying names.")

        self.assertTrue(redacted.startswith("Add "))
        self.assertIn("<ENTITY>", redacted)
        self.assertNotIn("ING", redacted)

    def test_rule_filters_reject_named_entity_fact_statements(self) -> None:
        pair = PairedSample(
            pair_id="pair_fact",
            client_id="client_0",
            round_id=0,
            bad_sample=Sample("bad_fact", "server", 0, "synthetic", "tmp", "instruction_tuning", "generic hiring note"),
            real_samples=[
                Sample(
                    "real_fact",
                    "client_0",
                    0,
                    "real",
                    "tmp",
                    "instruction_tuning",
                    "The candidate should be able to treat adults in the acute unit and coordinate triage.",
                )
            ],
        )

        self.assertFalse(
            _is_usable_rule(
                "A candidate should be able to treat adults in the acute unit.",
                pair,
            )
        )
        self.assertFalse(
            _is_usable_summary_rule("The company ING is headquartered in New York City.")
        )
        self.assertFalse(
            _is_usable_summary_rule("A user interacting with a platform.")
        )
        self.assertFalse(
            _is_usable_summary_rule("Add a new word or phrase to the sentence.")
        )
        self.assertFalse(_is_usable_summary_rule("Add"))
        self.assertFalse(_is_usable_rule("Focus on <ENTITY>", pair))
        self.assertTrue(
            _is_usable_summary_rule(
                "Add more concrete detail and domain-specific structure to match the retrieved real examples."
            )
        )

    def test_research_pipeline_uses_prompt_updates_across_rounds(self) -> None:
        repo_root = Path.cwd().resolve()
        model_path = (repo_root / "models" / "all-MiniLM-L6-v2").resolve()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            seed_path = tmp_root / "seed.json"
            train_path.write_text(
                json.dumps(
                    {
                        "0": ["medical policy guidance", "clinical protocol update"],
                        "1": ["financial report summary", "budget hearing testimony"],
                    }
                ),
                encoding="utf-8",
            )
            seed_path.write_text(json.dumps(["seed alpha beta", "seed gamma delta", "seed epsilon zeta"]), encoding="utf-8")
            config_path = tmp_root / "research.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: tmp_research
  seed: 7
paths:
  repo_root: "{repo_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
data:
  dataset_name: tmp
  task_type: instruction_tuning
  sample_format: raw_text
  partition_strategy: preserve_buckets
  train_path: "{train_path.as_posix()}"
  public_seed_path: "{seed_path.as_posix()}"
  max_public_seed_samples: 3
  num_clients: 2
  max_samples_per_client: 2
  validation_ratio: 0.5
federation:
  rounds: 2
  top_k_bad: 1
generator:
  name: pretext_prompt_llm
  initial_prompt: "Generate medical-domain text."
  generated_per_round: 2
  exemplars_per_prompt: 1
scorer:
  name: datainf
  objective: domain_probe
  probe_epochs: 50
  probe_lr: 0.2
  damping: 0.01
retriever:
  name: knn
  top_k: 1
  embedding_model: "{model_path.as_posix()}"
  allow_hashing_fallback: true
critic:
  name: fedtextgrad_llm
  compress_to_n_rules: 2
  redact_enable: true
aggregator:
  name: dbscan_attn_tsgdm
  max_rules: 2
  embedding_model: "{model_path.as_posix()}"
  allow_hashing_fallback: true
  cluster_eps: 0.5
  cluster_min_samples: 1
  momentum_beta: 0.6
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            summary = ExperimentRunner(config).run()
            exp_dir = tmp_root / "out" / "tmp_research"
            round_0 = exp_dir / "round_000"
            round_1 = exp_dir / "round_001"
            generated_0 = (round_0 / "generated_samples.jsonl").read_text(encoding="utf-8")
            generated_1 = (round_1 / "generated_samples.jsonl").read_text(encoding="utf-8")
            self.assertNotEqual(generated_0, generated_1)
            self.assertTrue((round_0 / "generation_requests.jsonl").exists())
            self.assertTrue((round_1 / "aggregation_clusters.json").exists())
            self.assertTrue((round_1 / "probe_metrics.json").exists())
            self.assertEqual(summary["round_count"], 2)


if __name__ == "__main__":
    unittest.main()
