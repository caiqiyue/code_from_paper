from __future__ import annotations

import unittest

from thesis_platform.adapters.aggregators.uid import UIDAggregator
from thesis_platform.adapters.critics.fedtextgrad_critic import FedTextGradCritic
from thesis_platform.adapters.generators.pretext_generator import PretextSeedGenerator
from thesis_platform.adapters.retrievers.knn_retriever import KNNRetriever
from thesis_platform.adapters.scorers.datainf_scorer import DataInfScorer
from thesis_platform.adapters.scorers.gradmm_scorer import GradMMScorer
from thesis_platform.adapters.scorers.pretext_histogram import PretextHistogramScorer
from thesis_platform.core.context import ClientContext, RoundContext, ServerContext
from thesis_platform.core.schemas import Critique, PairedSample, Sample
from thesis_platform.models.embedding import HashingEmbedder


class AdapterSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = HashingEmbedder()
        self.real_samples = [
            Sample("real_0", "client_0", 0, "real", "demo", "instruction_tuning", "medical report includes diagnosis and treatment plan"),
            Sample("real_1", "client_0", 0, "real", "demo", "instruction_tuning", "medical summary includes symptoms diagnosis and medication"),
        ]
        self.client_ctx = ClientContext(
            client_id="client_0",
            train_samples=self.real_samples,
            validation_samples=self.real_samples[:1],
            all_samples=self.real_samples,
            embedder=self.embedder,
            config={},
        )
        self.synthetic = [
            Sample("syn_0", "server", 0, "synthetic", "demo", "instruction_tuning", "generic report with little detail"),
            Sample("syn_1", "server", 0, "synthetic", "demo", "instruction_tuning", "medical diagnosis summary with treatment"),
        ]

    def test_generator_outputs_samples(self) -> None:
        generator = PretextSeedGenerator({"generated_per_round": 3, "mask": 0.2, "t_steps": 1, "seed": 1}, None)
        round_ctx = RoundContext(0, "Prompt", self.real_samples, {}, None)
        samples = generator.generate(round_ctx)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample.text for sample in samples))

    def test_scorers_output_scored_samples(self) -> None:
        hist = PretextHistogramScorer({}, None).score(self.synthetic, self.client_ctx)
        datainf = DataInfScorer({"lambda_const_param": 10}, None).score(self.synthetic, self.client_ctx)
        gradmm = GradMMScorer({"alpha": 0.5}, None).score(self.synthetic, self.client_ctx)
        self.assertEqual(len(hist), len(self.synthetic))
        self.assertEqual(len(datainf), len(self.synthetic))
        self.assertEqual(len(gradmm), len(self.synthetic))

    def test_retriever_critic_aggregator_chain(self) -> None:
        scored = GradMMScorer({"alpha": 0.5}, None).score(self.synthetic, self.client_ctx)
        pairs = KNNRetriever({"top_k": 1}, None).retrieve(scored[:1], self.client_ctx)
        critiques = FedTextGradCritic({"compress_to_n_rules": 2, "redact_enable": True}, None).critique(pairs, self.client_ctx)
        update = UIDAggregator({"max_rules": 3}, None).aggregate(
            critiques,
            ServerContext("exp", "Base prompt", ["Base prompt"], {}, None),
        )
        self.assertEqual(len(pairs), 1)
        self.assertIsInstance(critiques[0], Critique)
        self.assertTrue(update is None or update.rules)


if __name__ == "__main__":
    unittest.main()
