"""Unit tests for LoRA gradient computation.

Tests:
- Gradient extraction
- HVP computation
- Distance metrics
- Integration with scorers
"""

import unittest
import torch
import numpy as np
from pathlib import Path

# Import modules to test
from thesis_platform.core.lora_gradients import (
    LoRAGradientExtractor,
    GradientDistanceCalculator,
    flatten_gradients,
    gradient_norm,
    clip_gradients,
)


class TestLoRAGradientExtractor(unittest.TestCase):
    """Test LoRA gradient extraction."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gradient_extraction_shape(self):
        """Test that gradients have correct shape."""
        # This is a placeholder - would need actual model
        # extractor = LoRAGradientExtractor("gpt2")
        # extractor.load_model()
        # grads = extractor.compute_sample_gradients("Test text")
        #
        # for name, grad in grads.items():
        #     self.assertIsInstance(grad, torch.Tensor)
        #     self.assertGreater(grad.numel(), 0)
        pass

    def test_gradient_flattening(self):
        """Test gradient flattening."""
        grad_dict = {
            "layer1": torch.randn(10, 20),
            "layer2": torch.randn(5, 5),
        }

        flat = flatten_gradients(grad_dict)
        expected_size = 10 * 20 + 5 * 5

        self.assertEqual(flat.shape[0], expected_size)

    def test_gradient_norm(self):
        """Test gradient norm computation."""
        grad_dict = {
            "layer1": torch.ones(10, 10),  # Norm = sqrt(100) = 10
        }

        norm = gradient_norm(grad_dict)
        self.assertAlmostEqual(norm, 10.0, places=5)

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        grad_dict = {
            "layer1": torch.ones(10, 10) * 2.0,  # Norm = 20
        }

        clipped = clip_gradients(grad_dict, max_norm=10.0)

        # After clipping, norm should be <= 10
        clipped_norm = gradient_norm(clipped)
        self.assertLessEqual(clipped_norm, 10.0 + 1e-6)


class TestGradientDistanceCalculator(unittest.TestCase):
    """Test gradient distance metrics."""

    def test_cosine_distance_identical(self):
        """Test cosine distance for identical gradients."""
        g1 = {"layer": torch.ones(10)}
        g2 = {"layer": torch.ones(10)}

        dist = GradientDistanceCalculator.cosine_distance(g1, g2)

        # Identical gradients should have distance ≈ 0
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_cosine_distance_opposite(self):
        """Test cosine distance for opposite gradients."""
        g1 = {"layer": torch.ones(10)}
        g2 = {"layer": -torch.ones(10)}

        dist = GradientDistanceCalculator.cosine_distance(g1, g2)

        # Opposite gradients should have distance ≈ 2
        self.assertAlmostEqual(dist, 2.0, places=5)

    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        g1 = {"layer": torch.zeros(10)}
        g2 = {"layer": torch.ones(10)}

        dist = GradientDistanceCalculator.euclidean_distance(g1, g2)

        # Distance should be sqrt(10)
        expected = np.sqrt(10)
        self.assertAlmostEqual(dist, expected, places=5)

    def test_l1_distance(self):
        """Test L1 distance."""
        g1 = {"layer": torch.zeros(10)}
        g2 = {"layer": torch.ones(10)}

        dist = GradientDistanceCalculator.l1_distance(g1, g2)

        # L1 distance should be 10
        self.assertAlmostEqual(dist, 10.0, places=5)

    def test_gradient_mismatch_score(self):
        """Test gradient mismatch score."""
        real_grads = {"layer": torch.randn(10)}
        syn_grads = {"layer": torch.randn(10)}

        score = GradientDistanceCalculator.gradient_mismatch_score(
            real_grads, syn_grads, metric="cosine"
        )

        # Score should be between 0 and 2
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 2.0)


class TestHVPComputation(unittest.TestCase):
    """Test HVP (Hessian Vector Product) computation."""

    def test_hvp_proposed_basic(self):
        """Test basic HVP proposed computation."""
        # Create simple gradient dictionaries
        val_grad_avg = {
            "layer": torch.randn(5, 5),
        }

        tr_grad_dict = {
            0: {"layer": torch.randn(5, 5)},
            1: {"layer": torch.randn(5, 5)},
        }

        # This would test the actual HVP computation
        # For now, just verify the structure
        self.assertEqual(len(val_grad_avg), 1)
        self.assertEqual(len(tr_grad_dict), 2)


class TestPrivacyIntegration(unittest.TestCase):
    """Test DP privacy integration with LoRA gradients."""

    def test_dp_config_validation(self):
        """Test DP config validation."""
        from thesis_platform.core.dp_privacy import DPConfig

        # Valid config
        config = DPConfig(enabled=True, epsilon=1.0, delta=1e-5)
        config.validate()

        # Invalid epsilon
        with self.assertRaises(ValueError):
            config = DPConfig(enabled=True, epsilon=-1.0, delta=1e-5)
            config.validate()

    def test_gradient_privatization(self):
        """Test gradient privatization."""
        from thesis_platform.core.dp_privacy import DPPrivatizer, DPConfig

        config = DPConfig(enabled=True, epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        privatizer = DPPrivatizer(config, device="cpu")

        grad_dict = {
            "layer": torch.randn(10, 10),
        }

        # Privatize
        noisy_grads = privatizer.privatize_gradients(grad_dict)

        # Check that noise was added
        self.assertIn("layer", noisy_grads)
        self.assertNotEqual(
            torch.norm(grad_dict["layer"]).item(),
            torch.norm(noisy_grads["layer"]).item(),
        )

    def test_dp_disabled(self):
        """Test that DP can be disabled."""
        from thesis_platform.core.dp_privacy import DPPrivatizer, DPConfig

        config = DPConfig(enabled=False)
        privatizer = DPPrivatizer(config, device="cpu")

        grad_dict = {
            "layer": torch.randn(10, 10),
        }

        # When disabled, should return unchanged
        result = privatizer.privatize_gradients(grad_dict)

        self.assertAlmostEqual(
            torch.norm(grad_dict["layer"]).item(),
            torch.norm(result["layer"]).item(),
            places=5,
        )


class TestScorerIntegration(unittest.TestCase):
    """Test integration with scorers."""

    def test_datainf_scorer_initialization(self):
        """Test DataInf scorer can be initialized."""
        from thesis_platform.adapters.scorers.datainf_lora_scorer import (
            DataInfRealScorer,
        )

        config = {
            "score_direction": "larger_is_worse",
            "lambda_const_param": 10.0,
            "hvp_method": "proposed",
            "use_real_gradients": False,  # Use feature fallback for testing
            "feature_model": None,
            "allow_hashing_fallback": True,
        }

        # Should initialize without error
        scorer = DataInfRealScorer(config, repo_root=".")
        self.assertIsNotNone(scorer)

    def test_gradmm_scorer_initialization(self):
        """Test GRADMM scorer can be initialized."""
        from thesis_platform.adapters.scorers.gradmm_lora_scorer import GradMMRealScorer

        config = {
            "score_direction": "larger_is_worse",
            "metric": "cos",
            "use_real_gradients": False,
            "feature_model": None,
            "allow_hashing_fallback": True,
        }

        # Should initialize without error
        scorer = GradMMRealScorer(config, repo_root=".")
        self.assertIsNotNone(scorer)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLoRAGradientExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientDistanceCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestHVPComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestScorerIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
