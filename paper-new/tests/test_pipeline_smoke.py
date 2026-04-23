import unittest

from paper_new_selector.pipeline import run_pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_returns_stage1_stage2_boundary_and_generator_contract(self):
        summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=True)
        self.assertIn("stage1", summary)
        self.assertIn("stage2", summary)
        self.assertIn("boundary_state", summary["stage1"])
        self.assertIn("generator_contract", summary)
        self.assertIn("eval", summary)
        self.assertTrue(summary["eval"]["enabled"])
        self.assertEqual(summary["eval"]["mode"], "pretext_small")


if __name__ == "__main__":
    unittest.main()
