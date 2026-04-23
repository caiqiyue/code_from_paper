import unittest

from paper_new_selector.generator_bridge import build_candidate_generator


class GeneratorBridgeTests(unittest.TestCase):
    def test_generator_bridge_uses_one_fixed_generator_source(self):
        generator = build_candidate_generator("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertEqual(generator.contract["backend"], "thesis_pretext_prompt")
        self.assertIn("pretext_prompt_generator", generator.contract["source"])


if __name__ == "__main__":
    unittest.main()
