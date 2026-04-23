import unittest

from paper_new_selector.generator_bridge import build_candidate_generator


class GeneratorBridgeTests(unittest.TestCase):
    def test_generator_bridge_uses_one_fixed_generator_source(self):
        generator = build_candidate_generator("configs/single_node_jobs_selector.yaml")
        self.assertEqual(generator.contract["backend"], "thesis_pretext_prompt")
        self.assertIn("pretext_prompt_generator", generator.contract["source"])
        self.assertEqual(generator.contract["max_prompt_chars"], 192)
        self.assertEqual(generator.contract["max_exemplar_chars"], 220)


if __name__ == "__main__":
    unittest.main()
