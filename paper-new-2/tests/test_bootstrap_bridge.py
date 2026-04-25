import unittest

from paper_new_stage2_selector.bootstrap_bridge import attach_generated_outputs, build_bootstrap_prompt_records


class BootstrapBridgeTests(unittest.TestCase):
    def test_prompt_records_keep_seed_metadata(self):
        records = build_bootstrap_prompt_records(
            ["alpha sample", "beta sample", "gamma sample", "delta sample"],
            num_prompts=2,
            seed=7,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].prompt_index, 0)
        self.assertEqual(len(records[0].seed_texts), 3)
        self.assertIn("Original Text Sample 1", records[0].prompt_text)

    def test_attach_generated_outputs_preserves_prompt_mapping(self):
        prompt_records = build_bootstrap_prompt_records(
            ["alpha sample", "beta sample", "gamma sample"],
            num_prompts=2,
            seed=3,
        )
        generated = attach_generated_outputs(prompt_records, ["useful output text here", "another useful output text"])
        self.assertEqual(generated[0].prompt_index, 0)
        self.assertEqual(generated[1].raw_text, "another useful output text")
        self.assertEqual(len(generated[0].seed_texts), 3)


if __name__ == "__main__":
    unittest.main()
