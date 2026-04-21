from __future__ import annotations

import unittest

from thesis_platform.adapters.generators.pretext_prompt_generator import PretextPromptLLMGenerator
from thesis_platform.core.context import RoundContext
from thesis_platform.core.schemas import Sample


class PretextPromptGeneratorTests(unittest.TestCase):
    def test_can_bound_prompt_and_exemplar_text(self) -> None:
        seed = Sample(
            "seed_0",
            "server",
            0,
            "public_seed",
            "tmp",
            "instruction_tuning",
            "A" * 80,
        )
        generator = PretextPromptLLMGenerator(
            {
                "generated_per_round": 1,
                "exemplars_per_prompt": 1,
                "max_prompt_chars": 12,
                "max_exemplar_chars": 16,
            },
            None,
        )

        prompt = generator._build_prompt(
            round_ctx=RoundContext(0, "B" * 80, [seed], {}, None),
            source_samples=[seed],
        )

        self.assertIn("B" * 12 + "...", prompt)
        self.assertNotIn("B" * 13, prompt)
        self.assertIn("A" * 16 + "...", prompt)
        self.assertNotIn("A" * 17, prompt)


if __name__ == "__main__":
    unittest.main()
