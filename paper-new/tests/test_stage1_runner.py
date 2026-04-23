import unittest

from paper_new_selector.stage1_runner import _select_seed_samples


class Stage1RunnerTests(unittest.TestCase):
    def test_seed_sample_selection_rotates_across_rounds_but_is_deterministic(self):
        init_samples = [f"sample_{index}" for index in range(8)]
        round0 = _select_seed_samples(
            init_samples,
            exemplar_count=3,
            round_id=0,
            meta_seed=42,
        )
        round1 = _select_seed_samples(
            init_samples,
            exemplar_count=3,
            round_id=1,
            meta_seed=42,
        )
        round0_again = _select_seed_samples(
            init_samples,
            exemplar_count=3,
            round_id=0,
            meta_seed=42,
        )
        self.assertEqual(round0, round0_again)
        self.assertNotEqual(round0, round1)
        self.assertEqual(len(round0), 3)
        self.assertEqual(len(set(round0)), 3)


if __name__ == "__main__":
    unittest.main()
