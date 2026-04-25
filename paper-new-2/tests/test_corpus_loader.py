import unittest

from paper_new_stage2_selector.corpus_loader import extract_baseline_training_text


class CorpusLoaderTests(unittest.TestCase):
    def test_baseline_cleaning_matches_pretext_eval_heuristic(self):
        cleaned = extract_baseline_training_text("useful synthetic text for training Orig trailing junk")
        self.assertEqual(cleaned, "useful synthetic text for training")
        self.assertEqual(extract_baseline_training_text("too short"), "")


if __name__ == "__main__":
    unittest.main()
