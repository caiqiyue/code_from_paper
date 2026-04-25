import unittest

from paper_new_stage2_selector.contracts import GeneratedSampleRecord
from paper_new_stage2_selector.selector import select_seed_aware_records


class Stage2SelectorTests(unittest.TestCase):
    def test_selector_rejects_low_consistency_and_near_duplicates(self):
        records = [
            GeneratedSampleRecord(0, 0, "p0", ["seed-a", "seed-b", "seed-c"], "text one four words", "text one four words"),
            GeneratedSampleRecord(1, 1, "p1", ["seed-a", "seed-b", "seed-c"], "text one four words again", "text one four words again"),
            GeneratedSampleRecord(2, 2, "p2", ["seed-x", "seed-y", "seed-z"], "text two four words", "text two four words"),
        ]
        result = select_seed_aware_records(
            records=records,
            generated_vectors=[[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]],
            prompt_seed_vectors=[
                [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
                [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
                [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
            ],
            selector_cfg={
                "target_count_mode": "match_eval_clean_count",
                "consistency_threshold": 0.42,
                "duplicate_threshold": 0.95,
                "min_words": 1,
                "prompt_echo_ngram": 6,
                "unique_token_ratio_floor": 0.0,
                "w_consistency": 1.0,
                "w_template": 0.35,
                "w_duplicate": 0.30,
            },
        )
        self.assertEqual(result.target_count, 3)
        self.assertEqual(len(result.selected_records), 2)
        self.assertEqual(result.rejected_records[0].record_index, 1)
        self.assertEqual(result.rejected_records[0].rejected_reason, "near_duplicate")


if __name__ == "__main__":
    unittest.main()
