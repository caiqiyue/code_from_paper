from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from thesis_platform.data.loaders import load_texts


class LoaderTests(unittest.TestCase):
    """Validate the dataset JSON shapes accepted by the platform."""

    def test_dict_of_lists_expands_to_multiple_samples(self) -> None:
        """Top-level dataset objects should emit one sample per contained text."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.json"
            path.write_text(json.dumps({"0": ["alpha", "beta"], "1": ["gamma"]}), encoding="utf-8")

            texts = load_texts(path)

            self.assertEqual(texts, ["alpha", "beta", "gamma"])

    def test_single_record_dict_remains_one_sample(self) -> None:
        """A JSON object that looks like one record should stay as one flattened sample."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "record.json"
            path.write_text(
                json.dumps({"instruction": "summarize this", "response": "done"}),
                encoding="utf-8",
            )

            texts = load_texts(path)

            self.assertEqual(texts, ["summarize this done"])

    def test_directory_with_train_and_eval_json_counts_every_text(self) -> None:
        """Directories should aggregate JSON files without collapsing eval lists into one sample."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "jobs_train.json").write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
            (root / "jobs_eval.json").write_text(json.dumps({"1": ["gamma", "delta"]}), encoding="utf-8")

            texts = load_texts(root)

            self.assertEqual(len(texts), 4)
            self.assertEqual(set(texts), {"alpha", "beta", "gamma", "delta"})


if __name__ == "__main__":
    unittest.main()
