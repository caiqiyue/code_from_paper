from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from thesis_platform.data.loaders import load_samples, load_texts


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

    def test_pretext_formatted_train_recovers_bucket_ids_from_raw_sidecar(self) -> None:
        """Formatted PrE-Text corpora should restore source-domain buckets from raw JSONL sidecars."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            formatted_dir = root / "formatted"
            raw_dir = root / "raw"
            formatted_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            (formatted_dir / "jobs_train.json").write_text(
                json.dumps(["alpha job post", "beta hiring memo"]),
                encoding="utf-8",
            )
            (raw_dir / "train.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"text": "alpha job post", "url": "https://jobs.example.com/posting/1"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"text": "beta hiring memo", "url": "https://careers.other.org/open-role"},
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            samples = load_samples(
                formatted_dir / "jobs_train.json",
                dataset_name="jobs",
                source="real",
                task_type="instruction_tuning",
                round_id=0,
                client_id="raw",
                prefix="job",
            )

            self.assertEqual(samples[0].meta["bucket_id"], "jobs.example.com")
            self.assertEqual(samples[0].meta["source_domain"], "jobs.example.com")
            self.assertEqual(samples[1].meta["bucket_id"], "careers.other.org")
            self.assertIn("source_url", samples[1].meta)


if __name__ == "__main__":
    unittest.main()
