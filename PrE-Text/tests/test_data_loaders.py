from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.data.loaders import load_dataset_bundle, load_eval_texts, load_initialization_texts, load_train_texts


class DataLoaderTests(unittest.TestCase):
    """Validate the supported dataset shapes and deterministic sampling behavior."""

    def test_train_loader_supports_flat_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.json"
            path.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
            texts, meta = load_train_texts(path, max_samples_per_client=8, seed=7)
            self.assertEqual(texts, ["a", "b", "c"])
            self.assertEqual(meta["shape"], "flat_list")

    def test_train_loader_supports_client_buckets_with_deterministic_subsampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.json"
            payload = {
                "0": [f"a{i}" for i in range(6)],
                "1": [f"b{i}" for i in range(6)],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            first, meta_first = load_train_texts(path, max_samples_per_client=3, seed=11)
            second, meta_second = load_train_texts(path, max_samples_per_client=3, seed=11)
            self.assertEqual(first, second)
            self.assertEqual(meta_first["sampled_train_sample_count"], 6)
            self.assertEqual(meta_second["train_client_count"], 2)

    def test_eval_and_initialization_loaders_support_both_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_list = Path(tmp_dir) / "eval_list.json"
            eval_dict = Path(tmp_dir) / "eval_dict.json"
            init_path = Path(tmp_dir) / "init.json"
            eval_list.write_text(json.dumps(["x", "y"]), encoding="utf-8")
            eval_dict.write_text(json.dumps({"1": ["m", "n"]}), encoding="utf-8")
            init_path.write_text(json.dumps(["short text", "this sample contains enough words to survive the initialization filter"]), encoding="utf-8")
            self.assertEqual(load_eval_texts(eval_list), ["x", "y"])
            self.assertEqual(load_eval_texts(eval_dict), ["m", "n"])
            self.assertEqual(
                load_initialization_texts(init_path, min_words=5),
                ["this sample contains enough words to survive the initialization filter"],
            )

    def test_dataset_bundle_uses_default_roots_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            (datasets_dir / "demo_train.json").write_text(json.dumps({"0": ["a", "b"], "1": ["c", "d"]}), encoding="utf-8")
            (datasets_dir / "demo_eval.json").write_text(json.dumps(["e", "f"]), encoding="utf-8")
            (datasets_dir / "initial_set.json").write_text(
                json.dumps(["this initialization sample has many words for filtering purposes only"]),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "demo"},
                    "paths": {
                        "repo_root": ".",
                        "dataset_root": "./datasets",
                        "output_root": "./out",
                    },
                    "data": {
                        "dataset_name": "demo",
                        "max_samples_per_client": 1,
                        "initialization_min_words": 5,
                    },
                },
                base_dir=root,
            )
            bundle = load_dataset_bundle(config)
            self.assertEqual(bundle.dataset_name, "demo")
            self.assertEqual(bundle.train_client_count, 2)
            self.assertEqual(bundle.sampled_train_sample_count, 2)
            self.assertEqual(bundle.eval_texts, ["e", "f"])

    def test_dataset_bundle_applies_optional_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            (datasets_dir / "demo_train.json").write_text(json.dumps(["a", "b", "c", "d"]), encoding="utf-8")
            (datasets_dir / "demo_eval.json").write_text(json.dumps(["e", "f", "g"]), encoding="utf-8")
            (datasets_dir / "initial_set.json").write_text(
                json.dumps(
                    [
                        "this initialization sample has many words for filtering purposes only",
                        "this second initialization sample also has enough words to remain available",
                    ]
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "demo"},
                    "paths": {
                        "repo_root": ".",
                        "dataset_root": "./datasets",
                        "output_root": "./out",
                    },
                    "data": {
                        "dataset_name": "demo",
                        "max_samples_per_client": 8,
                        "initialization_min_words": 5,
                        "train_limit": 2,
                        "eval_limit": 1,
                        "initialization_limit": 1,
                    },
                },
                base_dir=root,
            )
            bundle = load_dataset_bundle(config)
            self.assertEqual(bundle.train_texts, ["a", "b"])
            self.assertEqual(bundle.eval_texts, ["e"])
            self.assertEqual(len(bundle.initialization_texts), 1)
            self.assertEqual(bundle.sampled_train_sample_count, 2)
