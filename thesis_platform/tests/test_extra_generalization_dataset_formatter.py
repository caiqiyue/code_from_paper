from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thesis_platform.dataset_scripts.format_extra_generalization_datasets import (  # noqa: E402
    write_pretext_json_dataset,
)


class ExtraGeneralizationDatasetFormatterTests(unittest.TestCase):
    def test_write_pretext_json_dataset_uses_congressional_compatible_shape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="extra_generalization_") as tmp:
            dataset_root = Path(tmp) / "twitter_emotion_binary"
            report = write_pretext_json_dataset(
                dataset_root=dataset_root,
                dataset_name="twitter_emotion_binary",
                train_texts=["happy short post", "sad short post"],
                eval_texts=["held out post"],
                source_note="unit-test",
            )

            train = json.loads((dataset_root / "formatted" / "twitter_emotion_binary_train.json").read_text(encoding="utf-8"))
            eval_payload = json.loads((dataset_root / "formatted" / "twitter_emotion_binary_eval.json").read_text(encoding="utf-8"))
            metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(train, ["happy short post", "sad short post"])
            self.assertEqual(eval_payload, {"1": ["held out post"]})
            self.assertEqual(metadata["formatter_name"], "pretext_json")
            self.assertEqual(metadata["formatted_format"], "json")
            self.assertEqual(metadata["split_sizes"], {"train": 2, "eval": 1})
            self.assertEqual(report["train_count"], 2)
            self.assertEqual(report["eval_count"], 1)


if __name__ == "__main__":
    unittest.main()
