import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_new_stage2_selector.eval_bridge import run_eval_from_stage2_dir, write_selected_stage2_dir


class EvalBridgeTests(unittest.TestCase):
    def test_write_selected_stage2_dir_persists_pretext_style_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = write_selected_stage2_dir(["a", "b"], output_dir=Path(tmpdir))
            payload = json.loads((stage2_dir / "llama7b_text_syn.json").read_text(encoding="utf-8"))
            self.assertEqual(payload, ["a", "b"])

    def test_run_eval_from_stage2_dir_reads_selected_texts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stage2_dir = write_selected_stage2_dir(["alpha", "beta"], output_dir=output_dir)
            with patch(
                "paper_new_stage2_selector.eval_bridge.run_eval_selected_texts",
                return_value={"enabled": True, "best_top1": 0.33},
            ) as run_eval_selected_texts:
                summary = run_eval_from_stage2_dir("demo.yaml", stage2_dir=stage2_dir, output_dir=output_dir / "eval")
            run_eval_selected_texts.assert_called_once()
            self.assertEqual(summary["best_top1"], 0.33)


if __name__ == "__main__":
    unittest.main()
