import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from paper_new_selector.pipeline import run_pipeline
from paper_new_selector.run_selector_single_node import main
from paper_new_selector.thesis_bridge import load_yaml_config


class PipelineSmokeTests(unittest.TestCase):
    def _deep_merge(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _write_temp_config(self, base_config: str, overrides: dict) -> Path:
        config = self._deep_merge(load_yaml_config(base_config), overrides)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_pipeline_returns_stage1_stage2_boundary_and_generator_contract(self):
        summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=True)
        self.assertIn("stage1", summary)
        self.assertIn("stage2", summary)
        self.assertIn("boundary_state", summary["stage1"])
        self.assertIn("generator_contract", summary)
        self.assertIn("eval", summary)
        self.assertTrue(summary["eval"]["enabled"])
        self.assertEqual(summary["eval"]["mode"], "pretext_small")

    def test_validate_only_reports_vllm_contract_for_both_generation_stages(self):
        summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=True)
        self.assertEqual(summary["generator_contract"]["llm_backend"], "vllm")
        self.assertEqual(summary["stage2"]["bootstrap_cfg"]["generator_backend"], "vllm")

    def test_cli_wraps_invalid_non_vllm_config_in_clear_system_exit(self):
        config_path = self._write_temp_config(
            "configs/single_node_jobs_selector.yaml",
            {"llm": {"generator": {"engine": "transformers"}}},
        )
        stdout = io.StringIO()
        with patch("sys.argv", ["run_selector_single_node", "--config", str(config_path)]):
            with patch("sys.stdout", stdout):
                with self.assertRaisesRegex(SystemExit, "strict vLLM generation"):
                    main()


if __name__ == "__main__":
    unittest.main()
