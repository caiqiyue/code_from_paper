import os
import tempfile
import unittest
from pathlib import Path

from paper_new_selector.thesis_bridge import load_yaml_config, resolve_config_path


class ThesisBridgeConfigPathTests(unittest.TestCase):
    def test_relative_config_path_resolves_from_cwd_without_resource_root(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "configs"
            config_dir.mkdir()
            config_path = config_dir / "experiment.yaml"
            config_path.write_text("meta:\n  experiment_id: local_test\n", encoding="utf-8")

            try:
                os.chdir(tmp_path)
                self.assertEqual(
                    resolve_config_path("configs/experiment.yaml"),
                    config_path.resolve(),
                )
                self.assertEqual(
                    load_yaml_config("configs/experiment.yaml")["meta"]["experiment_id"],
                    "local_test",
                )
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
