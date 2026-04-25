from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CoreImportCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]

    def _run_python(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_downstream_eval_import_succeeds_in_fresh_interpreter(self) -> None:
        result = self._run_python(
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd())); "
            "from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager; "
            "print(DownstreamEvalManager.__name__)"
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("DownstreamEvalManager", result.stdout)

    def test_core_still_exports_single_node_runner(self) -> None:
        result = self._run_python(
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd())); "
            "from thesis_platform.core import SingleNodeRunner; "
            "print(SingleNodeRunner.__name__)"
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SingleNodeRunner", result.stdout)


if __name__ == "__main__":
    unittest.main()
