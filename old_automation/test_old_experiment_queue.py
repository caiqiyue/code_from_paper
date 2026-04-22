from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import old_automation.old_experiment_queue as queue


class OldExperimentQueueTests(unittest.TestCase):
    def test_thesis_metrics_summary_counts_as_success_without_latest_pointer(self) -> None:
        exp = queue.experiment_def("SN-C1")
        remote_outputs = [
            (0, "", "", "host-a"),
            (0, "success:metrics_summary present\n", "", "host-a"),
        ]

        with patch.object(queue, "run_remote", side_effect=remote_outputs):
            status, note = queue.inspect_experiment(exp)

        self.assertEqual(status, "success")
        self.assertEqual(note, "metrics_summary present")

    def test_pretext_partial_artifacts_with_live_process_counts_as_running(self) -> None:
        exp = queue.experiment_def("SP-C1")
        remote_outputs = [
            (
                0,
                "k8smaster 123 1 0 python -m pretext_platform.scripts.run_pipeline --config "
                "configs/experiments/single_node_formal/sp_c1_jobs_base.yaml\n",
                "",
                "host-a",
            ),
            (0, "partial:metrics exists but eval_small missing\n", "", "host-a"),
        ]

        with patch.object(queue, "run_remote", side_effect=remote_outputs):
            status, note = queue.inspect_experiment(exp)

        self.assertEqual(status, "running")
        self.assertIn("metrics exists but eval_small missing", note)

    def test_main_reassigns_current_label_when_stale_label_is_not_running(self) -> None:
        state = {
            "queue": [
                {
                    "label": "SN-C1",
                    "actual_experiment_id": "sn_c1_jobs_base",
                    "config_path": "thesis_platform/configs/experiments/single_node_formal/sn_c1_jobs_base.yaml",
                    "env_name": "caiqiyue-vllm",
                    "family": "thesis",
                    "status": "running",
                    "note": "stale",
                    "pid": "111",
                    "last_checked_at": None,
                },
                {
                    "label": "SP-C1",
                    "actual_experiment_id": "sp_c1_jobs_base",
                    "config_path": "PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml",
                    "env_name": "pretext",
                    "family": "pretext",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
                {
                    "label": "SN-C2",
                    "actual_experiment_id": "sn_c2_congressional_base",
                    "config_path": "thesis_platform/configs/experiments/single_node_formal/sn_c2_congressional_base.yaml",
                    "env_name": "caiqiyue-vllm",
                    "family": "thesis",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
                {
                    "label": "SP-C2",
                    "actual_experiment_id": "sp_c2_congressional_base",
                    "config_path": "PrE-Text/configs/experiments/single_node_formal/sp_c2_congressional_base.yaml",
                    "env_name": "pretext",
                    "family": "pretext",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
            ],
            "current_label": "SN-C1",
        }

        inspect_results = [
            ("failed", "missing experiment dir"),
            ("running", "metrics exists but eval_small missing"),
        ]

        with patch.object(queue, "PLINK", Path(sys.executable)), patch.object(
            queue, "load_state", return_value=state
        ), patch.object(queue, "save_state"), patch.object(queue, "log"), patch.object(
            queue, "inspect_experiment", side_effect=inspect_results
        ), patch.object(
            queue, "any_remote_experiment_running", return_value=True
        ):
            queue.main()

        self.assertEqual(state["current_label"], "SP-C1")
        self.assertEqual(state["queue"][0]["status"], "failed")
        self.assertEqual(state["queue"][0]["pid"], None)
        self.assertEqual(state["queue"][1]["status"], "running")


if __name__ == "__main__":
    unittest.main()
