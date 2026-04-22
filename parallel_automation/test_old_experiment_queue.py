from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import parallel_automation.old_experiment_queue as queue


class OldExperimentQueueTests(unittest.TestCase):
    def test_thesis_metrics_summary_counts_as_success_without_latest_pointer(self) -> None:
        exp = queue.experiment_def("SP-C6")
        remote_outputs = [
            (0, "", "", "host-a"),
            (0, "success:metrics and eval_small present\n", "", "host-a"),
        ]

        with patch.object(queue, "run_remote", side_effect=remote_outputs):
            status, note = queue.inspect_experiment(exp)

        self.assertEqual(status, "success")
        self.assertEqual(note, "metrics and eval_small present")

    def test_pretext_partial_artifacts_with_live_process_counts_as_running(self) -> None:
        exp = queue.experiment_def("SP-C7")
        remote_outputs = [
            (
                0,
                "k8smaster 123 1 0 python -m pretext_platform.scripts.run_pipeline --config "
                "configs/experiments/single_node_formal/sp_c7_jobs_no_privacy.yaml\n",
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
                    "label": "SP-C6",
                    "actual_experiment_id": "sp_c6_jobs_eps758",
                    "config_path": "PrE-Text/configs/experiments/single_node_formal/sp_c6_jobs_eps758.yaml",
                    "env_name": "pretext",
                    "family": "pretext",
                    "status": "running",
                    "note": "stale",
                    "pid": "111",
                    "last_checked_at": None,
                },
                {
                    "label": "SP-C7",
                    "actual_experiment_id": "sp_c7_jobs_no_privacy",
                    "config_path": "PrE-Text/configs/experiments/single_node_formal/sp_c7_jobs_no_privacy.yaml",
                    "env_name": "pretext",
                    "family": "pretext",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
                {
                    "label": "SP-C8",
                    "actual_experiment_id": "sp_c8_jobs_seed123",
                    "config_path": "PrE-Text/configs/experiments/single_node_formal/sp_c8_jobs_seed123.yaml",
                    "env_name": "pretext",
                    "family": "pretext",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
                {
                    "label": "SP-C9",
                    "actual_experiment_id": "sp_c9_jobs_seed456",
                    "config_path": "PrE-Text/configs/experiments/single_node_formal/sp_c9_jobs_seed456.yaml",
                    "env_name": "pretext",
                    "family": "pretext",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
            ],
            "current_label": "SP-C6",
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

        self.assertEqual(state["current_label"], "SP-C7")
        self.assertEqual(state["queue"][0]["status"], "failed")
        self.assertEqual(state["queue"][0]["pid"], None)
        self.assertEqual(state["queue"][1]["status"], "running")


if __name__ == "__main__":
    unittest.main()
