from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import old_automation.old_experiment_queue as queue


class OldExperimentQueueTests(unittest.TestCase):
    def test_round2_tuning_experiments_are_appended_after_existing_queue(self) -> None:
        labels = [exp.label for exp in queue.QUEUE]

        self.assertEqual(labels[0], "NS-C1")
        self.assertEqual(labels[17], "SP-C9")
        self.assertEqual(labels[18], "NS-T2-E1-JOBS")
        self.assertEqual(labels[-1], "NS-T2-E6-MICRO")

    def test_load_state_appends_round2_items_without_resetting_existing_current_label(self) -> None:
        legacy_state = {
            "queue": [
                {
                    "label": "NS-C1",
                    "actual_experiment_id": "ns_c1_jobs_base",
                    "config_path": "paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
                    "status": "success",
                    "note": "done",
                    "pid": None,
                    "last_checked_at": "2026-04-25T10:00:00",
                },
                {
                    "label": "NS-C2",
                    "actual_experiment_id": "ns_c2_congressional_base",
                    "config_path": "paper-new/configs/experiments/single_node_formal/ns_c2_congressional_base.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
                    "status": "running",
                    "note": "in flight",
                    "pid": "123",
                    "last_checked_at": "2026-04-25T10:05:00",
                },
            ],
            "current_label": "NS-C2",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "old_experiment_queue_state.json"
            state_path.write_text(json.dumps(legacy_state), encoding="utf-8")
            with patch.object(queue, "STATE_PATH", state_path):
                state = queue.load_state()

        self.assertEqual(state["current_label"], "NS-C2")
        self.assertEqual(state["queue"][1]["status"], "running")
        self.assertEqual(state["queue"][18]["label"], "NS-T2-E1-JOBS")
        self.assertEqual(state["queue"][-1]["label"], "NS-T2-E6-MICRO")
        self.assertEqual(state["queue"][18]["status"], "pending")

    def test_paper_new_downstream_summary_counts_as_success(self) -> None:
        exp = queue.experiment_def("NS-C1")
        remote_outputs = [
            (0, "", "", "host-a"),
            (0, "success:downstream summary and stage2 corpus present\n", "", "host-a"),
        ]

        with patch.object(queue, "run_remote", side_effect=remote_outputs):
            status, note = queue.inspect_experiment(exp)

        self.assertEqual(status, "success")
        self.assertEqual(note, "downstream summary and stage2 corpus present")

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
                    "label": "NS-C1",
                    "actual_experiment_id": "ns_c1_jobs_base",
                    "config_path": "paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
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
                    "label": "NS-C2",
                    "actual_experiment_id": "ns_c2_congressional_base",
                    "config_path": "paper-new/configs/experiments/single_node_formal/ns_c2_congressional_base.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
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
            "current_label": "NS-C1",
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

        self.assertEqual(state["current_label"], "NS-C2")
        self.assertEqual(state["queue"][0]["status"], "failed")
        self.assertEqual(state["queue"][0]["pid"], None)
        self.assertEqual(state["queue"][2]["status"], "running")


if __name__ == "__main__":
    unittest.main()
