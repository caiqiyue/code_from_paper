from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import old_automation.old_experiment_queue as queue


class OldExperimentQueueTests(unittest.TestCase):
    def test_queue_contains_only_round3_tuning_experiments(self) -> None:
        labels = [exp.label for exp in queue.QUEUE]

        self.assertEqual(len(labels), 16)
        self.assertEqual(labels[0], "NS-T3-F1-JOBS")
        self.assertEqual(labels[-1], "NS-T3-F4-MICRO")
        self.assertTrue(all(label.startswith("NS-T3-") for label in labels))

    def test_queue_uses_pretext_env_a6000_and_round3_config_paths(self) -> None:
        self.assertTrue(queue.VISIBLE_DEVICE_INDEX == "1")
        self.assertTrue(all(exp.env_name == "pretext" for exp in queue.QUEUE))
        self.assertTrue(all(exp.family == "paper_new" for exp in queue.QUEUE))
        self.assertTrue(
            all(
                exp.config_path.startswith(
                    "paper-new/configs/experiments/single_node_tuning_round3/"
                )
                for exp in queue.QUEUE
            )
        )

    def test_load_state_replaces_legacy_queue_with_round3_items(self) -> None:
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
            ],
            "current_label": "NS-C2",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "old_experiment_queue_state.json"
            state_path.write_text(json.dumps(legacy_state), encoding="utf-8")
            with patch.object(queue, "STATE_PATH", state_path):
                state = queue.load_state()

        self.assertIsNone(state["current_label"])
        self.assertEqual(state["queue"][0]["label"], "NS-T3-F1-JOBS")
        self.assertEqual(state["queue"][-1]["label"], "NS-T3-F4-MICRO")
        self.assertTrue(all(item["status"] == "pending" for item in state["queue"]))

    def test_paper_new_downstream_summary_counts_as_success(self) -> None:
        exp = queue.experiment_def("NS-T3-F1-JOBS")
        remote_outputs = [
            (0, "", "", "host-a"),
            (0, "success:downstream summary and stage2 corpus present\n", "", "host-a"),
        ]

        with patch.object(queue, "run_remote", side_effect=remote_outputs):
            status, note = queue.inspect_experiment(exp)

        self.assertEqual(status, "success")
        self.assertEqual(note, "downstream summary and stage2 corpus present")

    def test_pretext_partial_artifacts_with_live_process_counts_as_running(self) -> None:
        exp = queue.experiment_def("NS-T3-F1-JOBS")
        remote_outputs = [
            (
                0,
                "k8smaster 123 1 0 python -m paper_new_selector.run_selector_single_node --config "
                "configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml\n",
                "",
                "host-a",
            ),
            (0, "partial:artifacts present (eval) without final downstream summary\n", "", "host-a"),
        ]

        with patch.object(queue, "run_remote", side_effect=remote_outputs):
            status, note = queue.inspect_experiment(exp)

        self.assertEqual(status, "running")
        self.assertIn("without final downstream summary", note)

    def test_main_reassigns_current_label_when_stale_label_is_not_running(self) -> None:
        state = {
            "queue": [
                {
                    "label": "NS-T3-F1-JOBS",
                    "actual_experiment_id": "ns_tune3_f1_jobs",
                    "config_path": "paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
                    "status": "running",
                    "note": "stale",
                    "pid": "111",
                    "last_checked_at": None,
                },
                {
                    "label": "NS-T3-F1-CONG",
                    "actual_experiment_id": "ns_tune3_f1_congressional",
                    "config_path": "paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_congressional.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
                {
                    "label": "NS-T3-F1-FORUMS",
                    "actual_experiment_id": "ns_tune3_f1_forums",
                    "config_path": "paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_forums.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
                {
                    "label": "NS-T3-F1-MICRO",
                    "actual_experiment_id": "ns_tune3_f1_microblog",
                    "config_path": "paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_microblog.yaml",
                    "env_name": "pretext",
                    "family": "paper_new",
                    "status": "pending",
                    "note": "",
                    "pid": None,
                    "last_checked_at": None,
                },
            ],
            "current_label": "NS-T3-F1-JOBS",
        }

        inspect_results = [
            ("failed", "missing experiment dir"),
            ("running", "artifacts present (eval) without final downstream summary"),
        ]

        with patch.object(queue, "PLINK", Path(sys.executable)), patch.object(
            queue, "load_state", return_value=state
        ), patch.object(queue, "save_state"), patch.object(queue, "log"), patch.object(
            queue, "inspect_experiment", side_effect=inspect_results
        ), patch.object(
            queue, "any_remote_experiment_running", return_value=True
        ):
            queue.main()

        self.assertEqual(state["current_label"], "NS-T3-F1-CONG")
        self.assertEqual(state["queue"][0]["status"], "failed")
        self.assertEqual(state["queue"][0]["pid"], None)
        self.assertEqual(state["queue"][1]["status"], "running")


if __name__ == "__main__":
    unittest.main()
