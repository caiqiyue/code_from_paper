import tempfile
import unittest
from pathlib import Path

from paper_new_selector.round21_repeat5_runner import (
    append_round21_repeat5_summary_row,
    build_round21_repeat5_child_env,
    build_round21_repeat5_run_specs,
    resolve_round21_repeat5_runtime_output_dir,
)


class Round21Repeat5RunnerTests(unittest.TestCase):
    def test_build_round21_repeat5_run_specs_creates_60_serial_runs(self):
        specs = build_round21_repeat5_run_specs(repeat_count=5)
        self.assertEqual(len(specs), 60)
        self.assertTrue(specs[0].experiment_id.endswith("_repeat01"))
        self.assertTrue(specs[-1].experiment_id.endswith("_repeat05"))

    def test_child_env_sets_cuda_device_order_when_visible_devices_present(self):
        env = build_round21_repeat5_child_env({"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_runtime_output_dir_stays_inside_round21_outputs(self):
        specs = build_round21_repeat5_run_specs(repeat_count=1)
        output_dir = resolve_round21_repeat5_runtime_output_dir(
            Path("D:/tmp/paper-new-round21"),
            specs[0],
        )
        self.assertIn("outputs", output_dir.parts)
        self.assertIn("r21_repeat5", output_dir.parts)

    def test_summary_row_includes_validation_status(self):
        spec = next(
            current
            for current in build_round21_repeat5_run_specs(repeat_count=1)
            if current.base_experiment_id.endswith("_fallback")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "output"
            (outdir / "eval").mkdir(parents=True)
            (outdir / "stage1_budget_calibration.json").write_text(
                '{"mode":"hierarchical_shape_routing","regime":"uncertain","configured_seed_top_k":20,"resolved_seed_top_k":21,"selection_stage":"uncertain_fallback_policy","arbitration_triggered":false}',
                encoding="utf-8",
            )
            (outdir / "eval" / "downstream_eval_summary.json").write_text(
                '{"metrics":{"best_top1":0.1,"best_top3":0.2,"best_top5":0.3,"best_top10":0.4}}',
                encoding="utf-8",
            )
            summary_path = root / "summary.tsv"
            status = append_round21_repeat5_summary_row(summary_path, spec, 0, outdir)
            self.assertEqual(status, "OK")
            content = summary_path.read_text(encoding="utf-8")
            self.assertIn("OK", content)


if __name__ == "__main__":
    unittest.main()
