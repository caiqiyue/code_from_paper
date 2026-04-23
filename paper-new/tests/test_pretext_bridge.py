import unittest

from paper_new_selector.pretext_bridge import prepare_bootstrap_runtime, resolve_bootstrap_model_path
from paper_new_selector.thesis_bridge import resolve_dataset_paths


class BridgeTests(unittest.TestCase):
    def test_thesis_bridge_resolves_existing_dataset_roots(self):
        train_path, eval_path, init_path = resolve_dataset_paths("configs/single_node_jobs_selector.yaml")
        self.assertIn("thesis_platform/datasets", train_path.as_posix())
        self.assertIn("thesis_platform/datasets", eval_path.as_posix())
        self.assertIn("thesis_platform/datasets", init_path.as_posix())

    def test_pretext_bridge_uses_existing_open_model_root(self):
        model_path = resolve_bootstrap_model_path("thesis_platform/open_model", "llama2_7b")
        self.assertIn("thesis_platform/open_model", model_path.as_posix())

    def test_pretext_bridge_prepares_real_bootstrap_call_contract(self):
        runtime = prepare_bootstrap_runtime("configs/single_node_jobs_selector.yaml")
        self.assertTrue(callable(runtime["build_bootstrap_prompts"]))
        self.assertTrue(callable(runtime["generate_bootstrapped_samples"]))
        self.assertEqual(runtime["bootstrap_cfg"]["generator_backend"], "huggingface")

    def test_pretext_bridge_rejects_non_vllm_backend(self):
        runtime = prepare_bootstrap_runtime("configs/single_node_jobs_selector.yaml")
        self.assertEqual(runtime["generate_bootstrapped_samples"].__name__, "generate_bootstrapped_samples_hf")


if __name__ == "__main__":
    unittest.main()
