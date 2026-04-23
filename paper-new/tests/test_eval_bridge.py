import unittest

from paper_new_selector.eval_bridge import prepare_eval_runtime


class EvalBridgeTests(unittest.TestCase):
    def test_eval_runtime_uses_real_pretext_small_eval_contract(self):
        runtime = prepare_eval_runtime("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertTrue(runtime["enabled"])
        self.assertEqual(runtime["mode"], "pretext_small")
        self.assertEqual(runtime["small_eval_mode"], "gpt2")


if __name__ == "__main__":
    unittest.main()
