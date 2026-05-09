import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_new_selector.shared_vllm import SharedVllmSession, build_shared_vllm_session
from thesis_platform.models.backends import VllmTextBackend


class SharedVllmTests(unittest.TestCase):
    def test_backend_exposes_reusable_session_methods(self):
        backend = VllmTextBackend(
            model_path=Path("demo-model"),
            max_model_len=512,
            gpu_memory_utilization=0.35,
            startup_required_free_gb=2,
            tensor_parallel_size=1,
            top_p=1.0,
            enforce_eager=True,
        )
        self.assertTrue(hasattr(backend, "ensure_session"))
        self.assertTrue(hasattr(backend, "generate_batch"))

    def test_build_shared_vllm_session_is_lazy_and_uses_backend_metadata(self):
        backend = SimpleNamespace(
            ensure_session=lambda: ("llm", "sampling"),
            generate_batch=lambda prompts, **_: prompts,
            _model_path=Path("local-llama"),
            _max_model_len=512,
            _gpu_memory_utilization=0.35,
            _tensor_parallel_size=1,
            _enforce_eager=True,
        )
        with patch.object(backend, "ensure_session", wraps=backend.ensure_session) as ensure_session:
            session = build_shared_vllm_session(backend)
        self.assertIsInstance(session, SharedVllmSession)
        self.assertEqual(session.to_dict()["model_path"], "local-llama")
        ensure_session.assert_not_called()

    def test_shared_session_delegates_batch_generation_to_existing_backend(self):
        calls = []

        class _Backend:
            def ensure_session(self):
                return ("llm", "sampling")

            def generate_batch(self, prompts, *, max_new_tokens, temperature=None):
                calls.append((list(prompts), max_new_tokens, temperature))
                return [f"generated::{prompt}" for prompt in prompts]

        backend = _Backend()
        backend._model_path = Path("local-llama")
        backend._max_model_len = 512
        backend._gpu_memory_utilization = 0.35
        backend._tensor_parallel_size = 1
        backend._enforce_eager = True
        session = build_shared_vllm_session(backend)
        outputs = session.generate_batch(["p1", "p2"], max_new_tokens=64, temperature=0.8)
        self.assertEqual(outputs, ["generated::p1", "generated::p2"])
        self.assertEqual(calls, [(["p1", "p2"], 64, 0.8)])


if __name__ == "__main__":
    unittest.main()
