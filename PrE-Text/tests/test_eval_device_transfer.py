from __future__ import annotations

import sys
import types
import unittest
from typing import Callable
from unittest.mock import patch

from pretext_platform.evaluation import distilgpt2_eval, gpt2_eval


class StopForward(RuntimeError):
    """Raised by the fake model to stop after inspecting the forwarded batch."""


class FakeTensor:
    """Minimal tensor-like object for testing `.to(device)` behavior without torch."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def to(self, device: object) -> "FakeTensor":
        return FakeTensor(str(device))


class RecordingModel:
    """Fake model that records the batch received by `evaluate()`."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.seen_batch: dict[str, FakeTensor] | None = None

    def eval(self) -> "RecordingModel":
        return self

    def __call__(self, **batch: FakeTensor):
        self.seen_batch = batch
        raise StopForward("stop after recording batch devices")


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class EvalDeviceTransferTests(unittest.TestCase):
    def _assert_eval_moves_batch_to_model_device(self, evaluate_fn: Callable[..., object]) -> None:
        fake_torch = types.SimpleNamespace(no_grad=lambda: _NoGrad())
        loader = [
            {
                "input_ids": FakeTensor(),
                "attention_mask": FakeTensor(),
                "labels": FakeTensor(),
            }
        ]
        model = RecordingModel(device="cuda:0")

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(StopForward):
                evaluate_fn(model, loader, xent_loss=object())

        assert model.seen_batch is not None
        self.assertEqual(
            {name: tensor.device for name, tensor in model.seen_batch.items()},
            {
                "input_ids": "cuda:0",
                "attention_mask": "cuda:0",
                "labels": "cuda:0",
            },
        )

    def test_gpt2_eval_moves_eval_batch_to_model_device(self) -> None:
        self._assert_eval_moves_batch_to_model_device(gpt2_eval.evaluate)

    def test_distilgpt2_eval_moves_eval_batch_to_model_device(self) -> None:
        self._assert_eval_moves_batch_to_model_device(distilgpt2_eval.evaluate)


if __name__ == "__main__":
    unittest.main()
