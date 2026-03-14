from __future__ import annotations

from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.pipeline import run_pipeline
from pretext_platform.core.types import DatasetBundle, ModelPaths, StageSummary

try:
    import numpy as np
    import torch

    HAS_STAGE_DEPS = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    HAS_STAGE_DEPS = False


if HAS_STAGE_DEPS:
    class FakeAccelerator:
        """Tiny stand-in for accelerate.Accelerator used by the Stage 1 tests."""

        is_main_process = True
        num_processes = 1
        process_index = 0

        def print(self, *args, **kwargs):
            del args, kwargs

        def prepare_model(self, model, evaluation_mode=False):
            del evaluation_mode
            return model

        def prepare(self, dataloader):
            return dataloader

        def gather(self, tensor):
            return tensor

        def wait_for_everyone(self):
            return None

        @contextmanager
        def split_between_processes(self, items):
            yield [items[0]]


    class FakeTokenizer:
        """Small tokenizer mock with deterministic token ids and decoding."""

        pad_token_id = 0
        mask_token_id = 99

        def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
            del return_tensors, padding, truncation, max_length
            rows = []
            for idx, _ in enumerate(texts, start=1):
                rows.append([idx, idx + 1, 0, 0])
            input_ids = torch.tensor(rows)
            attention_mask = (input_ids != 0).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def batch_decode(self, inputs, skip_special_tokens=True):
            del skip_special_tokens
            if isinstance(inputs, torch.Tensor):
                rows = inputs.tolist()
            else:
                rows = [item.tolist() if isinstance(item, torch.Tensor) else item for item in inputs]
            return [f"decoded_{row[0]}" for row in rows]


    class FakeMPNet:
        """Sentence-transformer stand-in used by the Stage 1 tests."""

        def get_sentence_embedding_dimension(self):
            return 2


    class FakeModel(torch.nn.Module):
        """Minimal torch module accepted by the Stage 1 runner."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(1, 1)

        def forward(self, input_ids=None, attention_mask=None):
            del attention_mask
            shape = (input_ids.shape[0], input_ids.shape[1], 8)
            return SimpleNamespace(logits=torch.zeros(shape))


class Stage1AndPipelineTests(unittest.TestCase):
    """Validate the Stage 1 orchestration and the top-level pipeline shell."""

    @unittest.skipUnless(HAS_STAGE_DEPS, "numpy and torch are required for the Stage 1 orchestration test")
    def test_stage1_writes_expected_artifacts_with_mocked_algorithm_steps(self) -> None:
        from pretext_platform.algorithms.stage1 import run_private_evolution_stage

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "stage1_demo", "seed": 7},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "data": {"max_samples_per_client": 8},
                    "stage1": {
                        "rounds": 3,
                        "sigma": 2.31,
                        "delta": 3e-6,
                        "sensitivity": 8,
                        "mask": 0.3,
                        "lookahead": 1,
                        "multiplier": 1,
                        "seq_len": 4,
                        "t_steps": 1,
                        "batch_size": 4,
                        "embed_batch_size": 4,
                        "num_workers": 1,
                        "H_multiplier": 0.25,
                    },
                    "runtime": {"device": "cpu"},
                },
                base_dir=root,
            )
            bundle = DatasetBundle(
                dataset_name="demo",
                train_texts=["real sample one", "real sample two"],
                eval_texts=[],
                initialization_texts=["this initialization sample has enough words to pass filtering"],
            )
            model_paths = ModelPaths(
                minilm=root / "minilm",
                roberta_large=root / "roberta",
                llama2_7b=root / "llama",
                distilgpt2=root / "distilgpt2",
            )

            def fake_model_loader(_model_paths, *, device):
                del _model_paths, device
                return FakeTokenizer(), FakeModel(), FakeMPNet()

            def fake_histogram(private_embeddings, parent_set, attention_mask, mlm_probability, config_dict):
                del private_embeddings, attention_mask, mlm_probability, config_dict
                return np.ones(parent_set.shape[0]), 0.5, np.array([0])

            def fake_variation(parent_set, variation_deg, config_dict):
                del variation_deg, config_dict
                return parent_set

            with patch("pretext_platform.algorithms.stage1._compute_epsilon", return_value=1.23), patch(
                "pretext_platform.algorithms.stage1.Similarity.concat_embedding",
                return_value=np.zeros((2, 2)),
            ), patch(
                "pretext_platform.algorithms.stage1.NN_Histogram.dp_nn_histogram",
                side_effect=fake_histogram,
            ), patch(
                "pretext_platform.algorithms.stage1.Variation.produce_variation",
                side_effect=fake_variation,
            ):
                summary = run_private_evolution_stage(
                    config,
                    bundle,
                    model_paths,
                    root / "out" / "stage1",
                    accelerator=FakeAccelerator(),
                    model_loader=fake_model_loader,
                )

            self.assertEqual(summary.metrics["rounds"], 3)
            self.assertEqual(len(summary.artifacts["generated_files"]), 3)
            self.assertTrue((root / "out" / "stage1" / "private_embeds.npy").exists())
            self.assertTrue((root / "out" / "stage1" / "generated_text_it0.json").exists())
            self.assertTrue((root / "out" / "stage1" / "surviving_text_it2.json").exists())

    def test_pipeline_writes_stage_summaries_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "pipe_demo"},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "stage1": {"enabled": True},
                    "bootstrap": {"enabled": True},
                    "eval_small": {"enabled": False},
                    "eval_large": {"enabled": False},
                },
                base_dir=root,
            )
            stage1_summary = StageSummary("stage1", root / "out" / "pipe_demo" / "stage1", {"generated_files": []}, {"rounds": 2})
            stage2_summary = StageSummary("stage2", root / "out" / "pipe_demo" / "stage2", {"synthetic_corpus_path": "x"}, {"generated_count": 3})
            with patch("pretext_platform.core.pipeline.run_stage1", return_value=stage1_summary), patch(
                "pretext_platform.core.pipeline.run_bootstrap",
                return_value=stage2_summary,
            ):
                summary = run_pipeline(config)
            exp_dir = root / "out" / "pipe_demo"
            self.assertEqual(summary["experiment_id"], "pipe_demo")
            self.assertTrue((exp_dir / "resolved_config.json").exists())
            self.assertTrue((exp_dir / "stage1_summary.json").exists())
            self.assertTrue((exp_dir / "stage2_summary.json").exists())
            self.assertTrue((exp_dir / "metrics_summary.json").exists())
