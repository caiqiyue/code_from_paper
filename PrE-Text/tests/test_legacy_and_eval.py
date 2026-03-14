from __future__ import annotations

import io
import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pretext_platform.core.legacy import build_legacy_config, legacy_experiment_id
from pretext_platform.core.types import DatasetBundle, ModelPaths
from pretext_platform.evaluation.distilgpt2_eval import run_distilgpt2_eval


def _load_module_from_path(name: str, path: Path):
    """Load a module from an arbitrary path for wrapper compatibility tests."""

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LegacyAndEvalTests(unittest.TestCase):
    """Cover the legacy CLI mapping helpers and strict small-eval validation."""

    def test_legacy_config_mapping_preserves_original_experiment_id(self) -> None:
        args = Namespace(
            datadir="congressional",
            outputdir="out",
            sensitivity=8,
            delta=3e-6,
            sigma=2.31,
            mask=0.3,
            lookahead=4,
            multiplier=4,
            seq_len=64,
            t_steps=2,
            trial=0,
            H_multiplier=0.25,
        )
        config = build_legacy_config(args, base_dir=Path.cwd(), stage="stage1")
        self.assertEqual(config.experiment_id(), legacy_experiment_id(args))
        self.assertFalse(config.bootstrap["enabled"])

    def test_legacy_main_wrapper_calls_new_stage_runner(self) -> None:
        module = _load_module_from_path("legacy_main_wrapper", Path(__file__).resolve().parents[1] / "main.py")
        argv = [
            "main.py",
            "-datadir",
            "congressional",
            "-outputdir",
            "out",
            "-sensitivity",
            "8",
            "-sigma",
            "2.31",
        ]
        with patch.object(module, "run_stage1", return_value={"ok": True}) as runner, patch.object(
            sys, "argv", argv
        ), redirect_stdout(io.StringIO()):
            module.main()
        runner.assert_called_once()

    def test_eval_small_fails_fast_when_checkpoint_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = build_legacy_config(
                Namespace(
                    datadir="congressional",
                    outputdir=str(root / "out"),
                    sensitivity=8,
                    delta=3e-6,
                    sigma=2.31,
                    mask=0.3,
                    lookahead=4,
                    multiplier=4,
                    seq_len=64,
                    t_steps=2,
                    trial=0,
                    H_multiplier=0.25,
                ),
                base_dir=root,
                stage="eval_small",
            )
            bundle = DatasetBundle("demo", [], ["eval"], [])
            model_paths = ModelPaths(
                minilm=root / "minilm",
                roberta_large=root / "roberta",
                llama2_7b=root / "llama",
                distilgpt2=root / "distilgpt2",
                c4_checkpoint=root / "missing_checkpoint.pth",
            )
            with self.assertRaisesRegex(ValueError, "c4_checkpoint"):
                run_distilgpt2_eval(config, bundle, model_paths, root / "stage2", root / "eval_small")
