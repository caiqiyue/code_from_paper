from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from thesis_platform.dataset_downloaders import create_dataset_downloader, get_registered_dataset_names
from thesis_platform.dataset_downloaders import controller as dataset_controller
from thesis_platform.dataset_downloaders import datainf_generation
from thesis_platform.dataset_downloaders.base import BaseDatasetDownloader
from thesis_platform.dataset_downloaders import common as dataset_common
from thesis_platform.dataset_downloaders.common import datasets_root, package_root as dataset_package_root, to_package_relative
from thesis_platform.dataset_formatters import get_registered_dataset_formatter_names
from thesis_platform.model_downloaders import create_model_downloader, get_registered_model_names
from thesis_platform.model_downloaders import controller as model_controller
from thesis_platform.model_downloaders.base import BaseModelDownloader
from thesis_platform.model_downloaders.common import models_root, package_root as model_package_root


class _SuccessfulDatasetDownloader(BaseDatasetDownloader):
    name = "success"
    description = "test dataset"

    def __init__(self, root: Path) -> None:
        self._root = root

    def dataset_root(self) -> Path:
        return self._root / self.name

    def perform_download_raw(self, force: bool):
        raw = self.raw_path()
        assert raw is not None
        raw.mkdir(parents=True, exist_ok=True)
        return {"message": "ok", "metadata": {"source_dataset": "test", "raw_format": "directory"}}


class _FailingDatasetDownloader(_SuccessfulDatasetDownloader):
    name = "failure"
    description = "failing dataset"

    def perform_download_raw(self, force: bool):
        raise RuntimeError("boom")


class _SuccessfulModelDownloader(BaseModelDownloader):
    name = "success_model"
    description = "test model"
    repo_id = "org/test-model"
    optional = False

    def __init__(self, root: Path, repo_override: str | None = None) -> None:
        super().__init__(repo_override=repo_override)
        self._root = root

    def target_path(self) -> Path:
        return self._root / self.name

    def perform_download(self, force: bool):
        self.target_path().mkdir(parents=True, exist_ok=True)
        return {"message": "ok", "metadata": {"source_type": "test"}}


class _FailingModelDownloader(_SuccessfulModelDownloader):
    name = "failure_model"
    description = "failing model"
    repo_id = "org/failure-model"

    def perform_download(self, force: bool):
        raise RuntimeError("boom")


class DownloaderTests(unittest.TestCase):
    """Validate downloader registration, controller behavior, and helper paths."""

    def test_registered_names_include_expected_entries(self) -> None:
        """Verify dataset registries expose the expected public names."""

        dataset_names = get_registered_dataset_names()
        self.assertIn("glue_sst2", dataset_names)
        self.assertIn("datainf_math_with_reason", dataset_names)
        self.assertIn("rt_polarity", dataset_names)

        formatter_names = get_registered_dataset_formatter_names()
        self.assertIn("identity", formatter_names)
        self.assertIn("imdb", formatter_names)
        self.assertIn("gsm8k", formatter_names)
        self.assertIn("livebench", formatter_names)

        all_model_names = get_registered_model_names(include_optional=True)
        default_model_names = get_registered_model_names(include_optional=False)
        self.assertIn("roberta_large", all_model_names)
        self.assertIn("llama_2_13b_chat_hf", all_model_names)
        self.assertNotIn("llama_2_13b_chat_hf", default_model_names)

    def test_path_helpers_ignore_current_working_directory(self) -> None:
        """Verify default download roots stay anchored to the package, not the shell cwd."""

        expected_datasets_root = dataset_package_root() / "datasets"
        expected_models_root = model_package_root() / "open_model"
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                self.assertEqual(datasets_root(), expected_datasets_root)
                self.assertEqual(models_root(), expected_models_root)
                self.assertEqual(to_package_relative(expected_datasets_root), "datasets")
                self.assertNotIn("\\", to_package_relative(expected_datasets_root))
            finally:
                os.chdir(original_cwd)

    def test_dataset_controller_continues_after_failure(self) -> None:
        """Verify the dataset controller writes a report even when one downloader fails."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datasets"
            downloaders = [_SuccessfulDatasetDownloader(root), _FailingDatasetDownloader(root)]
            with mock.patch.object(dataset_controller, "resolve_dataset_downloaders", return_value=downloaders):
                with mock.patch.object(dataset_controller, "datasets_root", return_value=root):
                    summary = dataset_controller.download_datasets()

            self.assertEqual(summary["counts"]["downloaded"], 1)
            self.assertEqual(summary["counts"]["failed"], 1)
            self.assertTrue((root / "download_report.json").exists())
            self.assertIn("raw_path", summary["results"][0])
            self.assertIn("formatted_path", summary["results"][0])

    def test_model_controller_continues_after_failure(self) -> None:
        """Verify the model controller writes a report even when one downloader fails."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "open_model"
            downloaders = [_SuccessfulModelDownloader(root), _FailingModelDownloader(root)]
            with mock.patch.object(model_controller, "resolve_model_downloaders", return_value=downloaders):
                with mock.patch.object(model_controller, "models_root", return_value=root):
                    summary = model_controller.download_models()

            self.assertEqual(summary["counts"]["downloaded"], 1)
            self.assertEqual(summary["counts"]["failed"], 1)
            self.assertTrue((root / "download_report.json").exists())

    def test_datainf_wrappers_share_one_script_invocation(self) -> None:
        """Verify the three DataInf wrappers only trigger the upstream script once."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            thesis_root = tmp_root / "thesis_platform"
            thesis_root.mkdir(parents=True, exist_ok=True)
            datasets_dir = thesis_root / "datasets"

            def fake_run(*args, **kwargs):
                for pair in datainf_generation.datainf_output_paths().values():
                    for path in pair:
                        path.mkdir(parents=True, exist_ok=True)
                return None

            with mock.patch.object(datainf_generation, "package_root", return_value=thesis_root):
                with mock.patch.object(datainf_generation, "repo_root", return_value=tmp_root):
                    with mock.patch.object(datainf_generation, "datasets_root", return_value=datasets_dir):
                        with mock.patch.object(dataset_common, "datasets_root", return_value=datasets_dir):
                            with mock.patch.object(datainf_generation.subprocess, "run", side_effect=fake_run) as run_mock:
                                datainf_generation.reset_datainf_generation_cache()
                                create_dataset_downloader("datainf_grammars").download()
                                create_dataset_downloader("datainf_math_without_reason").download()
                                create_dataset_downloader("datainf_math_with_reason").download()

            self.assertEqual(run_mock.call_count, 1)
            self.assertTrue((datasets_dir / "datainf_grammars" / "formatted" / "train.hf").exists())
            self.assertTrue((datasets_dir / "datainf_math_without_reason" / "formatted" / "train.hf").exists())
            self.assertTrue((datasets_dir / "datainf_math_with_reason" / "formatted" / "train.hf").exists())

    def test_model_controller_excludes_optional_by_default(self) -> None:
        """Verify optional models are excluded from the default resolved set."""

        default_names = [downloader.name for downloader in model_controller.resolve_model_downloaders(include_optional=False)]
        optional_names = [downloader.name for downloader in model_controller.resolve_model_downloaders(include_optional=True)]
        self.assertIn("llama_3_1_8b_instruct", default_names)
        self.assertNotIn("llama_2_13b_chat_hf", default_names)
        self.assertIn("llama_2_13b_chat_hf", optional_names)

    def test_repo_overrides_change_the_resolved_model_source(self) -> None:
        """Verify CLI-style repo overrides win over the default model source."""

        [downloader] = model_controller.resolve_model_downloaders(
            names=["llama_3_1_8b_instruct"],
            repo_overrides={"llama_3_1_8b_instruct": "custom-user/Llama-3.1-8B-Instruct"},
        )
        self.assertEqual(downloader.default_repo_id, "unsloth/Meta-Llama-3.1-8B-Instruct")
        self.assertEqual(downloader.resolved_repo_id, "custom-user/Llama-3.1-8B-Instruct")
        self.assertTrue(downloader.repo_overridden)

    def test_llama_repo_validation_rejects_non_transformers_mirrors(self) -> None:
        """Verify community-mirror validation rejects non-Transformers repos."""

        downloader = create_model_downloader("llama_3_1_8b_instruct", repo_override="user/llama-gguf")
        info = types.SimpleNamespace(library_name=None, pipeline_tag="text-generation", tags=["gguf"], sha="abc")
        fake_hf = types.SimpleNamespace(model_info=mock.Mock(return_value=info))
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            with self.assertRaises(ValueError):
                downloader.validate_repo()


if __name__ == "__main__":
    unittest.main()
