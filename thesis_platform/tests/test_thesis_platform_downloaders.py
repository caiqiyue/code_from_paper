from __future__ import annotations

from datetime import datetime, timezone
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
from thesis_platform.dataset_downloaders import pretext_utils
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
        (raw / "train.jsonl").write_text('{"value": 1}\n{"value": 2}\n', encoding="utf-8")
        return {"message": "ok", "metadata": {"source_dataset": "test", "raw_format": "jsonl"}}


class _FailingDatasetDownloader(_SuccessfulDatasetDownloader):
    name = "failure"
    description = "failing dataset"

    def perform_download_raw(self, force: bool):
        raise RuntimeError("boom")


class _SuccessfulPretextDatasetDownloader(_SuccessfulDatasetDownloader):
    name = "pretext_success"
    description = "test pretext dataset"
    optional = True
    pretext_c4_category = "jobs"


class _SuccessfulPretextForumsDatasetDownloader(_SuccessfulDatasetDownloader):
    name = "pretext_forums_success"
    description = "test pretext forums dataset"
    optional = True
    pretext_c4_category = "forums"


class _SuccessfulModelDownloader(BaseModelDownloader):
    name = "success_model"
    description = "test model"
    repo_id = "org/test-model"
    optional = False
    parameter_count_billions = 1.0
    model_size_label = "1B"

    def __init__(self, root: Path, repo_override: str | None = None) -> None:
        super().__init__(repo_override=repo_override)
        self._root = root

    def target_path(self) -> Path:
        return self._root / self.name

    def perform_download(self, force: bool):
        self.target_path().mkdir(parents=True, exist_ok=True)
        (self.target_path() / "weights.bin").write_bytes(b"1234567890")
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

        dataset_names = get_registered_dataset_names(include_optional=True)
        default_dataset_names = get_registered_dataset_names(include_optional=False)
        self.assertIn("glue_sst2", dataset_names)
        self.assertIn("datainf_math_with_reason", dataset_names)
        self.assertIn("rt_polarity", dataset_names)
        self.assertIn("pretext_jobs", dataset_names)
        self.assertIn("pretext_initialization_c4_en", dataset_names)
        self.assertNotIn("pretext_jobs", default_dataset_names)

        formatter_names = get_registered_dataset_formatter_names()
        self.assertIn("identity", formatter_names)
        self.assertIn("imdb", formatter_names)
        self.assertIn("gsm8k", formatter_names)
        self.assertIn("livebench", formatter_names)
        self.assertIn("pretext_json", formatter_names)

        all_model_names = get_registered_model_names(include_optional=True, include_large=True)
        default_model_names = get_registered_model_names(include_optional=False, include_large=False)
        self.assertIn("all_minilm_l6_v2", all_model_names)
        self.assertIn("distilgpt2", all_model_names)
        self.assertIn("llama_2_7b_hf", all_model_names)
        self.assertIn("flan_t5_3b", all_model_names)
        self.assertIn("roberta_large", all_model_names)
        self.assertIn("llama_2_13b_chat_hf", all_model_names)
        self.assertIn("all_minilm_l6_v2", default_model_names)
        self.assertNotIn("distilgpt2", default_model_names)
        self.assertNotIn("llama_2_13b_chat_hf", default_model_names)
        self.assertNotIn("llama_3_1_405b_instruct", default_model_names)

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
            self.assertFalse(summary["include_optional"])
            self.assertIn("raw_path", summary["results"][0])
            self.assertIn("formatted_path", summary["results"][0])
            self.assertIn("sample_counts", summary["results"][0])
            self.assertEqual(summary["results"][0]["sample_counts"]["raw"]["splits"]["train"], 2)

    def test_dataset_controller_excludes_optional_by_default(self) -> None:
        """Verify optional datasets stay out of the default resolved set."""

        default_names = [downloader.name for downloader in dataset_controller.resolve_dataset_downloaders()]
        optional_names = [
            downloader.name
            for downloader in dataset_controller.resolve_dataset_downloaders(include_optional=True)
        ]
        self.assertNotIn("pretext_jobs", default_names)
        self.assertNotIn("pretext_initialization_c4_en", default_names)
        self.assertIn("pretext_jobs", optional_names)
        self.assertIn("pretext_initialization_c4_en", optional_names)

    def test_pretext_cache_builds_only_requested_categories(self) -> None:
        """Verify the C4 cache builder only materializes the requested PrE-Text buckets."""

        text = " ".join(["sample"] * 25)
        fake_rows = [
            {"text": f"{text} init {index}", "url": f"https://example.com/article/{index}"}
            for index in range(3)
        ] + [
            {"text": f"{text} jobs {index}", "url": f"https://indeed.com/jobs/{index}"}
            for index in range(2)
        ]
        fake_datasets = types.SimpleNamespace(load_dataset=mock.Mock(return_value=fake_rows))

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datasets"
            with mock.patch.object(pretext_utils, "datasets_root", return_value=root):
                with mock.patch.object(
                    pretext_utils,
                    "_PRETEXT_C4_TARGETS",
                    {"jobs": 2, "forums": 2, "microblog": 2, "code": 2, "initialization": 3},
                ):
                    with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                        cache_root = pretext_utils.ensure_pretext_c4_cache(
                            required_categories=["initialization", "jobs"],
                            force=True,
                        )
                        self.assertTrue((cache_root / "initialization.jsonl").exists())
                        self.assertTrue((cache_root / "jobs.jsonl").exists())
                        self.assertFalse((cache_root / "forums.jsonl").exists())
                        self.assertFalse((cache_root / "microblog.jsonl").exists())
                        self.assertFalse((cache_root / "code.jsonl").exists())

    def test_dataset_controller_prewarms_only_selected_pretext_categories(self) -> None:
        """Verify one download run prewarms only the PrE-Text buckets requested this time."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datasets"
            downloaders = [_SuccessfulPretextDatasetDownloader(root), _SuccessfulDatasetDownloader(root)]
            with mock.patch.object(dataset_controller, "resolve_dataset_downloaders", return_value=downloaders):
                with mock.patch.object(dataset_controller, "datasets_root", return_value=root):
                    with mock.patch.object(pretext_utils, "ensure_pretext_c4_cache") as warm_cache:
                        dataset_controller.download_datasets(include_optional=True)

        warm_cache.assert_called_once_with(required_categories=["jobs"], force=False)

    def test_pretext_cache_preserves_completed_categories_before_failure(self) -> None:
        """Verify completed PrE-Text categories survive one interrupted C4 streaming pass."""

        text = " ".join(["sample"] * 25)

        def interrupted_rows():
            yield {"text": f"{text} job 0", "url": "https://indeed.com/jobs/0"}
            yield {"text": f"{text} job 1", "url": "https://indeed.com/jobs/1"}
            raise RuntimeError("network boom")

        fake_datasets = types.SimpleNamespace(load_dataset=mock.Mock(return_value=interrupted_rows()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datasets"
            with mock.patch.object(pretext_utils, "datasets_root", return_value=root):
                with mock.patch.object(
                    pretext_utils,
                    "_PRETEXT_C4_TARGETS",
                    {"jobs": 2, "forums": 2, "microblog": 2, "code": 2, "initialization": 3},
                ):
                    with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                        with self.assertRaises(RuntimeError):
                            pretext_utils.ensure_pretext_c4_cache(
                                required_categories=["jobs", "forums"],
                                force=True,
                            )

            self.assertTrue((root / "_pretext_c4_cache" / "jobs.jsonl").exists())
            self.assertFalse((root / "_pretext_c4_cache" / "forums.jsonl").exists())

    def test_dataset_controller_uses_categories_completed_before_cache_failure(self) -> None:
        """Verify one partial cache build still downloads categories already persisted to cache."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datasets"
            cache_root = root / "_pretext_c4_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            (cache_root / "jobs.jsonl").write_text('{"text":"ok","url":"https://indeed.com/jobs/0"}\n', encoding="utf-8")
            downloaders = [
                _SuccessfulPretextDatasetDownloader(root),
                _SuccessfulPretextForumsDatasetDownloader(root),
            ]
            with mock.patch.object(dataset_controller, "resolve_dataset_downloaders", return_value=downloaders):
                with mock.patch.object(dataset_controller, "datasets_root", return_value=root):
                    with mock.patch.object(pretext_utils, "datasets_root", return_value=root):
                        with mock.patch.object(
                            pretext_utils,
                            "ensure_pretext_c4_cache",
                            side_effect=RuntimeError("interrupted after jobs completed"),
                        ):
                            summary = dataset_controller.download_datasets(include_optional=True)

        self.assertEqual(summary["counts"]["downloaded"], 1)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["results"][0]["name"], "pretext_success")
        self.assertEqual(summary["results"][0]["status"], "downloaded")
        self.assertEqual(summary["results"][1]["name"], "pretext_forums_success")
        self.assertEqual(summary["results"][1]["status"], "failed")

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
            self.assertGreater(summary["results"][0]["disk_usage_bytes"], 0)
            self.assertEqual(summary["results"][0]["model_size_label"], "1B")

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

    def test_model_controller_excludes_optional_and_large_by_default(self) -> None:
        """Verify optional models and >15B models are excluded from the default resolved set."""

        default_names = [
            downloader.name
            for downloader in model_controller.resolve_model_downloaders(
                include_optional=False,
                include_large=False,
            )
        ]
        optional_names = [
            downloader.name
            for downloader in model_controller.resolve_model_downloaders(
                include_optional=True,
                include_large=False,
            )
        ]
        large_names = [
            downloader.name
            for downloader in model_controller.resolve_model_downloaders(
                include_optional=False,
                include_large=True,
            )
        ]
        self.assertIn("llama_3_1_8b_instruct", default_names)
        self.assertNotIn("llama_2_13b_chat_hf", default_names)
        self.assertNotIn("llama_3_1_405b_instruct", default_names)
        self.assertIn("llama_2_13b_chat_hf", optional_names)
        self.assertIn("llama_3_1_405b_instruct", large_names)

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

    def test_write_jsonl_serializes_datetime_values(self) -> None:
        """Verify JSONL helpers serialize datetime fields for LiveBench-style rows."""

        from thesis_platform.core.io_utils import write_jsonl

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.jsonl"
            write_jsonl(path, [{"created": datetime(2026, 3, 11, 13, 3, 49, tzinfo=timezone.utc)}])
            payload = path.read_text(encoding="utf-8")

        self.assertIn("2026-03-11T13:03:49+00:00", payload)


if __name__ == "__main__":
    unittest.main()
