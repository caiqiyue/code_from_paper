from __future__ import annotations

import inspect
import os
import shutil
import time
from pathlib import Path

from .hf import HuggingFaceModelDownloader
from .common import remove_path
from .registry import register_model_downloader


@register_model_downloader
class MetaLlama3_8BDownloader(HuggingFaceModelDownloader):
    """Download the public community mirror of the Meta Llama 3 8B model."""

    name = "Meta-Llama-3-8B"
    repo_id = "NousResearch/Meta-Llama-3-8B"
    description = "Download the public community-mirrored Meta Llama 3 8B checkpoint."
    optional = True
    community_mirror_only = True
    parameter_count_billions = 8.0
    model_size_label = "8B"
    allow_patterns = [
        "*.json",
        "*.md",
        "*.model",
        "*.safetensors",
        "*.txt",
        "LICENSE",
        "USE_POLICY.md",
    ]
    ignore_patterns = [
        "*.bin",
        "*.pth",
        "original/*",
    ]

    def _repo_cache_path(self) -> Path:
        cache_root = (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or str(Path.home() / ".cache" / "huggingface" / "hub")
        )
        return Path(cache_root) / f"models--{self.resolved_repo_id.replace('/', '--')}"

    def _clear_partial_artifacts(self) -> None:
        target = self.target_path()
        if target.exists():
            remove_path(target)
        repo_cache = self._repo_cache_path()
        if repo_cache.exists():
            shutil.rmtree(repo_cache)

    def _token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def validate_repo(self) -> dict[str, object]:
        from huggingface_hub import HfApi

        api_kwargs: dict[str, object] = {}
        if "endpoint" in inspect.signature(HfApi).parameters:
            api_kwargs["endpoint"] = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        api = HfApi(**api_kwargs)
        info = api.model_info(self.resolved_repo_id, token=self._token())
        tags = list(info.tags or [])
        has_transformers = info.library_name == "transformers" or "transformers" in tags
        if not has_transformers:
            raise ValueError(f"{self.resolved_repo_id} is not a Transformers-compatible repository.")
        return {
            "library_name": info.library_name,
            "pipeline_tag": info.pipeline_tag,
            "tags": tags,
            "sha": info.sha,
        }

    def perform_download(self, force: bool) -> dict[str, object]:
        from huggingface_hub import snapshot_download

        max_retries = 5
        base_delay = 10
        repo_info = None
        last_error = None

        for attempt in range(max_retries):
            try:
                self._clear_partial_artifacts()
                repo_info = self.validate_repo()

                kwargs: dict[str, object] = {
                    "repo_id": self.resolved_repo_id,
                    "local_dir": str(self.target_path()),
                    "allow_patterns": self.allow_patterns,
                    "ignore_patterns": self.ignore_patterns,
                    "token": self._token(),
                }
                signature = inspect.signature(snapshot_download).parameters
                if "endpoint" in signature:
                    kwargs["endpoint"] = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
                if "force_download" in signature:
                    kwargs["force_download"] = True
                if "resume_download" in signature:
                    kwargs["resume_download"] = False
                if "local_dir_use_symlinks" in signature:
                    kwargs["local_dir_use_symlinks"] = False

                snapshot_download(**kwargs)
                break
            except Exception as exc:
                last_error = exc
                self._clear_partial_artifacts()
                if attempt >= max_retries - 1:
                    raise RuntimeError(
                        f"Failed to download {self.resolved_repo_id} from the public community mirror after "
                        f"{max_retries} attempts. Last error: {last_error}"
                    ) from last_error
                delay = base_delay * (2 ** attempt)
                print(
                    f"Model download | retrying {self.resolved_repo_id} after a clean cache reset "
                    f"(attempt {attempt + 1}/{max_retries}, wait {delay}s): {exc}"
                )
                time.sleep(delay)

        return {
            "message": "Model snapshot downloaded from a public Hugging Face community mirror.",
            "metadata": {
                "source_type": "huggingface_snapshot",
                "repo_validation": repo_info,
                "public_community_mirror": True,
                "official_meta_gated_repo": "meta-llama/Meta-Llama-3-8B",
                "cache_reset_before_download": True,
                "weights_policy": "safetensors_only",
            },
        }
