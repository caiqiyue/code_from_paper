from __future__ import annotations

import os
import time

from .base import BaseModelDownloader
from .common import remove_path

# HuggingFace token for authenticated downloads (improves stability, avoids rate limits)
# Set via HF_TOKEN environment variable - will be used automatically by huggingface_hub
HF_TOKEN = os.environ.get("HF_TOKEN", None)


class HuggingFaceModelDownloader(BaseModelDownloader):
    """Download a model snapshot from Hugging Face Hub."""

    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None

    def validate_repo(self) -> dict[str, object]:
        """Validate the selected repository before downloading it."""

        from huggingface_hub import model_info

        info = model_info(self.resolved_repo_id, token=HF_TOKEN)
        tags = list(info.tags or [])
        has_transformers = info.library_name == "transformers" or "transformers" in tags
        if self.community_mirror_only and not has_transformers:
            raise ValueError(
                f"{self.resolved_repo_id} is not a Transformers-compatible Hugging Face repository and cannot be used here."
            )
        return {
            "library_name": info.library_name,
            "pipeline_tag": info.pipeline_tag,
            "tags": tags,
            "sha": info.sha,
        }

    def perform_download(self, force: bool) -> dict[str, object]:
        """Download a Hugging Face repository into the shared model directory with retry logic."""

        from huggingface_hub import snapshot_download

        target = self.target_path()
        if target.exists():
            remove_path(target)

        max_retries = 5
        base_delay = 5
        repo_info = None
        last_error = None

        for attempt in range(max_retries):
            try:
                repo_info = self.validate_repo()
                snapshot_download(
                    repo_id=self.resolved_repo_id,
                    local_dir=str(target),
                    allow_patterns=self.allow_patterns,
                    ignore_patterns=self.ignore_patterns,
                    token=HF_TOKEN,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Model download | network error for {self.resolved_repo_id} (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {exc}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Failed to download model {self.resolved_repo_id} after {max_retries} attempts. Last error: {last_error}"
                    ) from last_error

        return {
            "message": "Model snapshot downloaded from Hugging Face Hub.",
            "metadata": {"source_type": "huggingface_snapshot", "repo_validation": repo_info},
        }
