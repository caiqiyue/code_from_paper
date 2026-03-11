from __future__ import annotations

from .base import BaseModelDownloader
from .common import remove_path


class HuggingFaceModelDownloader(BaseModelDownloader):
    """Download a model snapshot from Hugging Face Hub."""

    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None

    def validate_repo(self) -> dict[str, object]:
        """Validate the selected repository before downloading it."""

        from huggingface_hub import model_info

        info = model_info(self.resolved_repo_id)
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
        """Download a Hugging Face repository into the shared model directory."""

        from huggingface_hub import snapshot_download

        target = self.target_path()
        if target.exists():
            remove_path(target)
        repo_info = self.validate_repo()
        snapshot_download(
            repo_id=self.resolved_repo_id,
            local_dir=str(target),
            allow_patterns=self.allow_patterns,
            ignore_patterns=self.ignore_patterns,
        )
        return {
            "message": "Model snapshot downloaded from Hugging Face Hub.",
            "metadata": {"source_type": "huggingface_snapshot", "repo_validation": repo_info},
        }
