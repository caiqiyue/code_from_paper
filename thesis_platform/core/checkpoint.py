"""Checkpoint management for experiment resumption.

Supports:
- Saving experiment state after each round
- Resuming from checkpoints
- Automatic cleanup of old checkpoints
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def serialize_server_context(server_ctx: Any) -> dict[str, Any]:
    """Serialize ServerContext to a JSON-serializable dictionary.

    Only serializes essential fields that can be safely saved/restored.
    Excludes non-serializable fields like text_backend.
    """
    return {
        "experiment_id": getattr(server_ctx, "experiment_id", ""),
        "prompt_text": getattr(server_ctx, "prompt_text", ""),
        "prompt_history": list(getattr(server_ctx, "prompt_history", [])),
        "base_prompt": getattr(server_ctx, "base_prompt", None),
        "cluster_prompts": dict(getattr(server_ctx, "cluster_prompts", {})),
        "aggregation_memory": dict(getattr(server_ctx, "aggregation_memory", {})),
        "generated_history": [
            list(batch) for batch in getattr(server_ctx, "generated_history", [])
        ],
    }


class CheckpointManager:
    """Manage experiment checkpoints for fault tolerance."""

    def __init__(
        self,
        output_dir: Path,
        max_checkpoints: int = 3,
        save_artifacts: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.max_checkpoints = max_checkpoints
        self.save_artifacts = save_artifacts

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        round_id: int,
        experiment_state: Dict[str, Any],
        server_ctx: Any,
        privacy_ledger: Any,
        config: Dict[str, Any],
    ) -> Path:
        """Save a checkpoint after completing a round.

        Args:
            round_id: The completed round number
            experiment_state: Dictionary containing experiment state
            server_ctx: Server context object
            privacy_ledger: Privacy ledger for accounting
            config: Experiment configuration

        Returns:
            Path to the saved checkpoint
        """
        checkpoint_id = f"checkpoint_round_{round_id:03d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)

        # Prepare checkpoint data
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "round_id": round_id,
            "timestamp": datetime.now().isoformat(),
            "experiment_state": experiment_state,
            "config": config,
        }

        # Save checkpoint metadata
        with open(checkpoint_path / "checkpoint.json", "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save server context (serialized)
        try:
            server_ctx_data = serialize_server_context(server_ctx)
            with open(checkpoint_path / "server_ctx.json", "w") as f:
                json.dump(server_ctx_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not serialize server_ctx: {e}")

        # Save privacy ledger
        if privacy_ledger:
            try:
                with open(checkpoint_path / "privacy_ledger.json", "w") as f:
                    json.dump(privacy_ledger.report(), f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save privacy ledger: {e}")

        # Copy latest round artifacts if requested
        if self.save_artifacts:
            round_dir = self.output_dir / f"round_{round_id:03d}"
            if round_dir.exists():
                artifacts_dir = checkpoint_path / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                for artifact in round_dir.glob("*"):
                    if artifact.is_file():
                        try:
                            shutil.copy2(artifact, artifacts_dir / artifact.name)
                        except Exception as e:
                            logger.debug(f"Could not copy artifact {artifact}: {e}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load a checkpoint for resumption.

        Args:
            checkpoint_id: Specific checkpoint to load, or None for latest

        Returns:
            Checkpoint data dictionary, or None if no checkpoint found
        """
        if checkpoint_id:
            checkpoint_path = self.checkpoint_dir / checkpoint_id
        else:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_round_*"))
            if not checkpoints:
                return None
            checkpoint_path = checkpoints[-1]

        if not checkpoint_path.exists():
            return None

        # Load checkpoint metadata
        try:
            with open(checkpoint_path / "checkpoint.json", "r") as f:
                checkpoint_data = json.load(f)
        except Exception as e:
            logger.error(f"Could not load checkpoint metadata: {e}")
            return None

        # Load server context
        server_ctx_path = checkpoint_path / "server_ctx.json"
        if server_ctx_path.exists():
            try:
                with open(server_ctx_path, "r") as f:
                    checkpoint_data["server_ctx_data"] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load server_ctx: {e}")

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data

    def get_latest_round(self) -> int:
        """Get the latest completed round from checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_round_*"))
        if not checkpoints:
            return -1

        latest = checkpoints[-1]
        try:
            round_num = int(latest.name.split("_")[-1])
            return round_num
        except (ValueError, IndexError):
            return -1

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints most recent."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_round_*"))

        if len(checkpoints) > self.max_checkpoints:
            to_remove = checkpoints[:-self.max_checkpoints]
            for checkpoint in to_remove:
                try:
                    shutil.rmtree(checkpoint)
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {checkpoint}: {e}")

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_round_*"))
        return [
            {
                "id": c.name,
                "path": str(c),
                "round": int(c.name.split("_")[-1]),
            }
            for c in checkpoints
        ]
