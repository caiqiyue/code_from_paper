from __future__ import annotations

from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = "thesis_platform.runtime.v1"


def build_round_manifest(
    *,
    experiment_id: str,
    round_id: int,
    round_dir: Path,
) -> dict[str, Any]:
    """Return the stable manifest payload for one round directory."""

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "round_manifest",
        "experiment_id": experiment_id,
        "round_id": round_id,
        "round_dir": str(round_dir),
        "required_files": {
            "round_metrics": str(round_dir / "round_metrics.json"),
            "routing_summary": str(round_dir / "routing_summary.json"),
            "probe_metrics": str(round_dir / "probe_metrics.json"),
            "prompt_update": str(round_dir / "prompt_update.json"),
            "client_prototypes": str(round_dir / "client_prototypes.jsonl"),
            "cluster_prompts": str(round_dir / "cluster_prompts.json"),
        },
    }


def build_experiment_manifest(
    *,
    experiment_id: str,
    experiment_dir: Path,
    resolved_config_path: Path,
    metrics_summary_path: Path,
    privacy_ledger_path: Path,
    round_manifests: list[dict[str, Any]],
    downstream_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the stable manifest for the whole experiment output tree."""

    downstream_files = {}
    if downstream_summary:
        downstream_files = {
            "downstream_eval_summary": downstream_summary.get("summary_path"),
            "stage2_corpus": downstream_summary.get("synthetic_corpus_path"),
            "large_eval_summary": downstream_summary.get("stages", {}).get("large_eval", {}).get("summary_path"),
            "small_eval_summary": downstream_summary.get("stages", {}).get("small_eval", {}).get("summary_path"),
            "baseline_summaries": downstream_summary.get("baseline_summaries_path"),
        }
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "experiment_manifest",
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "required_files": {
            "resolved_config": str(resolved_config_path),
            "metrics_summary": str(metrics_summary_path),
            "privacy_ledger": str(privacy_ledger_path),
        },
        "rounds": round_manifests,
        "downstream_eval": downstream_files,
    }
