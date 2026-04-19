from __future__ import annotations

from typing import Any

from pretext_platform.core.types import StageSummary


def make_round_privacy_record(
    *,
    round_id: int,
    client_summaries: dict[str, StageSummary],
    merged_surviving_count: int,
    server_stage2_sample_count: int,
) -> dict[str, Any]:
    """Build one experiment-level privacy record from per-client Stage1 summaries."""

    client_stage1_stats = {}
    for client_id, summary in client_summaries.items():
        client_stage1_stats[client_id] = {
            "epsilon": summary.metrics.get("epsilon"),
            "delta": summary.metrics.get("delta"),
            "rounds": summary.metrics.get("rounds"),
            "surviving_count": summary.metrics.get("surviving_count"),
        }
    return {
        "round_id": round_id,
        "participating_clients": sorted(client_summaries.keys()),
        "client_stage1_stats": client_stage1_stats,
        "merged_surviving_count": merged_surviving_count,
        "server_stage2_sample_count": server_stage2_sample_count,
    }


def build_privacy_summary(round_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize experiment-level privacy ledger information."""

    return {
        "round_count": len(round_records),
        "total_participations": sum(len(record["participating_clients"]) for record in round_records),
        "total_merged_surviving_count": sum(int(record["merged_surviving_count"]) for record in round_records),
        "total_server_stage2_sample_count": sum(int(record["server_stage2_sample_count"]) for record in round_records),
    }
