#!/usr/bin/env python3
"""Build E9 federated train partitions and a partition manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from round23_federated_utils import (
    E9_IMBALANCE_WEIGHTS_8,
    default_partition_output_dir,
    load_config_with_inherits,
    repo_relative_str,
    resolve_dataset_train_path,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_platform.data.loaders import load_samples  # noqa: E402


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SUPPORTED_SPLIT_MODES = ("uniform", "noniid", "imbalance_noniid")


def _stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def _load_train_records(*, dataset_name: str, train_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    samples = load_samples(
        train_path,
        dataset_name=dataset_name,
        source="private_train",
        task_type="raw_text",
        round_id=0,
        client_id="single_node",
        prefix=f"e9_{dataset_name}",
        limit=limit,
    )
    if not samples:
        raise ValueError(f"No train samples were loaded from {train_path}")
    return [
        {
            "sample_id": sample.sample_id,
            "text": sample.render_text().strip(),
            "meta": dict(sample.meta),
        }
        for sample in samples
        if sample.render_text().strip()
    ]


def _compute_length_thresholds(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    lengths = sorted(max(1, len(record["text"].split())) for record in records)
    if not lengths:
        return (32, 96, 192)

    def q(fraction: float) -> int:
        index = min(len(lengths) - 1, max(0, int((len(lengths) - 1) * fraction)))
        return max(1, lengths[index])

    q1 = q(0.25)
    q2 = max(q1 + 1, q(0.50))
    q3 = max(q2 + 1, q(0.75))
    return (q1, q2, q3)


def _derive_source_bucket(record: dict[str, Any]) -> str:
    meta = dict(record.get("meta", {}))
    bucket = str(meta.get("bucket_id", "")).strip().lower()
    if bucket and not bucket.isdigit():
        return bucket
    tokens = [token.lower() for token in TOKEN_RE.findall(record["text"]) if len(token) >= 4]
    if not tokens:
        tokens = [token.lower() for token in TOKEN_RE.findall(record["text"])]
    if not tokens:
        return "misc"
    return max(tokens[:10], key=lambda token: (len(token), token))


def _length_bucket(word_count: int, thresholds: tuple[int, int, int]) -> str:
    q1, q2, q3 = thresholds
    if word_count <= q1:
        return "len_q1"
    if word_count <= q2:
        return "len_q2"
    if word_count <= q3:
        return "len_q3"
    return "len_q4"


def _annotate_records(records: list[dict[str, Any]], *, num_clients: int) -> list[dict[str, Any]]:
    thresholds = _compute_length_thresholds(records)
    annotated: list[dict[str, Any]] = []
    for record in records:
        word_count = max(1, len(record["text"].split()))
        source_bucket = _derive_source_bucket(record)
        length_band = _length_bucket(word_count, thresholds)
        lexical_band = _stable_hash(source_bucket) % max(num_clients * 3, 8)
        cluster_id = f"{length_band}:g{lexical_band:02d}:{source_bucket}"
        enriched = dict(record)
        enriched["word_count"] = word_count
        enriched["source_bucket"] = source_bucket
        enriched["length_bucket"] = length_band
        enriched["cluster_id"] = cluster_id
        annotated.append(enriched)
    return annotated


def _allocate_target_sizes(total: int, num_clients: int, weights: list[float] | None = None) -> list[int]:
    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    raw = [total * weight for weight in weights]
    sizes = [int(value) for value in raw]
    remainder = total - sum(sizes)
    ranking = sorted(range(num_clients), key=lambda idx: (raw[idx] - sizes[idx], -idx), reverse=True)
    for idx in ranking[:remainder]:
        sizes[idx] += 1
    return sizes


def _uniform_partition(records: list[dict[str, Any]], *, num_clients: int, seed: int) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    target_sizes = _allocate_target_sizes(len(shuffled), num_clients)
    assignments: list[list[dict[str, Any]]] = []
    cursor = 0
    for size in target_sizes:
        assignments.append(shuffled[cursor: cursor + size])
        cursor += size
    return assignments


def _group_records_for_noniid(
    records: list[dict[str, Any]],
    *,
    num_clients: int,
    seed: int,
    imbalance_weights: list[float] | None,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    annotated = _annotate_records(records, num_clients=num_clients)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in annotated:
        grouped[str(record["cluster_id"])].append(record)

    target_sizes = _allocate_target_sizes(len(annotated), num_clients, imbalance_weights)
    ordered_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    loads = [0] * num_clients

    for group_key, group_records in ordered_groups:
        remaining = list(group_records)
        preferred_start = _stable_hash(f"{group_key}:{seed}") % num_clients
        preferred_order = [(preferred_start + offset) % num_clients for offset in range(num_clients)]
        while remaining:
            candidates = [idx for idx in preferred_order if loads[idx] < target_sizes[idx]]
            if not candidates:
                candidates = sorted(range(num_clients), key=lambda idx: (loads[idx], idx))
            client_index = max(
                candidates,
                key=lambda idx: (target_sizes[idx] - loads[idx], -preferred_order.index(idx) if idx in preferred_order else 0),
            )
            remaining_capacity = max(0, target_sizes[client_index] - loads[client_index])
            if remaining_capacity == 0:
                remaining_capacity = len(remaining)
            take = min(len(remaining), remaining_capacity)
            assignments[client_index].extend(remaining[:take])
            loads[client_index] += take
            remaining = remaining[take:]
    return assignments, target_sizes


def _build_manifest(
    *,
    dataset_name: str,
    train_path: Path,
    output_dir: Path,
    split_mode: str,
    seed: int,
    num_clients: int,
    client_target_sizes: list[int],
    client_records: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    total = sum(len(items) for items in client_records)
    imbalance_mode = "fixed_tail_v1" if split_mode == "imbalance_noniid" else "none"
    split_notes = {
        "uniform": "random shuffle with near-even client sizes",
        "noniid": "new E9 text-shape grouping by length bucket and lexical/source signature",
        "imbalance_noniid": "new E9 text-shape grouping plus fixed 8-client imbalance template",
    }
    manifest_clients: list[dict[str, Any]] = []
    for client_index, records in enumerate(client_records):
        client_id = f"client_{client_index:03d}"
        client_path = output_dir / f"{client_id}_train.json"
        write_json(client_path, [record["text"] for record in records])
        length_counts = Counter(str(record.get("length_bucket", "")) for record in records)
        cluster_counts = Counter(str(record.get("cluster_id", "")) for record in records)
        source_counts = Counter(str(record.get("source_bucket", "")) for record in records)
        word_counts = [int(record.get("word_count", 0)) for record in records]
        manifest_clients.append(
            {
                "client_id": client_id,
                "train_path": repo_relative_str(client_path),
                "train_count": len(records),
                "target_count": client_target_sizes[client_index],
                "assigned_fraction": round((len(records) / total), 6) if total else 0.0,
                "dominant_length_bucket": length_counts.most_common(1)[0][0] if length_counts else "",
                "dominant_cluster": cluster_counts.most_common(1)[0][0] if cluster_counts else "",
                "top_source_buckets": [name for name, _ in source_counts.most_common(3)],
                "avg_words": round(sum(word_counts) / len(word_counts), 3) if word_counts else 0.0,
                "min_words": min(word_counts) if word_counts else 0,
                "max_words": max(word_counts) if word_counts else 0,
            }
        )

    return {
        "schema_version": "e9_partition_manifest_v1",
        "dataset_name": dataset_name,
        "source_train_path": repo_relative_str(train_path),
        "seed": seed,
        "num_clients": num_clients,
        "split_mode": split_mode,
        "imbalance_mode": imbalance_mode,
        "partitioner": "build_e9_federated_partitions.py",
        "split_note": split_notes[split_mode],
        "client_size_targets": client_target_sizes,
        "output_dir": repo_relative_str(output_dir),
        "total_train_count": total,
        "clients": manifest_clients,
    }


def build_partition_artifact(
    *,
    dataset_name: str,
    train_path: str | Path,
    num_clients: int,
    split_mode: str,
    seed: int,
    output_dir: str | Path,
    train_limit: int | None = None,
) -> dict[str, Any]:
    if split_mode not in SUPPORTED_SPLIT_MODES:
        raise ValueError(f"Unsupported split_mode '{split_mode}'. Expected one of {SUPPORTED_SPLIT_MODES}.")
    resolved_train_path = Path(train_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    records = _load_train_records(dataset_name=dataset_name, train_path=resolved_train_path, limit=train_limit)

    if split_mode == "uniform":
        client_records = _uniform_partition(records, num_clients=num_clients, seed=seed)
        client_target_sizes = [len(items) for items in client_records]
    elif split_mode == "noniid":
        client_records, client_target_sizes = _group_records_for_noniid(
            records,
            num_clients=num_clients,
            seed=seed,
            imbalance_weights=None,
        )
    else:
        if num_clients != len(E9_IMBALANCE_WEIGHTS_8):
            raise ValueError("imbalance_noniid currently requires num_clients=8")
        client_records, client_target_sizes = _group_records_for_noniid(
            records,
            num_clients=num_clients,
            seed=seed,
            imbalance_weights=list(E9_IMBALANCE_WEIGHTS_8),
        )

    manifest = _build_manifest(
        dataset_name=dataset_name,
        train_path=resolved_train_path,
        output_dir=resolved_output_dir,
        split_mode=split_mode,
        seed=seed,
        num_clients=num_clients,
        client_target_sizes=client_target_sizes,
        client_records=client_records,
    )
    write_json(resolved_output_dir / "partition_manifest.json", manifest)
    return manifest


def _resolve_build_request(args: argparse.Namespace) -> tuple[str, Path]:
    if args.config:
        return resolve_dataset_train_path(args.config)
    if not args.dataset_name or not args.train_path:
        raise ValueError("Either --config or both --dataset-name and --train-path must be provided.")
    return str(args.dataset_name), Path(args.train_path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build E9 federated client train partitions")
    parser.add_argument("--config", help="Optional selector config path used to resolve dataset_name/train_path")
    parser.add_argument("--dataset-name", help="Dataset name when not using --config")
    parser.add_argument("--train-path", help="Train json path when not using --config")
    parser.add_argument("--federated-setting", required=True, help="Setting name, e.g. e9_f4_uniform_all6_repeat5")
    parser.add_argument("--num-clients", required=True, type=int)
    parser.add_argument("--split-mode", required=True, choices=SUPPORTED_SPLIT_MODES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-limit", type=int, help="Optional limit for smoke validation")
    parser.add_argument("--output-dir", help="Optional explicit output directory")
    args = parser.parse_args()

    dataset_name, train_path = _resolve_build_request(args)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_partition_output_dir(
            federated_setting=str(args.federated_setting),
            dataset_name=dataset_name,
            seed=int(args.seed),
        )
    )
    manifest = build_partition_artifact(
        dataset_name=dataset_name,
        train_path=train_path,
        num_clients=int(args.num_clients),
        split_mode=str(args.split_mode),
        seed=int(args.seed),
        output_dir=output_dir,
        train_limit=args.train_limit,
    )
    print(json.dumps({"manifest_path": str(output_dir / "partition_manifest.json"), "clients": manifest["clients"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
