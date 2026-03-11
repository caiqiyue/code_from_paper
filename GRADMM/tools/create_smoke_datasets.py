import argparse
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path


DATASET_LAYOUT = {
    "imdb": {
        "train": "imdb/train_len256.jsonl",
        "validation": "imdb/validation_len256.jsonl",
    },
    "rtpolarity": {
        "train": "rtpolarity/train.jsonl",
        "validation": "rtpolarity/validation.jsonl",
    },
}


def resolve_path(path_value, base_dir):
    """Resolve a path relative to this script when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_jsonl(path):
    """Load a JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    """Write a list of dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_sample(rows, sample_size, seed):
    """Draw a balanced sample while preserving the original row schema."""
    if sample_size >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["label"])].append(row)

    labels = sorted(grouped.keys())
    if not labels:
        return []

    base_target = sample_size // len(labels)
    remainder = sample_size % len(labels)
    sampled = []
    leftovers = []

    for idx, label in enumerate(labels):
        group = list(grouped[label])
        rng.shuffle(group)
        take = min(len(group), base_target + (1 if idx < remainder else 0))
        sampled.extend(group[:take])
        leftovers.extend(group[take:])

    if len(sampled) < sample_size:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: sample_size - len(sampled)])

    rng.shuffle(sampled)
    return sampled[:sample_size]


def build_smoke_datasets(source_root, output_root, sample_size, seed):
    """Create tiny local dataset copies for smoke testing."""
    manifest = {}
    for dataset_name, split_map in DATASET_LAYOUT.items():
        manifest[dataset_name] = {}
        for split_name, relative_path in split_map.items():
            src_path = source_root / relative_path
            out_path = output_root / relative_path
            rows = read_jsonl(src_path)
            sampled_rows = stratified_sample(rows, sample_size, seed)
            write_jsonl(out_path, sampled_rows)
            manifest[dataset_name][split_name] = {
                "source": str(src_path),
                "output": str(out_path),
                "rows": len(sampled_rows),
                "labels": sorted({int(row["label"]) for row in sampled_rows}),
                "keys": sorted(sampled_rows[0].keys()) if sampled_rows else [],
            }
    return manifest


def run_self_test():
    """Validate sampling, schema preservation, and output creation on fixtures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        source_root = tmp_root / "data"
        output_root = tmp_root / "data_smoke"

        fixtures = {
            source_root / "imdb" / "train_len256.jsonl": [
                {"text": f"imdb train bad {idx}", "label": 0} for idx in range(30)
            ]
            + [{"text": f"imdb train great {idx}", "label": 1} for idx in range(30)],
            source_root / "imdb" / "validation_len256.jsonl": [
                {"text": f"imdb valid bad {idx}", "label": 0} for idx in range(30)
            ]
            + [{"text": f"imdb valid great {idx}", "label": 1} for idx in range(30)],
            source_root / "rtpolarity" / "train.jsonl": [
                {"id": idx, "inputs": f"rt train bad {idx}", "label": 0}
                for idx in range(30)
            ]
            + [
                {"id": idx + 100, "inputs": f"rt train great {idx}", "label": 1}
                for idx in range(30)
            ],
            source_root / "rtpolarity" / "validation.jsonl": [
                {"id": idx, "inputs": f"rt valid bad {idx}", "label": 0}
                for idx in range(30)
            ]
            + [
                {"id": idx + 100, "inputs": f"rt valid great {idx}", "label": 1}
                for idx in range(30)
            ],
        }

        for path, rows in fixtures.items():
            write_jsonl(path, rows)

        manifest = build_smoke_datasets(
            source_root=source_root,
            output_root=output_root,
            sample_size=20,
            seed=42,
        )

        assert manifest["imdb"]["train"]["rows"] == 20
        assert manifest["rtpolarity"]["validation"]["rows"] == 20
        assert manifest["imdb"]["train"]["labels"] == [0, 1]
        assert manifest["rtpolarity"]["train"]["labels"] == [0, 1]
        assert manifest["imdb"]["train"]["keys"] == ["label", "text"]
        assert manifest["rtpolarity"]["train"]["keys"] == ["id", "inputs", "label"]

        for split_map in DATASET_LAYOUT.values():
            for relative_path in split_map.values():
                assert (output_root / relative_path).exists()

    print("self-test passed")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Create tiny local smoke datasets.")
    parser.add_argument("--source-root", default="../data")
    parser.add_argument("--output-root", default="../data_smoke")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    """CLI entrypoint."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    base_dir = Path(__file__).resolve().parent
    source_root = resolve_path(args.source_root, base_dir)
    output_root = resolve_path(args.output_root, base_dir)
    manifest = build_smoke_datasets(
        source_root=source_root,
        output_root=output_root,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
