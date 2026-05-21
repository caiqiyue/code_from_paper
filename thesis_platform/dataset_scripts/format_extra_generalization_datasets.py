"""Format extra held-out datasets into PrE-Text compatible JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return " ".join(text.split())


def _load_json_texts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        texts = payload
    elif isinstance(payload, dict):
        texts = []
        for value in payload.values():
            if isinstance(value, list):
                texts.extend(value)
            else:
                texts.append(value)
    else:
        raise TypeError(f"Unsupported JSON payload in {path}: {type(payload).__name__}")
    return [text for text in (_clean_text(value) for value in texts) if text]


def _load_hf_split_texts(split_dir: Path) -> list[str]:
    try:
        from datasets import load_from_disk  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The datasets package is required to convert HuggingFace arrow datasets.") from exc
    dataset = load_from_disk(str(split_dir))
    candidate_columns = ("text", "sentence", "inputs", "question")
    text_column = next((name for name in candidate_columns if name in dataset.column_names), None)
    if text_column is None:
        raise KeyError(f"No supported text column found in {split_dir}. Columns: {dataset.column_names}")
    return [text for text in (_clean_text(row[text_column]) for row in dataset) if text]


def write_pretext_json_dataset(
    *,
    dataset_root: str | Path,
    dataset_name: str,
    train_texts: Iterable[str],
    eval_texts: Iterable[str],
    source_note: str,
) -> dict[str, Any]:
    root = Path(dataset_root)
    formatted = root / "formatted"
    formatted.mkdir(parents=True, exist_ok=True)
    train = [text for text in (_clean_text(value) for value in train_texts) if text]
    eval_rows = [text for text in (_clean_text(value) for value in eval_texts) if text]
    if not train or not eval_rows:
        raise ValueError(f"{dataset_name} requires non-empty train and eval texts.")

    train_path = formatted / f"{dataset_name}_train.json"
    eval_path = formatted / f"{dataset_name}_eval.json"
    train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps({"1": eval_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "name": dataset_name,
        "description": f"{dataset_name} formatted for PrE-Text held-out generalization experiments",
        "dataset_root": f"datasets/{dataset_name}",
        "raw_path": f"datasets/{dataset_name}/raw" if (root / "raw").exists() else None,
        "formatted_path": f"datasets/{dataset_name}/formatted",
        "formatter_name": "pretext_json",
        "required_paths": [
            f"datasets/{dataset_name}/formatted/{dataset_name}_train.json",
            f"datasets/{dataset_name}/formatted/{dataset_name}_eval.json",
        ],
        "formatted_format": "json",
        "split_sizes": {"train": len(train), "eval": len(eval_rows)},
        "total_records": len(train) + len(eval_rows),
        "provenance_note": source_note,
        "held_out_generalization_candidate": True,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset_name": dataset_name, "train_count": len(train), "eval_count": len(eval_rows)}


def format_existing_dataset(dataset_root: Path, dataset_name: str) -> dict[str, Any]:
    formatted = dataset_root / "formatted"
    if dataset_name == "bioarxiv":
        train = _load_json_texts(formatted / "bioarxiv_train.json")
        eval_rows = _load_json_texts(formatted / "bioarxiv_eval.json")
        source_note = "Existing bioRxiv formatted texts normalized to the shared PrE-Text JSON contract."
    elif dataset_name == "rotten_tomatoes":
        train = _load_json_texts(formatted / "rotten_tomatoes_train.json")
        eval_rows = _load_json_texts(formatted / "rotten_tomatoes_eval.json")
        source_note = "Existing Rotten Tomatoes formatted texts normalized to the shared PrE-Text JSON contract."
    elif dataset_name == "twitter_emotion_binary":
        train = _load_hf_split_texts(formatted / "train")
        eval_rows = _load_hf_split_texts(formatted / "validation")
        source_note = "Twitter emotion binary HuggingFace artifacts exported to the shared PrE-Text JSON contract."
    else:
        raise ValueError(f"Unsupported extra generalization dataset: {dataset_name}")
    return write_pretext_json_dataset(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        train_texts=train,
        eval_texts=eval_rows,
        source_note=source_note,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format extra held-out datasets for PrE-Text experiments.")
    parser.add_argument("--datasets-root", default=str(Path(__file__).resolve().parents[1] / "datasets"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bioarxiv", "rotten_tomatoes", "twitter_emotion_binary"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.datasets_root)
    reports = [format_existing_dataset(root / name, name) for name in args.datasets]
    print(json.dumps({"formatted": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
