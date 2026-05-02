from pathlib import Path

from dp_prompt.data.loader import load_document_dataset


def test_load_document_dataset_reads_jsonl_and_assigns_splits(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"text":"a","label":1,"author_id":"u1"}\n', encoding="utf-8")
    val.write_text('{"text":"b","label":0,"author_id":"u2"}\n', encoding="utf-8")
    test.write_text('{"text":"c","label":1,"author_id":"u3"}\n', encoding="utf-8")

    cfg = {
        "dataset": {
            "text_field": "text",
            "label_field": "label",
            "author_field": "author_id",
            "splits": {
                "train": str(train),
                "validation": str(val),
                "test": str(test),
            },
        }
    }

    bundle = load_document_dataset(cfg)

    assert set(bundle.dataframe["split"]) == {"train", "validation", "test"}
    assert list(bundle.dataframe.columns)[:4] == ["sample_id", "text", "label", "author_id"]
