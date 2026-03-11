from __future__ import annotations


def build_glue_train_validation_dataset(subset: str):
    """Load only the GLUE train and validation splits used in thesis experiments."""

    from datasets import DatasetDict, load_dataset

    train_dataset, validation_dataset = load_dataset("glue", subset, split=["train", "validation"])
    return DatasetDict({"train": train_dataset, "validation": validation_dataset}), {
        "source_dataset": "glue",
        "subset": subset,
        "source_splits": ["train", "validation"],
        "provenance_note": "Raw artifacts keep only the train and validation splits consumed by the downstream paper experiments.",
    }
