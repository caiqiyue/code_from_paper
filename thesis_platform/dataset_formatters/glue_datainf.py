from __future__ import annotations

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class DataInfGlueFormatter(BaseDatasetFormatter):
    """Create the GLUE subset size used in DataInf classification experiments."""

    name = "glue_datainf"
    train_cap = 4500
    validation_cap = 500
    sample_seed = 0

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        from datasets import DatasetDict, load_from_disk
        import numpy as np

        raw_path = downloader.raw_path()
        if raw_path is None:
            raise ValueError(f"{downloader.name} requires raw Hugging Face artifacts before formatting.")
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        if target is None:
            raise ValueError("glue_datainf formatter requires a formatted path.")
        target.mkdir(parents=True, exist_ok=True)

        raw_dataset = load_from_disk(str(raw_path))
        train_dataset = raw_dataset["train"]
        validation_dataset = raw_dataset["validation"]
        train_original = len(train_dataset)
        validation_original = len(validation_dataset)

        rng = np.random.default_rng(self.sample_seed)
        if train_original > self.train_cap:
            selected = sorted(rng.choice(train_original, self.train_cap, replace=False).tolist())
            train_dataset = train_dataset.select(selected)
        if validation_original > self.validation_cap:
            selected = sorted(rng.choice(validation_original, self.validation_cap, replace=False).tolist())
            validation_dataset = validation_dataset.select(selected)

        formatted = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        formatted.save_to_disk(str(target))
        return {
            "message": "Created the train/validation GLUE subset used by DataInf classification experiments.",
            "metadata": {
                "formatted_format": "huggingface_save_to_disk",
                "sampling_seed": self.sample_seed,
                "paper_alignment": {
                    "paper": "DataInf",
                    "experiment": "classification noisy-label detection",
                },
                "split_caps": {
                    "train": self.train_cap,
                    "validation": self.validation_cap,
                },
                "raw_split_sizes": {
                    "train": train_original,
                    "validation": validation_original,
                },
                "split_sizes": {
                    "train": len(train_dataset),
                    "validation": len(validation_dataset),
                },
                "provenance_note": "Formatted artifacts mirror DataInf's practice of capping large GLUE train/validation splits to 4500 and 500 examples.",
            },
        }
