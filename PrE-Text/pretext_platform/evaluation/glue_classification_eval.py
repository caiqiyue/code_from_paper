"""GLUE downstream evaluation using synthetic-corpus adaptation plus task fine-tuning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.io_utils import ensure_dir
from pretext_platform.core.resource_cleanup import release_gpu_memory
from pretext_platform.core.types import ModelPaths, StageSummary


SUPPORTED_TASKS = {"sst2", "qqp", "qnli", "imdb", "rotten_tomatoes"}


def _load_training_texts(stage2_dir: Path) -> list[str]:
    with (stage2_dir / "llama7b_text_syn.json").open("r", encoding="utf-8") as handle:
        synthetic_outputs = json.load(handle)
    all_data = []
    for text in synthetic_outputs:
        split_samples = re.split("Orig", text)
        raw_sample = split_samples[0].strip().strip("\n")
        if len(raw_sample.split()) > 3:
            all_data.append(raw_sample.replace("\n\n", " ").replace("\n", " "))
    return all_data


def _local_split_path(task_name: str, dataset_root: Path, split: str) -> Path:
    if task_name in {"sst2", "qqp", "qnli"}:
        return dataset_root / f"glue_{task_name}" / "formatted"
    if task_name == "imdb":
        filename = "train_len256.jsonl" if split == "train" else "validation_len256.jsonl"
        return dataset_root / "imdb" / "formatted" / filename
    if task_name == "rotten_tomatoes":
        return dataset_root / "rotten_tomatoes" / "raw"
    raise ValueError(f"Unsupported task: {task_name}")


def _load_local_glue_split(task_name: str, dataset_root: Path, split: str):
    import datasets

    if task_name in {"sst2", "qqp", "qnli"}:
        dataset_dict = datasets.load_from_disk(str(_local_split_path(task_name, dataset_root, split)))
        return dataset_dict[split]

    if task_name == "imdb":
        path = _local_split_path(task_name, dataset_root, split)
        return datasets.load_dataset("json", data_files=str(path), split="train")

    if task_name == "rotten_tomatoes":
        dataset_dict = datasets.load_from_disk(str(_local_split_path(task_name, dataset_root, split)))
        mapped_split = "train" if split == "train" else "validation"
        return dataset_dict[mapped_split]

    raise FileNotFoundError(f"Local split for {task_name} is unavailable.")

def _load_glue_splits(task_name: str, dataset_root: Path):
    eval_split = "validation" if task_name in {"sst2", "qqp", "qnli"} else "validation"
    train_dataset = _load_local_glue_split(task_name, dataset_root, "train")
    eval_dataset = _load_local_glue_split(task_name, dataset_root, eval_split)
    return train_dataset, eval_dataset, "local"


def _text_pair_and_label(example: dict[str, Any], task_name: str) -> tuple[str, str | None, int]:
    if task_name == "sst2":
        return str(example["sentence"]), None, int(example["label"])
    if task_name == "qqp":
        return str(example["question1"]), str(example["question2"]), int(example["label"])
    if task_name == "qnli":
        return str(example["question"]), str(example["sentence"]), int(example["label"])
    if task_name == "imdb":
        return str(example["text"]), None, int(example["label"])
    if task_name == "rotten_tomatoes":
        return str(example["text"]), None, int(example["label"])
    raise ValueError(f"Unsupported task: {task_name}")


def evaluate_classification(model, eval_loader, device):
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total else 0.0
    return accuracy, correct, total


def _adapt_language_model(
    synthetic_texts: list[str],
    tokenizer,
    base_model_path: Path,
    output_dir: Path,
    *,
    cutoff_len: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> Path:
    import datasets
    import torch
    from datasets import Dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    adapted_dir = ensure_dir(output_dir / "adapted_lm")
    if not synthetic_texts or epochs <= 0:
        return base_model_path

    datasets.disable_progress_bars()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    optimizer = None
    loader = None
    tokenized = None
    dataset = None

    try:
        model = AutoModelForCausalLM.from_pretrained(str(base_model_path), local_files_only=True)
        model = model.to(device)

        def tokenize_synthetic(example):
            encoded = tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=cutoff_len,
            )
            labels = [
                -100 if token == tokenizer.pad_token_id else token
                for token in encoded["input_ids"]
            ]
            encoded["labels"] = labels
            return encoded

        dataset = Dataset.from_list([{"text": text} for text in synthetic_texts])
        tokenized = dataset.shuffle(seed=seed).map(tokenize_synthetic)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            for step, batch in enumerate(loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss / grad_accum_steps
                loss.backward()
                if step % grad_accum_steps == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

        model.save_pretrained(adapted_dir)
        tokenizer.save_pretrained(adapted_dir)
        return adapted_dir
    finally:
        model = None
        optimizer = None
        loader = None
        tokenized = None
        dataset = None
        release_gpu_memory()


def run_glue_classification_eval(
    config: ExperimentConfig,
    model_paths: ModelPaths,
    stage2_dir: Path,
    output_dir: Path,
    task_name: str = "sst2",
) -> StageSummary:
    import datasets
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task_name}")

    eval_cfg = config.eval_glue or {}
    output_dir = ensure_dir(output_dir)
    log_dir = ensure_dir(output_dir / f"glue_{task_name}_eval")
    datasets.disable_progress_bars()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    train_loader = None
    eval_loader = None
    optimizer = None
    train_tokenized = None
    eval_tokenized = None
    train_glue = None
    eval_glue = None

    try:
        base_model_path = model_paths.distilgpt2
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_glue, eval_glue, source = _load_glue_splits(task_name, config.dataset_root())
        synthetic_texts = _load_training_texts(stage2_dir)

        cutoff_len = int(eval_cfg.get("cutoff_len", 128))
        grad_accum_steps = int(eval_cfg.get("grad_accum_steps", 16))
        classifier_epochs = int(eval_cfg.get("epochs", 3))
        adaptation_epochs = int(eval_cfg.get("lm_epochs", 1))
        batch_size = int(eval_cfg.get("batch_size", 8))
        eval_batch_size = int(eval_cfg.get("eval_batch_size", 4))
        learning_rate = float(eval_cfg.get("learning_rate", 5e-5))
        num_synthetic_samples = int(eval_cfg.get("num_synthetic_samples", 10000))
        seed = int(config.meta.get("seed", 42))

        adapted_model_path = _adapt_language_model(
            synthetic_texts[:num_synthetic_samples],
            tokenizer,
            base_model_path,
            log_dir,
            cutoff_len=cutoff_len,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            epochs=adaptation_epochs,
            seed=seed,
        )

        classifier_tokenizer = AutoTokenizer.from_pretrained(str(adapted_model_path), local_files_only=True)
        if classifier_tokenizer.pad_token is None:
            classifier_tokenizer.pad_token = classifier_tokenizer.eos_token

        def tokenize_glue(example):
            text_a, text_b, label = _text_pair_and_label(example, task_name)
            encoded = classifier_tokenizer(
                text_a,
                text_b,
                truncation=True,
                padding="max_length",
                max_length=cutoff_len,
            )
            encoded["labels"] = label
            return encoded

        train_tokenized = train_glue.map(tokenize_glue)
        eval_tokenized = eval_glue.map(tokenize_glue)
        train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained(
            str(adapted_model_path),
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        model.config.pad_token_id = classifier_tokenizer.pad_token_id
        model = model.to(device)

        train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_tokenized, batch_size=eval_batch_size, shuffle=False, num_workers=0)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        best_accuracy = 0.0
        best_stats: dict[str, Any] | None = None

        for epoch in range(classifier_epochs):
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss / grad_accum_steps
                loss.backward()
                total_loss += loss.item()
                if step % grad_accum_steps == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            accuracy, correct, total = evaluate_classification(model, eval_loader, device)
            stats = {
                "epoch": epoch,
                "train_loss": total_loss / max(len(train_loader), 1),
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
            with (log_dir / f"epoch{epoch}_stats.json").open("w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2)

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_stats = stats
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "accuracy": accuracy,
                    },
                    log_dir / "best_model.pth",
                )

        if best_stats is not None:
            with (log_dir / "best_stats.json").open("w", encoding="utf-8") as handle:
                json.dump(best_stats, handle, indent=2)

        return StageSummary(
            stage_name=f"glue_eval_{task_name}",
            output_dir=output_dir,
            artifacts={
                "stats_dir": str(log_dir),
                "synthetic_train_count": min(len(synthetic_texts), num_synthetic_samples),
                "data_source": source,
                "adapted_model_path": str(adapted_model_path),
            },
            metrics={
                "task": task_name,
                "epochs": classifier_epochs,
                "lm_epochs": adaptation_epochs,
                "synthetic_train_count": min(len(synthetic_texts), num_synthetic_samples),
                "glue_train_count": len(train_tokenized),
                "eval_count": len(eval_tokenized),
                "best_accuracy": best_accuracy,
                "correct": best_stats.get("correct", 0) if best_stats else 0,
                "total": best_stats.get("total", 0) if best_stats else 0,
                "data_source": source,
            },
        )
    finally:
        model = None
        train_loader = None
        eval_loader = None
        optimizer = None
        train_tokenized = None
        eval_tokenized = None
        train_glue = None
        eval_glue = None
        release_gpu_memory()


def validate_local_glue_datasets(dataset_root: Path) -> dict[str, Any]:
    tasks = ["sst2", "qqp", "qnli", "imdb", "rotten_tomatoes"]
    result = {
        "tasks": {},
        "all_available": True,
        "all_required_available": True,
        "missing": [],
        "fallback_only": [],
        "warnings": [],
    }

    for task in tasks:
        if task in {"sst2", "qqp", "qnli"}:
            root = dataset_root / f"glue_{task}" / "formatted"
            available = (root / "dataset_dict.json").exists() and (root / "train").exists() and (root / "validation").exists()
            reason = "Found local DatasetDict" if available else f"Missing DatasetDict under {root}"
        elif task == "imdb":
            train_path = dataset_root / "imdb" / "formatted" / "train_len256.jsonl"
            val_path = dataset_root / "imdb" / "formatted" / "validation_len256.jsonl"
            available = train_path.exists() and val_path.exists()
            reason = "Found train/validation JSONL" if available else "Missing IMDB train/validation JSONL"
        else:
            root = dataset_root / "rotten_tomatoes" / "raw"
            available = (root / "dataset_dict.json").exists() and (root / "train").exists() and (root / "validation").exists()
            if available:
                reason = "Found local DatasetDict under rotten_tomatoes/raw"
            else:
                reason = f"Missing rotten_tomatoes raw DatasetDict under {root}"

        if not available:
            result["all_available"] = False
            result["all_required_available"] = False
            result["missing"].append(task)

        result["tasks"][task] = {"available": available, "reason": reason}

    return result


def print_glue_validation_report(dataset_root: Path) -> None:
    print("=" * 60)
    print("GLUE Dataset Local File Validation")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    validation = validate_local_glue_datasets(dataset_root)
    for task, info in validation["tasks"].items():
        status = "AVAILABLE" if info["available"] else "MISSING"
        print(f"{task:20s} {status:10s} {info['reason']}")
    if validation["all_available"]:
        print("All supported tasks are available locally.")
    else:
        if validation["missing"]:
            print(f"Missing local tasks: {', '.join(validation['missing'])}")
    print("=" * 60)
