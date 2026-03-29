"""GPT-2 XL (1.5B) downstream evaluation on synthetic data - Windows compatible.
Uses full fine-tuning without PEFT LoRA or Accelerator.
GPT-2 XL is distributed as a single-file model (non-sharded), avoiding Windows segfault issues.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.io_utils import ensure_dir
from pretext_platform.core.types import DatasetBundle, ModelPaths, StageSummary


def _load_training_texts(stage2_dir: Path) -> list[str]:
    """Load the synthetic corpus and keep only plausible training samples."""

    with (stage2_dir / "llama7b_text_syn.json").open("r", encoding="utf-8") as handle:
        synthetic_outputs = json.load(handle)
    all_data = []
    for text in synthetic_outputs:
        split_samples = re.split("Orig", text)
        raw_sample = split_samples[0].strip().strip("\n")
        if len(raw_sample.split(" ")) > 3:
            all_data.append(raw_sample.replace("\n\n", " ").replace("\n", " "))
    return all_data


def evaluate(model, eval_loader, device, xent_loss):
    """Compute loss and top-k token accuracy on the evaluation split."""

    import torch

    model.eval()
    total_loss = 0.0
    top_k_accuracies = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    total_evaluated_tokens = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            valid_mask = flat_labels != -100
            filtered_logits = flat_logits[valid_mask]
            filtered_labels = flat_labels[valid_mask]
            loss = xent_loss(filtered_logits, filtered_labels)
            total_loss += loss.item()

            _, top_k_indices = torch.topk(filtered_logits, k=max(top_k_accuracies.keys()), dim=-1)
            expanded_labels = filtered_labels.unsqueeze(1)
            correct_predictions = top_k_indices == expanded_labels
            for k in top_k_accuracies:
                top_k_accuracies[k] += correct_predictions[:, :k].sum().item()
            total_evaluated_tokens += valid_mask.sum().item()

    for k in top_k_accuracies:
        top_k_accuracies[k] = top_k_accuracies[k] / total_evaluated_tokens if total_evaluated_tokens > 0 else 0
    avg_loss = total_loss / total_evaluated_tokens if total_evaluated_tokens > 0 else 0.0
    return avg_loss, top_k_accuracies


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """Persist the current model and optimizer state to disk."""

    import torch
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def run_gpt2_xl_eval(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    model_paths: ModelPaths,
    stage2_dir: Path,
    output_dir: Path,
) -> StageSummary:
    """Fine-tune GPT-2 XL (1.5B) on synthetic data and evaluate next-token prediction.
    Windows compatible version - uses full fine-tuning without PEFT LoRA or Accelerator.
    GPT-2 XL is distributed as a single-file model, avoiding sharded safetensors segfault issues.
    Falls back to distilgpt2 if GPT-2 XL is not available locally and cannot be downloaded.
    """

    import datasets
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Clear CUDA cache before loading to avoid fragmentation issues on Windows
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    eval_cfg = config.eval_large
    output_dir = ensure_dir(output_dir)
    log_dir = ensure_dir(output_dir / "gpt2_xl_models_and_accuracies")

    # Use simple device assignment without Accelerator to avoid segfault issues on Windows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load GPT-2 XL from local path first, fall back to distilgpt2
    gpt2_xl_path = str(model_paths.gpt2_xl)
    distilgpt2_path = str(model_paths.distilgpt2)

    model_name = "GPT-2 XL (1.5B)"
    try:
        tokenizer = AutoTokenizer.from_pretrained(gpt2_xl_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(gpt2_xl_path, local_files_only=True)
    except Exception:
        # GPT-2 XL not available locally or cannot be downloaded, fall back to distilgpt2
        print(f"GPT-2 XL not available at {gpt2_xl_path}, falling back to distilgpt2")
        try:
            tokenizer = AutoTokenizer.from_pretrained(distilgpt2_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(distilgpt2_path, local_files_only=True)
            model_name = "DistilGPT-2 (82M)"
        except Exception as e:
            raise RuntimeError(f"Neither GPT-2 XL nor distilgpt2 could be loaded: {e}")

    model = model.to(device)

    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cutoff_len = int(eval_cfg.get("cutoff_len", 64))
    grad_accum_steps = int(eval_cfg.get("grad_accum_steps", 16))
    total_epochs = int(eval_cfg.get("epochs", 3))
    batch_size = int(eval_cfg.get("batch_size", 4))  # Smaller batch for 1.5B model
    eval_batch_size = int(eval_cfg.get("eval_batch_size", 2))
    learning_rate = float(eval_cfg.get("learning_rate", 0.0002))
    map_num_proc = int(eval_cfg.get("num_proc", 1))
    datasets.disable_progress_bars()
    train_texts = _load_training_texts(stage2_dir)

    def tokenize(example):
        sent = example["text"]
        encoded_dict = tokenizer(
            sent,
            max_length=cutoff_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_dict["input_ids"].flatten().long()
        labels = [-100 if token == tokenizer.pad_token_id else token for token in input_ids.tolist()]
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": encoded_dict["attention_mask"].flatten().long().tolist(),
            "labels": labels,
        }

    train_dataset = Dataset.from_list([{"text": text} for text in train_texts])
    train_tokenized = train_dataset.shuffle(seed=int(config.meta.get("seed", 42))).map(tokenize, num_proc=map_num_proc)
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    eval_dataset = Dataset.from_list([{"text": text} for text in dataset_bundle.eval_texts])
    eval_tokenized = eval_dataset.map(tokenize, num_proc=map_num_proc)
    eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, num_workers=0)
    eval_loader = DataLoader(eval_tokenized, batch_size=eval_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    best_accuracy = 0.0
    best_dict = None

    # Evaluate baseline
    avg_loss, top_k_accuracies = evaluate(model, eval_loader, device, cross_entropy_loss)
    baseline_stats = dict(top_k_accuracies)
    baseline_stats["cross_entropy_loss"] = avg_loss
    with (log_dir / "baseline_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_stats, handle)

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        actual_updates = len(train_loader) // grad_accum_steps + (1 if len(train_loader) % grad_accum_steps != 0 else 0)
        avg_loss = total_loss / actual_updates if actual_updates else 0.0
        avg_loss, top_k_accuracies = evaluate(model, eval_loader, device, cross_entropy_loss)

        stats = dict(top_k_accuracies)
        stats["cross_entropy_loss"] = avg_loss
        with (log_dir / f"epoch{epoch}_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle)
        if best_accuracy < top_k_accuracies[1]:
            best_accuracy = top_k_accuracies[1]
            best_dict = stats
        save_checkpoint(model, optimizer, epoch, filename=str(log_dir / f"checkpoint{epoch}.pth"))

    if best_dict is not None:
        with (log_dir / "best_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(best_dict, handle)

    return StageSummary(
        stage_name="eval_large",
        output_dir=output_dir,
        artifacts={"stats_dir": str(log_dir), "synthetic_train_count": len(train_texts)},
        metrics={
            "epochs": total_epochs,
            "synthetic_train_count": len(train_texts),
            "eval_count": len(dataset_bundle.eval_texts),
            "best_top1": best_accuracy,
            "model": model_name,
            "fine_tune_method": "full",
        },
    )