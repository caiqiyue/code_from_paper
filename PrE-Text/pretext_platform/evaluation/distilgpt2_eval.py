"""DistilGPT2 downstream evaluation on synthetic data."""

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


def evaluate(model, eval_loader, xent_loss):
    """Compute loss and top-k token accuracy on the evaluation split."""

    import torch

    model.eval()
    total_loss = 0.0
    top_k_accuracies = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    total_evaluated_tokens = 0
    with torch.no_grad():
        for batch in eval_loader:
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
            total_loss += loss

            _, top_k_indices = torch.topk(filtered_logits, k=max(top_k_accuracies.keys()), dim=-1)
            expanded_labels = filtered_labels.unsqueeze(1)
            correct_predictions = top_k_indices == expanded_labels
            for k in top_k_accuracies:
                top_k_accuracies[k] += correct_predictions[:, :k].sum()
            total_evaluated_tokens += valid_mask.sum()

    total_evaluated_tokens = float(total_evaluated_tokens.detach().cpu().item())
    total_loss = float(total_loss.detach().cpu().item())
    for k in top_k_accuracies:
        correct_tokens = float(top_k_accuracies[k].detach().cpu().item())
        top_k_accuracies[k] = correct_tokens / total_evaluated_tokens if total_evaluated_tokens > 0 else 0
    avg_loss = total_loss / total_evaluated_tokens if len(eval_loader) > 0 and total_evaluated_tokens > 0 else 0.0
    return avg_loss, top_k_accuracies


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """Persist the current model and optimizer state to disk."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    import torch

    torch.save(checkpoint, filename)


def run_distilgpt2_eval(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    model_paths: ModelPaths,
    stage2_dir: Path,
    output_dir: Path,
) -> StageSummary:
    """Fine-tune DistilGPT2 on synthetic data and evaluate next-token prediction."""

    eval_cfg = config.eval_small
    if model_paths.c4_checkpoint is None or not model_paths.c4_checkpoint.is_file():
        raise ValueError(
            "eval_small is enabled, but models.c4_checkpoint_path is missing or does not exist. "
            "Provide the warm-start c4_checkpoint.pth explicitly."
        )

    import datasets
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, GPT2Tokenizer

    output_dir = ensure_dir(output_dir)
    log_dir = ensure_dir(output_dir / "log_models_and_accuracies")
    device_name = str(config.runtime.get("device", "cuda:1"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(str(model_paths.distilgpt2), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_paths.distilgpt2), local_files_only=True)
    vocab_size = tokenizer.vocab_size
    cached_embedding = model.transformer.wte.weight[:vocab_size]
    dim = model.transformer.wte.weight.shape[1]
    pad_idx = vocab_size
    extended_embedding = nn.Embedding(vocab_size + 1, dim, padding_idx=pad_idx)
    extended_weight = torch.cat([cached_embedding, torch.zeros(1, dim)])
    extended_embedding.load_state_dict({"weight": extended_weight})
    model.transformer.wte = extended_embedding
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = model.to(device)

    cutoff_len = int(eval_cfg.get("cutoff_len", 64))
    grad_accum_steps = int(eval_cfg.get("grad_accum_steps", 64))
    total_epochs = int(eval_cfg.get("epochs", 20))
    batch_size = int(eval_cfg.get("batch_size", 256))
    eval_batch_size = int(eval_cfg.get("eval_batch_size", 8))
    learning_rate = float(eval_cfg.get("learning_rate", 0.0002))
    map_num_proc = int(eval_cfg.get("num_proc", 1))
    datasets.disable_progress_bars()
    train_texts = _load_training_texts(stage2_dir)

    def tokenize(example):
        sent = example["text"]
        sent = tokenizer.tokenize(sent)
        sent = [tokenizer.bos_token] + sent + [tokenizer.eos_token]
        encoded_dict = tokenizer.encode_plus(
            sent,
            max_length=cutoff_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True,
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

    checkpoint = torch.load(model_paths.c4_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    best_accuracy = 0.0
    best_dict = None
    avg_loss, top_k_accuracies = evaluate(model, eval_loader, cross_entropy_loss)
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
        avg_loss, top_k_accuracies = evaluate(model, eval_loader, cross_entropy_loss)
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
        stage_name="eval_small",
        output_dir=output_dir,
        artifacts={"stats_dir": str(log_dir), "synthetic_train_count": len(train_texts)},
        metrics={
            "epochs": total_epochs,
            "synthetic_train_count": len(train_texts),
            "eval_count": len(dataset_bundle.eval_texts),
            "best_top1": best_accuracy,
        },
    )
