"""Llama-3.2-3B-Instruct downstream evaluation on synthetic data - Windows compatible.
Uses full fine-tuning without PEFT LoRA (Llama-3.2-3B is small enough for full fine-tune).
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


def evaluate(model, eval_loader, accelerator, xent_loss):
    """Compute loss and top-k token accuracy on the evaluation split."""

    import torch

    model.eval()
    total_loss = 0.0
    top_k_accuracies = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    total_evaluated_tokens = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
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

    total_evaluated_tokens = torch.sum(accelerator.gather(total_evaluated_tokens).detach().cpu()).item()
    total_loss = torch.sum(accelerator.gather(total_loss).detach().cpu()).item()
    for k in top_k_accuracies:
        correct_tokens = torch.sum(accelerator.gather(top_k_accuracies[k]).detach().cpu()).item()
        top_k_accuracies[k] = correct_tokens / total_evaluated_tokens if total_evaluated_tokens > 0 else 0
    avg_loss = total_loss / total_evaluated_tokens if len(eval_loader) > 0 else 0.0
    return avg_loss, top_k_accuracies


def save_checkpoint(model, optimizer, accelerator, epoch, filename="checkpoint.pth"):
    """Persist the current model, optimizer, and accelerator state to disk."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accelerator_rng_state": accelerator.state,
    }
    accelerator.save(checkpoint, filename)


def run_llama32_eval(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    model_paths: ModelPaths,
    stage2_dir: Path,
    output_dir: Path,
) -> StageSummary:
    """Fine-tune Llama-3.2-3B-Instruct on synthetic data and evaluate next-token prediction.
    Windows compatible version - uses full fine-tuning without PEFT LoRA.
    """

    import datasets
    import torch
    import torch.nn as nn
    from accelerate import Accelerator
    from datasets import Dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_cfg = config.eval_large
    output_dir = ensure_dir(output_dir)
    log_dir = ensure_dir(output_dir / "llama32_models_and_accuracies")
    accelerator = Accelerator()

    # Use Llama-3.2-3B-Instruct (already downloaded)
    llama32_path = str(model_paths.llama_3_2_3b_instruct)

    tokenizer = AutoTokenizer.from_pretrained(llama32_path, local_files_only=True)
    # Load model in float16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        llama32_path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    # Add PEFT LoRA for efficient fine-tuning
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=int(eval_cfg.get("lora_rank", 4)),
        lora_alpha=int(eval_cfg.get("lora_alpha", 8)),
        lora_dropout=float(eval_cfg.get("lora_dropout", 0.0)),
        target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to accelerator device
    model = model.to(accelerator.device)

    cutoff_len = int(eval_cfg.get("cutoff_len", 64))
    grad_accum_steps = int(eval_cfg.get("grad_accum_steps", 16))
    total_epochs = int(eval_cfg.get("epochs", 3))
    batch_size = int(eval_cfg.get("batch_size", 4))  # Smaller batch for 3B model
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
    # Prepare dataloaders only (model is already on correct device)
    train_loader, eval_loader = accelerator.prepare(train_loader, eval_loader)

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    best_accuracy = 0.0
    best_dict = None
    avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
    if accelerator.is_main_process:
        baseline_stats = dict(top_k_accuracies)
        baseline_stats["cross_entropy_loss"] = avg_loss
        with (log_dir / "baseline_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(baseline_stats, handle)

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            accelerator.backward(loss)
            total_loss += loss.item()
            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        actual_updates = len(train_loader) // grad_accum_steps + (1 if len(train_loader) % grad_accum_steps != 0 else 0)
        avg_loss = total_loss / actual_updates if actual_updates else 0.0
        avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
        if accelerator.is_main_process:
            stats = dict(top_k_accuracies)
            stats["cross_entropy_loss"] = avg_loss
            with (log_dir / f"epoch{epoch}_stats.json").open("w", encoding="utf-8") as handle:
                json.dump(stats, handle)
            if best_accuracy < top_k_accuracies[1]:
                best_accuracy = top_k_accuracies[1]
                best_dict = stats
            save_checkpoint(model, optimizer, accelerator, epoch, filename=str(log_dir / f"checkpoint{epoch}.pth"))

    if accelerator.is_main_process and best_dict is not None:
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
            "model": "Llama-3.2-3B-Instruct",
            "fine_tune_method": "peft_lora",
        },
    )