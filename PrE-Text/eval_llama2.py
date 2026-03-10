"""Downstream next-token evaluation with Llama 2 and LoRA adapters."""

import argparse
import json
import os
import re
import sys

import datasets
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaTokenizer


def evaluate(model, eval_loader, accelerator, xent_loss):
    """Compute loss and top-k token accuracy on the evaluation split."""
    model.eval()
    total_loss = 0.0
    top_k_accuracies = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    total_evaluated_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[..., :-1, :].contiguous()  # Predict token t+1 from positions up to t.
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


def add_module_prefix(state_dict):
    """Normalize checkpoint keys so they match wrapped accelerator models."""
    return {
        ("module." + key if not key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_checkpoint(model, optimizer, accelerator, filename="checkpoint.pth"):
    """Load a previously saved training checkpoint."""
    checkpoint = torch.load(filename, map_location=accelerator.device)
    adjusted_model_state_dict = add_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(adjusted_model_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch


def find_latest_checkpoint(checkpoint_dir):
    """Return the most recent numbered checkpoint file in a directory."""
    checkpoint_files = [
        filename
        for filename in os.listdir(checkpoint_dir)
        if filename.startswith("checkpoint") and filename.endswith(".pth")
    ]
    checkpoint_files.sort(
        key=lambda x: int(x.replace("checkpoint", "").replace(".pth", "")),
        reverse=True,
    )
    return checkpoint_files[0] if checkpoint_files else None


def build_output_dir(args):
    """Construct the experiment directory name shared across pipeline stages."""
    return os.path.join(
        args.outputdir,
        "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}/".format(
            args.datadir,
            args.mask,
            args.lookahead,
            args.multiplier * 256,
            args.t_steps,
            args.H_multiplier,
            args.sensitivity,
            args.sigma,
            args.delta,
            args.trial,
        ),
    )


def main(args, output_dir, train_texts, eval_texts):
    """Fine-tune Llama 2 with LoRA adapters on synthetic data and evaluate it."""
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=args.cachedir)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=args.cachedir)
    vocab_size = tokenizer.vocab_size
    cached_embedding = model.model.embed_tokens.weight[:vocab_size]
    dim = model.model.embed_tokens.weight.shape[1]
    pad_idx = vocab_size
    extended_embedding = nn.Embedding(vocab_size + 1, dim, padding_idx=pad_idx)
    extended_weight = torch.cat([cached_embedding, torch.zeros(1, dim)])
    del cached_embedding
    extended_embedding.load_state_dict({"weight": extended_weight})
    model.model.embed_tokens = extended_embedding
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add a dedicated padding token for batched training.
    cutoff_len = 64
    grad_accum_steps = 16
    total_epochs = 1
    batch_size = 8
    log_dir = os.path.join(output_dir, "llama2_models_and_accuracies")
    accelerator = Accelerator()

    if accelerator.is_main_process and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    accelerator.wait_for_everyone()
    datasets.disable_progress_bars()

    def tokenize(example):
        """Tokenize one text example into input ids, attention mask, and labels."""
        sent = example["text"]
        sent = tokenizer.tokenize(sent)

        encoded_dict = tokenizer.encode_plus(
            sent,
            max_length=cutoff_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True,
        )

        input_ids = encoded_dict["input_ids"].flatten().long()
        labels = [
            -100 if token == tokenizer.pad_token_id else token
            for token in input_ids.tolist()
        ]  # Ignore padded positions when computing language-model loss.

        result = {
            "input_ids": input_ids.tolist(),
            "attention_mask": encoded_dict["attention_mask"].flatten().long().tolist(),
            "labels": labels,
        }
        return result

    train_dict = [{"text": x} for x in train_texts]
    train_dataset_hf = Dataset.from_list(train_dict)
    train_data_tokenized = train_dataset_hf.shuffle().map(tokenize, num_proc=5)
    train_data_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    test_data_dict = [{"text": x} for x in eval_texts]
    test_dataset_hf = Dataset.from_list(test_data_dict)
    test_data_tokenized = test_dataset_hf.map(tokenize, num_proc=5)
    test_data_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(train_data_tokenized, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(test_data_tokenized, batch_size=2, drop_last=False, shuffle=False)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )  # Train low-rank adapters instead of updating the full 7B-parameter model.
    model = get_peft_model(model, peft_config)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model,
        AdamW(model.parameters(), lr=0.0002),
        train_dataloader,
        test_dataloader,
    )

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
    if accelerator.is_main_process:
        print(f"No finetuning, evaluation Loss: {avg_loss:.4f}", file=sys.stderr)
        for k, accuracy in top_k_accuracies.items():
            print(f"No finetuning, Top-{k} Accuracy: {accuracy:.4f}", file=sys.stderr)

    best_accuracy = 0.0
    best_dict = None
    for epoch in range(total_epochs):
        model.train()
        total_loss = 0
        curr_step_loss = 0
        num_actual_steps = 1
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps  # Scale loss because gradients are accumulated across many mini-batches.
            accelerator.backward(loss)
            total_loss += loss.item()
            curr_step_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    print(f"Epoch {epoch}, Step {num_actual_steps} loss: {curr_step_loss}", file=sys.stderr)
                curr_step_loss = 0
                num_actual_steps += 1

        actual_updates = len(train_loader) // grad_accum_steps + (
            1 if len(train_loader) % grad_accum_steps != 0 else 0
        )
        avg_loss = total_loss / actual_updates
        if accelerator.is_main_process:
            print(f"Epoch {epoch} Avg training loss: {avg_loss}", file=sys.stderr)

        avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} evaluation Loss: {avg_loss:.4f}", file=sys.stderr)
            for k, accuracy in top_k_accuracies.items():
                print(f"Epoch {epoch} Top-{k} Accuracy: {accuracy:.4f}", file=sys.stderr)
            top_k_accuracies["cross_entropy_loss"] = avg_loss
            stats_path = os.path.join(log_dir, f"epoch{epoch}_stats.json")
            print("Saving stats in ", stats_path, file=sys.stderr)
            with open(stats_path, "w+") as file:
                json.dump(top_k_accuracies, file)  # Persist metrics after every evaluation epoch.

            if best_accuracy < top_k_accuracies[1]:
                best_accuracy = top_k_accuracies[1]
                best_dict = top_k_accuracies

            checkpoint_path = os.path.join(log_dir, f"checkpoint{epoch}.pth")
            save_checkpoint(model, optimizer, accelerator, epoch, filename=checkpoint_path)

        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        stats_path = os.path.join(log_dir, "best_stats.json")
        print("Saving stats in ", stats_path, file=sys.stderr)
        with open(stats_path, "w+") as file:
            json.dump(best_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate on downstream LLaMA-2 next token prediction task.")
    parser.add_argument("-datadir", type=str, default="", help="Dataset name prefix for ./data/<name>_eval.json.")
    parser.add_argument("-outputdir", type=str, required=True, help="Base directory where experiment outputs are stored.")
    parser.add_argument("-cachedir", type=str, required=True, help="Directory used to cache downloaded models.")
    parser.add_argument("-sensitivity", type=int, required=True, help="Maximum number of samples per client.")
    parser.add_argument("-delta", type=float, default=3e-6, help="Target delta in (epsilon, delta)-DP.")
    parser.add_argument("-sigma", type=float, required=True, help="Noise-to-sensitivity ratio used by the corresponding generation run.")
    parser.add_argument("-mask", type=float, default=0.3, help="Mask ratio used by the corresponding generation run.")
    parser.add_argument("-lookahead", type=int, default=4, help="Lookahead count used by the corresponding generation run.")
    parser.add_argument("-multiplier", type=int, default=4, help="Synthetic population multiplier used during stage one.")
    parser.add_argument("-seq_len", type=int, default=64, help="Sequence length used by the corresponding generation run.")
    parser.add_argument("-t_steps", type=int, default=2, help="Mutation steps used by the corresponding generation run.")
    parser.add_argument("-trial", type=int, default=0, help="Trial identifier appended to the experiment directory.")
    parser.add_argument("-H_multiplier", type=float, default=0.25, help="Histogram threshold multiplier used by the generation run.")
    args = parser.parse_args()
    output_dir = build_output_dir(args)
    with open(os.path.join(f"./data/{args.datadir}_eval.json"), "r", encoding="utf8") as file:
        test_data_raw = json.load(file)["1"]
    with open(os.path.join(output_dir, "llama7b_text_syn.json"), "r", encoding="utf8") as file:
        synthetic_outputs = json.load(file)
    all_data = []
    for text in synthetic_outputs:
        split_samples = re.split("Orig", text)
        raw_sample = split_samples[0].strip().strip("\n")
        if len(raw_sample.split(" ")) > 3:
            all_data.append(raw_sample.replace("\n\n", " ").replace("\n", " "))  # Keep only plausible generated training samples.
    main(args, output_dir, all_data, test_data_raw)
