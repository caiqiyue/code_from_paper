"""Utilities for mutating candidate texts during Private Evolution."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from custom_datasets import MatrixDataset


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Apply top-k and/or nucleus filtering to a batch of token logits."""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Bound k to a valid range.
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]  # Drop tokens outside top-k.
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # Track cumulative mass.

        sorted_indices_to_remove = cumulative_probs > top_p  # Mark tokens beyond the nucleus threshold.
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0  # Keep a minimum number of choices.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # Preserve first over-threshold token.
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class Variation:
    """Generate masked-token variations of candidate texts with a masked LM."""

    @staticmethod
    def collate_fn_tokenizer(inputs, num_mask_percentage, tokenizer):
        """Batch candidate texts and mask a shared fraction of valid tokens."""
        input_ids = torch.cat([x["input_ids"] for x in inputs], dim=0)  # Merge the per-item tensors into one batch.
        attention_mask = torch.cat([x["attention_mask"] for x in inputs], dim=0)
        num_valid_tokens = torch.min(torch.sum(attention_mask, dim=1)) * num_mask_percentage  # Use the shortest item as the mask budget.
        num_mask = int(num_valid_tokens)
        input_ids = input_ids.clone()
        labels = input_ids.clone()
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix = torch.full(labels.shape, 1.0)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # Never mask special tokens.
        masked_indices_col = torch.multinomial(probability_matrix, num_samples=num_mask)  # Sample token positions to rewrite.
        masked_indices_row = torch.arange(labels.shape[0])[:, None].expand(size=masked_indices_col.shape)
        input_ids[masked_indices_row, masked_indices_col] = tokenizer.mask_token_id  # Replace selected tokens with [MASK].
        collate_output = {
            "inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
            "masked_indices_col": masked_indices_col,
            "masked_indices_row": masked_indices_row,
            "num_masks": num_mask,
        }
        return collate_output

    @staticmethod
    def sample(inputs, masked_indices_col, masked_indices_row, num_masks, config):
        """Fill masked positions one at a time by sampling from the masked LM."""
        model = config["model"]
        curr_inputs = inputs["input_ids"].clone().to(model.device)  # Move a writable copy to the model device.
        masked_indices_col = masked_indices_col.to(model.device)
        masked_indices_row = masked_indices_row.to(model.device)
        for i in range(num_masks):
            outputs = model(
                input_ids=curr_inputs,
                attention_mask=inputs["attention_mask"],
            ).logits  # Recompute logits after every sampled replacement.
            mask_token_logits = (
                outputs[masked_indices_row[:, i], masked_indices_col[:, i], :].float()
                / config["temperature"]
            )
            filtered_logits = top_k_top_p_filtering(
                mask_token_logits,
                top_k=config["top_k"],
                top_p=config["top_p"],
            )
            probs = torch.nn.functional.softmax(filtered_logits, dim=1)
            replacement_tokens = torch.multinomial(probs, num_samples=1)  # Draw one replacement token per sequence.
            curr_inputs[masked_indices_row[:, i], masked_indices_col[:, i]] = replacement_tokens.squeeze()

        return {"input_ids": curr_inputs, "attention_mask": inputs["attention_mask"]}

    @staticmethod
    def produce_variation(parent_set, variation_deg, config):
        """Run one or more mask-fill rounds to produce the next candidate population."""
        accelerator = config["accelerator"]
        tokenizer = config["tokenizer"]
        t_steps = config["t_steps"]
        curr_ids = parent_set["input_ids"]
        curr_masks = parent_set["attention_mask"]
        for _ in range(t_steps):
            parent_dataset = MatrixDataset({"input_ids": curr_ids, "attention_mask": curr_masks})
            population_dataloader = DataLoader(
                parent_dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                collate_fn=lambda x: Variation.collate_fn_tokenizer(x, variation_deg, tokenizer),
            )
            population_dataloader = accelerator.prepare(population_dataloader)  # Shard the candidate batch across processes.
            offspring = []
            for _, batch in enumerate(population_dataloader):
                with torch.no_grad():
                    outputs = Variation.sample(**batch, config=config)
                    output_ids = outputs["input_ids"]
                    offspring.append(accelerator.gather(output_ids).cpu())  # Reassemble all generated sequences on CPU.
            produced_ids = torch.cat(offspring)
            curr_ids = produced_ids
            curr_masks = (curr_ids != tokenizer.pad_token_id).long()  # Refresh masks after replacing tokens.
        return {"input_ids": curr_ids, "attention_mask": curr_masks}
