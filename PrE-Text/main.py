"""Stage-one PrE-Text entry point for differentially private seed generation."""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from opacus.accountants.analysis import rdp as privacy_analysis
from sentence_transformers import SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaTokenizer

from nn_histogram import NN_Histogram
from similarity import Similarity
from variation import Variation


def main(args, accelerator):
    """Run Private Evolution and save the differentially private seed texts."""
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-large",
        use_fast=True,
        cache_dir=args.cachedir,
    )
    model = RobertaForMaskedLM.from_pretrained(
        "roberta-large",
        torch_dtype=torch.float16,
        cache_dir=args.cachedir,
    )  # Use RoBERTa as the masked-language model that mutates candidate texts.
    accelerator.print("miniLM", file=sys.stderr)
    mpnet = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=args.cachedir)  # Use MiniLM embeddings for nearest-neighbor scoring.

    datadir = f"./data/{args.datadir}_train.json"
    outputdir = args.outputdir
    seq_len = args.seq_len
    with open(datadir, encoding="utf8") as file:
        private_samples = json.load(file)  # Load the aggregated private federated text samples.

    accelerator.print("Num private train samples", len(private_samples), file=sys.stderr)
    accelerator.print("Private samples", private_samples[:5], file=sys.stderr)

    scale = args.sensitivity * args.sigma
    rdp = privacy_analysis.compute_rdp(
        q=1.0,
        noise_multiplier=args.sigma,
        steps=11,
        orders=[1.0 + 0.1 * t for t in range(1, 1000)],
    )  # Match the 11 Private Evolution rounds when estimating privacy cost.
    eps, _ = privacy_analysis.get_privacy_spent(
        orders=[1.0 + 0.1 * t for t in range(1, 1000)],
        rdp=rdp,
        delta=args.delta,
    )
    accelerator.print("Epsilon of this run", eps, file=sys.stderr)

    model = model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)  # Wrap the MLM for distributed inference only.

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": 256,
        "max_length": seq_len,
        "num_workers": 1,
        "num_gpus": 1,
        "embed_batch_size": 512,
        "mpnet": mpnet,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "nearest_neighbors_print": 3,
        "sigma": scale * 1.541 * np.sqrt(2),  # Convert the configured noise ratio into the histogram noise scale.
        "H": scale * 4.0 * args.H_multiplier,  # Set the threshold used to zero out weak noisy votes.
        "embed_dim": mpnet.get_sentence_embedding_dimension(),
        "lookahead": args.lookahead,
        "T": 11,
        "multiplier": args.multiplier,
        "device": "cuda",
        "t_steps": args.t_steps,
    }
    config["nsyn"] = config["batch_size"] * config["multiplier"] * config["num_gpus"]  # Total synthetic candidates per round.
    output_dir = os.path.join(
        outputdir,
        "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}/".format(
            args.datadir,
            args.mask,
            config["lookahead"],
            config["nsyn"],
            args.t_steps,
            args.H_multiplier,
            args.sensitivity,
            args.sigma,
            args.delta,
            args.trial,
        ),
    )
    accelerator.print(output_dir, file=sys.stderr)

    with open("./data/initialization.json", encoding="utf8") as file:
        load_list = json.load(file)

    load_list = [x for x in load_list if len(x.split(" ")) > 20]  # Keep only sufficiently long initialization texts.
    init_pop = load_list

    if accelerator.is_main_process:
        accelerator.print("Initial population size", len(load_list), file=sys.stderr)
    accelerator.print("Init pop size", len(init_pop))
    schedule = [args.mask for _ in range(config["T"])]  # Reuse a fixed mask ratio across all rounds.
    accelerator.print(output_dir, file=sys.stderr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if accelerator.is_main_process:
        accelerator.print(init_pop[0], file=sys.stderr)
        accelerator.print("Schedule", schedule, file=sys.stderr)

    parent_texts = random.choices(init_pop, k=config["nsyn"])
    parent_texts = sorted(parent_texts, key=lambda x: len(x))  # Sort by length for more stable padding behavior.
    accelerator.print("Num parent texts", len(parent_texts), file=sys.stderr)
    parent_set = tokenizer(
        parent_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )
    dist_list = []
    private_embed_path = os.path.join(output_dir, "private_embeds.npy")

    if not os.path.isfile(private_embed_path):
        accelerator.print("Making private embeddings", file=sys.stderr)
        t0 = time.time()
        private_embeddings = Similarity.concat_embedding(private_samples, config)
        t1 = time.time()
        accelerator.print("Time for private embeddings", t1 - t0, file=sys.stderr)
        np.save(private_embed_path, private_embeddings)  # Cache private embeddings so repeated runs can skip recomputation.
    else:
        private_embeddings = np.load(private_embed_path)
        accelerator.print("Embeddings shape", private_embeddings.shape)

    for t in range(config["T"]):
        attention_mask_pad_sums = torch.sum(parent_set["attention_mask"] == 0, axis=1)
        curr_inputs_pad_sums = torch.sum(parent_set["input_ids"] == tokenizer.pad_token_id, axis=1)
        all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        accelerator.print("At top all_ok?", all_ok)
        t0 = time.time()

        histogram, meandist, nearest_idx = NN_Histogram.dp_nn_histogram(
            private_embeddings,
            parent_set["input_ids"],
            parent_set["attention_mask"],
            schedule[t],
            config,
        )  # Score the current population with a noisy nearest-neighbor vote histogram.
        dist_list.append(meandist)
        accelerator.print("Current step", t, file=sys.stderr)
        accelerator.print("Dist list", dist_list, file=sys.stderr)
        accelerator.print(
            "Nearest generated samples",
            [
                tokenizer.batch_decode([parent_set["input_ids"][idx, :]], skip_special_tokens=True)
                for idx in nearest_idx
            ],
        )
        accelerator.print("Mean dist from nearest neighbor", meandist, file=sys.stderr)

        accelerator.print("Histogram sum", np.sum(histogram), file=sys.stderr)
        t1 = time.time()
        if accelerator.is_main_process:
            accelerator.print("Histogram time:", t1 - t0, file=sys.stderr)
            accelerator.print("Producing surviving parents...", file=sys.stderr)
        t0 = time.time()

        indices = np.random.choice(
            config["nsyn"],
            config["nsyn"],
            p=histogram / np.sum(histogram),
        )  # Resample candidates in proportion to their noisy DP scores.
        indices = np.sort(indices)
        surviving_parents_ids = parent_set["input_ids"][indices, :]
        surviving_parents_mask = parent_set["attention_mask"][indices, :]

        attention_mask_pad_sums = torch.sum(surviving_parents_mask == 0, axis=1)
        curr_inputs_pad_sums = torch.sum(surviving_parents_ids == tokenizer.pad_token_id, axis=1)
        all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        accelerator.print("After sampling all ok?", all_ok)
        t1 = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Choosing survivors time:", t1 - t0, file=sys.stderr)
            accelerator.print("Producing variations...", file=sys.stderr)
        t0 = time.time()
        new_variations = Variation.produce_variation(
            {
                "input_ids": surviving_parents_ids,
                "attention_mask": surviving_parents_mask,
            },
            schedule[t],
            config,
        )  # Mutate the surviving texts to form the next generation.

        attention_mask_pad_sums = torch.sum(new_variations["attention_mask"] == 0, axis=1)
        curr_inputs_pad_sums = torch.sum(new_variations["input_ids"] == tokenizer.pad_token_id, axis=1)
        all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        accelerator.print("Produced variations all_ok?", all_ok)

        t1 = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Producing variations time", t1 - t0, file=sys.stderr)
            accelerator.print("Checking similarity...", file=sys.stderr)

        generated_samples = tokenizer.batch_decode(
            new_variations["input_ids"],
            skip_special_tokens=True,
        )
        surviving_samples = tokenizer.batch_decode(
            surviving_parents_ids,
            skip_special_tokens=True,
        )

        parent_set["input_ids"] = new_variations["input_ids"]
        parent_set["attention_mask"] = new_variations["attention_mask"]

        if accelerator.is_main_process:
            with open(
                os.path.join(output_dir, f"generated_text_it{t}.json"),
                "w+",
                encoding="utf8",
            ) as json_file:
                json.dump(generated_samples, json_file, ensure_ascii=False)  # Save the full next generation for inspection.
            with open(
                os.path.join(output_dir, f"surviving_text_it{t}.json"),
                "w+",
                encoding="utf8",
            ) as json_file:
                json.dump(list(set(surviving_samples)), json_file, ensure_ascii=False)  # Save the unique parent seeds that survived selection.

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser("Run PE-text.")
    parser.add_argument("-datadir", type=str, default="", help="Dataset name prefix for ./data/<name>_train.json.")
    parser.add_argument("-outputdir", type=str, required=True, help="Base directory where experiment outputs are written.")
    parser.add_argument("-cachedir", type=str, required=True, help="Directory used to cache downloaded Hugging Face models.")
    parser.add_argument("-sensitivity", type=int, required=True, help="Maximum number of samples per client.")
    parser.add_argument("-delta", type=float, default=3e-6, help="Target delta in (epsilon, delta)-DP.")
    parser.add_argument("-sigma", type=float, required=True, help="Noise-to-sensitivity ratio.")
    parser.add_argument("-mask", type=float, default=0.3, help="Fraction of valid tokens to mask in each mutation round.")
    parser.add_argument("-lookahead", type=int, default=4, help="Number of future mutations averaged for candidate scoring.")
    parser.add_argument("-multiplier", type=int, default=4, help="Multiplier controlling the synthetic population size.")
    parser.add_argument("-seq_len", type=int, default=64, help="Maximum sequence length for tokenization.")
    parser.add_argument("-t_steps", type=int, default=2, help="How many mask-fill passes to apply per mutation.")
    parser.add_argument("-trial", type=int, default=0, help="Trial identifier appended to the experiment directory.")
    parser.add_argument("-H_multiplier", type=float, default=0.25, help="Multiplier controlling histogram threshold H.")
    with accelerator.main_process_first():
        argobj = parser.parse_args()
    print(argobj)
    main(argobj, accelerator)
