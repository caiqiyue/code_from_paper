"""Expand DP seed texts into a larger synthetic dataset with Llama 2."""

import argparse
import json
import os
import random
import sys

from vllm import LLM, SamplingParams


def build_output_dir(args):
    """Construct the experiment directory name shared by generation and evaluation."""
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


def load_surviving_seed_texts(output_dir, num_rounds=11):
    """Load all surviving seed texts produced during stage-one Private Evolution."""
    seed_texts = []
    for round_idx in range(num_rounds):
        with open(
            os.path.join(output_dir, f"surviving_text_it{round_idx}.json"),
            "r",
            encoding="utf8",
        ) as file:
            seed_texts.extend(json.load(file))
    return seed_texts


def build_bootstrap_prompts(seed_texts, num_prompts=50000):
    """Create few-shot prompts that ask Llama 2 to continue the synthetic corpus."""
    single_prompt = (
        "List of 6 diverse original text samples:\n"
        "Original Text Sample 1\n{0}\n"
        "Original Text Sample 2\n{1}\n"
        "Original Text Sample 3\n{2}\n"
        "Original Text Sample 4\n"
    )
    prompt_list = []
    for _ in range(num_prompts):
        examples = random.sample(seed_texts, 3)  # Draw three seed texts to anchor one prompt.
        curr_prompt = single_prompt.format(
            examples[0].replace("\n", " ").replace("\t", " "),
            examples[1].replace("\n", " ").replace("\t", " "),
            examples[2].replace("\n", " ").replace("\t", " "),
        )
        prompt_list.append(curr_prompt)
    return prompt_list


def generate_bootstrapped_samples(prompt_list, cache_dir):
    """Run vLLM generation for the bootstrap prompts and return the raw outputs."""
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        download_dir=cache_dir,
        max_model_len=1000,
    )
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=85)
    outputs = llm.generate(prompt_list, sampling_params)  # Generate one continuation for each prompt.
    return [output.outputs[0].text for output in outputs]


def parse_args():
    """Parse command-line arguments for the bootstrap stage."""
    parser = argparse.ArgumentParser("Run expansion.")
    parser.add_argument("-datadir", type=str, default="", help="Dataset name prefix used in the experiment directory.")
    parser.add_argument("-outputdir", type=str, required=True, help="Base directory where experiment outputs are stored.")
    parser.add_argument("-cachedir", type=str, required=True, help="Directory used to cache Hugging Face model downloads.")
    parser.add_argument("-sensitivity", type=int, required=True, help="Maximum number of samples per client.")
    parser.add_argument("-delta", type=float, default=3e-6, help="Target delta in (epsilon, delta)-DP.")
    parser.add_argument("-sigma", type=float, required=True, help="Noise-to-sensitivity ratio used in stage one.")
    parser.add_argument("-mask", type=float, default=0.3, help="Masking ratio used during stage-one generation.")
    parser.add_argument("-lookahead", type=int, default=4, help="Lookahead count used during stage-one generation.")
    parser.add_argument("-multiplier", type=int, default=4, help="Multiplier for the number of synthetic candidates per round.")
    parser.add_argument("-seq_len", type=int, default=64, help="Sequence length used during stage-one generation.")
    parser.add_argument("-t_steps", type=int, default=2, help="Number of mask-fill steps used for each candidate mutation.")
    parser.add_argument("-trial", type=int, default=0, help="Trial identifier appended to the output directory.")
    parser.add_argument("-H_multiplier", type=float, default=0.25, help="Multiplier that controls DP histogram thresholding.")
    return parser.parse_args()


def main():
    """Load stage-one seeds, bootstrap them with Llama 2, and save the outputs."""
    args = parse_args()
    output_dir = build_output_dir(args)
    seed_texts = load_surviving_seed_texts(output_dir)
    print(output_dir, file=sys.stderr)
    print("Number of seeds", len(seed_texts))

    prompt_list = build_bootstrap_prompts(seed_texts)
    print(prompt_list[0])  # Print one example prompt for debugging and reproducibility checks.
    output_list = generate_bootstrapped_samples(prompt_list, args.cachedir)

    with open(os.path.join(output_dir, "llama7b_text_syn.json"), "w+", encoding="utf8") as file:
        json.dump(output_list, file, ensure_ascii=False)  # Persist the expanded synthetic corpus for downstream training.


if __name__ == "__main__":
    main()
