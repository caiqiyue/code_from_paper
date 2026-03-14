from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from pretext_platform.core.config import ExperimentConfig


def legacy_experiment_id(args: Namespace) -> str:
    """Reproduce the original experiment directory naming scheme."""

    return (
        f"{args.datadir}_{args.mask}_{args.lookahead}_{args.multiplier * 256}_"
        f"{args.t_steps}_{args.H_multiplier}_{args.sensitivity}_{args.sigma}_{args.delta}_{args.trial}"
    )


def build_legacy_config(args: Namespace, *, base_dir: str | Path, stage: str) -> ExperimentConfig:
    """Translate one legacy CLI call into the new config structure."""

    experiment_id = legacy_experiment_id(args)
    eval_small_enabled = stage in {"pipeline", "eval_small"}
    eval_large_enabled = stage in {"pipeline", "eval_large"}
    raw = {
        "meta": {
            "experiment_id": experiment_id,
            "seed": 42,
        },
        "paths": {
            "repo_root": ".",
            "dataset_root": "../datasets",
            "model_root": "../thesis_platform/open_model",
            "output_root": args.outputdir,
        },
        "data": {
            "dataset_name": args.datadir,
            "train_path": f"../datasets/{args.datadir}_train.json",
            "eval_path": f"../datasets/{args.datadir}_eval.json",
            "initialization_path": "../datasets/initial_set.json",
            "max_samples_per_client": args.sensitivity,
            "initialization_min_words": 20,
        },
        "models": {
            "minilm_path": "../thesis_platform/open_model/all_minilm_l6_v2",
            "roberta_large_path": "../thesis_platform/open_model/roberta_large",
            "llama2_7b_path": "../thesis_platform/open_model/llama_2_7b_hf",
            "distilgpt2_path": "../thesis_platform/open_model/distilgpt2",
            "c4_checkpoint_path": "./c4_checkpoint.pth",
        },
        "stage1": {
            "enabled": stage in {"pipeline", "stage1"},
            "rounds": 11,
            "mask": args.mask,
            "lookahead": args.lookahead,
            "multiplier": args.multiplier,
            "seq_len": args.seq_len,
            "t_steps": args.t_steps,
            "sigma": args.sigma,
            "delta": args.delta,
            "sensitivity": args.sensitivity,
            "H_multiplier": args.H_multiplier,
            "batch_size": 256,
            "embed_batch_size": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "nearest_neighbors_print": 3,
            "num_workers": 1,
        },
        "bootstrap": {
            "enabled": stage in {"pipeline", "bootstrap"},
            "num_prompts": 50000,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 85,
            "max_model_len": 1000,
        },
        "eval_small": {
            "enabled": eval_small_enabled,
            "cutoff_len": 64,
            "grad_accum_steps": 64,
            "epochs": 20,
            "batch_size": 256,
            "eval_batch_size": 8,
            "learning_rate": 0.0002,
            "num_proc": 1,
        },
        "eval_large": {
            "enabled": eval_large_enabled,
            "cutoff_len": 64,
            "grad_accum_steps": 16,
            "epochs": 1,
            "batch_size": 8,
            "eval_batch_size": 2,
            "learning_rate": 0.0002,
            "num_proc": 1,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
        },
        "runtime": {
            "device": "cuda",
        },
    }
    return ExperimentConfig.from_mapping(raw, base_dir=base_dir, name=f"{experiment_id}.yaml")
