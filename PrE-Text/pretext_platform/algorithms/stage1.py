"""Stage 1 Private Evolution runner."""

from __future__ import annotations

import json
import random
import sys
import time
from typing import Any, Callable

import numpy as np

from pretext_platform.algorithms.histogram import NN_Histogram
from pretext_platform.algorithms.similarity import Similarity
from pretext_platform.algorithms.variation import Variation
from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.io_utils import ensure_dir
from pretext_platform.core.types import DatasetBundle, ModelPaths, StageSummary


def create_accelerator():
    """Create the default accelerator used by the legacy algorithm implementation."""

    from accelerate import Accelerator

    return Accelerator()


def load_stage1_models(model_paths: ModelPaths, *, device: str):
    """Load the local MLM tokenizer/model and sentence-transformer encoder."""

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(
        str(model_paths.roberta_large),
        use_fast=True,
        local_files_only=True,
    )
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    model = RobertaForMaskedLM.from_pretrained(str(model_paths.roberta_large), **model_kwargs)
    mpnet = SentenceTransformer(str(model_paths.minilm))
    return tokenizer, model, mpnet


def _set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch randomness for reproducible experiments."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_epsilon(*, rounds: int, sigma_ratio: float, delta: float) -> float:
    """Compute the RDP epsilon estimate for the configured Stage 1 run."""

    from opacus.accountants.analysis import rdp as privacy_analysis

    orders = [1.0 + 0.1 * t for t in range(1, 1000)]
    rdp = privacy_analysis.compute_rdp(
        q=1.0,
        noise_multiplier=sigma_ratio,
        steps=rounds,
        orders=orders,
    )
    eps, _ = privacy_analysis.get_privacy_spent(
        orders=orders,
        rdp=rdp,
        delta=delta,
    )
    return float(eps)


def _write_json(path, payload: Any) -> None:
    """Write a UTF-8 JSON file."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def run_private_evolution_stage(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    model_paths: ModelPaths,
    output_dir,
    *,
    accelerator=None,
    model_loader: Callable[..., Any] = load_stage1_models,
) -> StageSummary:
    """Run Stage 1 exactly in the original Private Evolution order."""

    import torch

    stage1_cfg = config.stage1
    runtime_cfg = config.runtime
    output_dir = ensure_dir(output_dir)
    accelerator = accelerator or create_accelerator()

    seed = int(config.meta.get("seed", 42))
    _set_random_seed(seed)
    device = str(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, model, mpnet = model_loader(model_paths, device=device)

    rounds = int(stage1_cfg.get("rounds", 11))
    sensitivity = int(stage1_cfg.get("sensitivity", config.data.get("max_samples_per_client", 8)))
    sigma_ratio = float(stage1_cfg.get("sigma", 2.31))
    delta = float(stage1_cfg.get("delta", 3e-6))
    mask = float(stage1_cfg.get("mask", 0.3))
    scale = sensitivity * sigma_ratio
    epsilon = _compute_epsilon(rounds=rounds, sigma_ratio=sigma_ratio, delta=delta)

    accelerator.print("Epsilon of this run", epsilon, file=sys.stderr)
    model = model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)

    stage_config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": int(stage1_cfg.get("batch_size", 256)),
        "max_length": int(stage1_cfg.get("seq_len", 64)),
        "num_workers": int(stage1_cfg.get("num_workers", 1)),
        "embed_batch_size": int(stage1_cfg.get("embed_batch_size", 512)),
        "mpnet": mpnet,
        "temperature": float(stage1_cfg.get("temperature", 1.0)),
        "top_p": float(stage1_cfg.get("top_p", 1.0)),
        "top_k": int(stage1_cfg.get("top_k", 0)),
        "nearest_neighbors_print": int(stage1_cfg.get("nearest_neighbors_print", 3)),
        "sigma": scale * 1.541 * np.sqrt(2.0),
        "H": scale * 4.0 * float(stage1_cfg.get("H_multiplier", 0.25)),
        "embed_dim": mpnet.get_sentence_embedding_dimension(),
        "lookahead": int(stage1_cfg.get("lookahead", 4)),
        "T": rounds,
        "multiplier": int(stage1_cfg.get("multiplier", 4)),
        "device": device,
        "t_steps": int(stage1_cfg.get("t_steps", 2)),
    }
    stage_config["nsyn"] = stage_config["batch_size"] * stage_config["multiplier"]

    init_pop = list(dataset_bundle.initialization_texts)
    if not init_pop:
        raise ValueError("Initialization pool is empty after filtering. Adjust data.initialization_min_words or the seed dataset.")

    parent_texts = random.choices(init_pop, k=stage_config["nsyn"])
    parent_texts = sorted(parent_texts, key=lambda text: len(text))
    parent_set = tokenizer(
        parent_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=stage_config["max_length"],
    )

    private_embed_path = output_dir / "private_embeds.npy"
    if not private_embed_path.is_file():
        accelerator.print("Making private embeddings", file=sys.stderr)
        t0 = time.time()
        private_embeddings = Similarity.concat_embedding(dataset_bundle.train_texts, stage_config)
        np.save(private_embed_path, private_embeddings)
        accelerator.print("Time for private embeddings", time.time() - t0, file=sys.stderr)
    else:
        private_embeddings = np.load(private_embed_path)
        accelerator.print("Embeddings shape", private_embeddings.shape, file=sys.stderr)

    schedule = [mask for _ in range(stage_config["T"])]
    mean_distances: list[float] = []
    generated_files: list[str] = []
    surviving_files: list[str] = []
    histogram_sums: list[float] = []

    for round_idx in range(stage_config["T"]):
        histogram, mean_distance, nearest_idx = NN_Histogram.dp_nn_histogram(
            private_embeddings,
            parent_set["input_ids"],
            parent_set["attention_mask"],
            schedule[round_idx],
            stage_config,
        )
        histogram_sum = float(np.sum(histogram))
        if histogram_sum <= 0.0:
            raise ValueError(
                "DP histogram sum is zero after thresholding. Increase stage1.multiplier, reduce stage1.H_multiplier, "
                "or adjust stage1.sigma before retrying."
            )

        mean_distances.append(float(mean_distance))
        histogram_sums.append(histogram_sum)
        accelerator.print("Current step", round_idx, file=sys.stderr)
        accelerator.print("Dist list", mean_distances, file=sys.stderr)
        accelerator.print(
            "Nearest generated samples",
            [
                tokenizer.batch_decode([parent_set["input_ids"][idx, :]], skip_special_tokens=True)
                for idx in nearest_idx
            ],
        )

        indices = np.random.choice(stage_config["nsyn"], stage_config["nsyn"], p=histogram / histogram_sum)
        indices = np.sort(indices)
        surviving_parent_ids = parent_set["input_ids"][indices, :]
        surviving_parent_mask = parent_set["attention_mask"][indices, :]
        new_variations = Variation.produce_variation(
            {"input_ids": surviving_parent_ids, "attention_mask": surviving_parent_mask},
            schedule[round_idx],
            stage_config,
        )
        generated_samples = tokenizer.batch_decode(new_variations["input_ids"], skip_special_tokens=True)
        surviving_samples = tokenizer.batch_decode(surviving_parent_ids, skip_special_tokens=True)

        parent_set["input_ids"] = new_variations["input_ids"]
        parent_set["attention_mask"] = new_variations["attention_mask"]

        generated_path = output_dir / f"generated_text_it{round_idx}.json"
        surviving_path = output_dir / f"surviving_text_it{round_idx}.json"
        if accelerator.is_main_process:
            _write_json(generated_path, generated_samples)
            _write_json(surviving_path, list(set(surviving_samples)))
        generated_files.append(str(generated_path))
        surviving_files.append(str(surviving_path))

    accelerator.wait_for_everyone()
    return StageSummary(
        stage_name="stage1",
        output_dir=output_dir,
        artifacts={
            "private_embeddings": str(private_embed_path),
            "generated_files": generated_files,
            "surviving_files": surviving_files,
        },
        metrics={
            "epsilon": epsilon,
            "delta": delta,
            "sigma_ratio": sigma_ratio,
            "sensitivity": sensitivity,
            "private_train_count": len(dataset_bundle.train_texts),
            "init_population_count": len(init_pop),
            "nsyn": stage_config["nsyn"],
            "rounds": rounds,
            "histogram_sums": histogram_sums,
            "mean_distances": mean_distances,
        },
    )
