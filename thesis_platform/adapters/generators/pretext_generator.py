from __future__ import annotations

from thesis_platform.algorithms.generators.pretext_variation import VariationEngine
from thesis_platform.core.schemas import Sample


class PretextSeedGenerator:
    def __init__(self, config, repo_root):
        del repo_root
        self.generated_per_round = int(config.get("generated_per_round", 100))
        self.mask_ratio = float(config.get("mask", 0.3))
        self.t_steps = int(config.get("t_steps", 2))
        self.seed = int(config.get("seed", 42))

    def generate(self, round_ctx):
        if not round_ctx.public_seed_samples:
            raise ValueError("pretext_seed requires public_seed_samples in the round context.")
        engine = VariationEngine(seed=self.seed + round_ctx.round_id, mask_ratio=self.mask_ratio, t_steps=self.t_steps)
        pool = round_ctx.public_seed_samples
        generated: list[Sample] = []
        for idx in range(self.generated_per_round):
            source = pool[idx % len(pool)]
            mutated = engine.mutate(source.text, index=idx)
            generated.append(
                Sample(
                    sample_id=f"syn_r{round_ctx.round_id}_{idx}",
                    client_id="server",
                    round_id=round_ctx.round_id,
                    source="synthetic",
                    dataset_name=source.dataset_name,
                    task_type=source.task_type,
                    text=mutated,
                    meta={"seed_sample_id": source.sample_id, "prompt": round_ctx.prompt_text},
                )
            )
        return generated
