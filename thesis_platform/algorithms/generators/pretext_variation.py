from __future__ import annotations

import random


class VariationEngine:
    """Small text variation engine inspired by PrE-Text mutation behavior."""

    def __init__(self, *, seed: int, mask_ratio: float, t_steps: int):
        """Store the mutation hyper-parameters used for one generator instance."""

        self.seed = seed
        self.mask_ratio = max(0.0, min(mask_ratio, 0.6))
        self.t_steps = max(1, t_steps)

    def mutate(self, text: str, *, index: int) -> str:
        """Produce one mutated text candidate from a seed sample."""

        tokens = text.split()
        if len(tokens) < 8:
            return text.strip()

        current = list(tokens)
        rng = random.Random(self.seed + index)
        for _ in range(self.t_steps):
            drop_count = max(1, int(len(current) * self.mask_ratio * 0.5))
            keep_indices = sorted(rng.sample(range(len(current)), k=max(1, len(current) - drop_count)))  # Randomly mask part of the text.
            current = [current[idx] for idx in keep_indices]
            if len(current) > 12:
                window = max(8, int(len(current) * 0.8))
                start = rng.randint(0, max(0, len(current) - window))  # Keep a shorter contiguous window for readability.
                current = current[start : start + window]
        return " ".join(current).strip()
