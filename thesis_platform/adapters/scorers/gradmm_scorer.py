from __future__ import annotations

from thesis_platform.algorithms.scorers.probe_core import _append_bias, build_probe_bundle, sample_features, sample_gradient
from thesis_platform.core.schemas import ScoredSample


class GradMMScorer:
    """Research-mode gradient-mismatch scorer built on the shared client probe."""

    def __init__(self, config, repo_root):
        """Store the GRADMM hyper-parameters."""

        del repo_root
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))
        self.objective = str(config.get("objective", "domain_probe"))
        self.probe_epochs = int(config.get("probe_epochs", 80))
        self.probe_lr = float(config.get("probe_lr", 0.1))
        self.damping = float(config.get("damping", 1e-2))

    def score(self, samples, client_ctx):
        """Score synthetic samples against the client's probe-space reference gradient."""

        objective = self.objective or client_ctx.objective_type
        bundle = build_probe_bundle(
            client_ctx,
            objective=objective,
            probe_epochs=self.probe_epochs,
            probe_lr=self.probe_lr,
            damping=self.damping,
        )
        sample_matrix = sample_features(samples, client_ctx.embedder, objective=objective)
        sample_aug = _append_bias(sample_matrix)
        scores: list[float] = []
        metas: list[dict[str, float | str]] = []
        for feature in sample_aug:
            gradient = sample_gradient(feature, 1.0, bundle.weights)
            grad_distance = float(((gradient - bundle.positive_reference_gradient) ** 2).sum())
            scores.append(grad_distance)
            metas.append(
                {
                    "gradient_distance": grad_distance,
                    "objective": objective,
                    "embedder": bundle.embedder_name,
                }
            )

        client_ctx.probe_state["last_metrics"] = {
            "objective": objective,
            "val_loss_before": bundle.val_loss_before,
            "val_loss_after": bundle.simulate_update_loss(sample_aug),
            "probe_epochs": self.probe_epochs,
            "probe_lr": self.probe_lr,
            "damping": self.damping,
        }
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="gradmm",
                score_direction=self.score_direction,
                meta=dict(meta),
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]
