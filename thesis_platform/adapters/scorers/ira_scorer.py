from __future__ import annotations

from thesis_platform.algorithms.scorers.ira_core import compute_ira_scores
from thesis_platform.core.schemas import ScoredSample


class IRAScorer:
    """Instruction-response alignment scorer backed by the client-side LLM."""

    def __init__(self, config, repo_root):
        """Store the scoring direction for the IRA adapter."""

        del repo_root
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))

    def score(self, samples, client_ctx):
        """Score structured instruction-response samples with conditional LM losses."""

        if client_ctx.objective_type != "pair_alignment":
            raise ValueError("IRA is only valid for instruction_response experiments with objective=pair_alignment.")
        if client_ctx.text_backend is None:
            raise ValueError("IRA requires a client text backend.")
        scores, metas = compute_ira_scores(samples, text_backend=client_ctx.text_backend)
        client_ctx.probe_state["last_metrics"] = {
            "objective": "pair_alignment",
            "val_loss_before": 0.0,
            "val_loss_after": 0.0,
            "backend_name": getattr(client_ctx.text_backend, "backend_name", type(client_ctx.text_backend).__name__),
        }
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="ira",
                score_direction=self.score_direction,
                meta=dict(meta),
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]
