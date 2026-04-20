from __future__ import annotations

from thesis_platform.algorithms.aggregators.dbscan_core import aggregate_dbscan_critiques
from thesis_platform.models.embedding import build_embedder


class DBSCANAttnAggregator:
    """Cluster critique rules semantically and rank them by support and severity."""

    def __init__(self, config, repo_root):
        """Build the embedder used for clustering critique rules."""

        self.max_rules = int(config.get("max_rules", 5))
        self.cluster_eps = float(config.get("cluster_eps", 0.35))
        self.cluster_min_samples = int(config.get("cluster_min_samples", 2))
        self.prototype_cluster_method = config.get("prototype_cluster_method", "dbscan")
        embedding_model = config.get("embedding_model")
        if not embedding_model:
            raise ValueError("dbscan_attn requires aggregator.embedding_model.")
        self.embedder = build_embedder(
            embedding_model,
            repo_root,
            allow_fallback=bool(config.get("allow_hashing_fallback", False)),
            device=str(config.get("device", "cpu")),
        )

    def aggregate(self, client_critiques, server_ctx):
        """Aggregate critiques through embedding clustering without momentum memory."""

        prompt_update, _ = aggregate_dbscan_critiques(
            client_critiques,
            round_id=len(server_ctx.prompt_history) - 1,
            max_rules=self.max_rules,
            embedder=self.embedder,
            text_backend=server_ctx.text_backend,
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            use_memory=False,
            memory=server_ctx.aggregation_memory,
            base_prompt=server_ctx.base_prompt,
            prototype_feedbacks=list(server_ctx.prototype_feedbacks),
            personalized_mix_ratio=server_ctx.routing_state.get("personalized_mix_ratio"),
            prototype_cluster_method=self.prototype_cluster_method,
        )
        return prompt_update
