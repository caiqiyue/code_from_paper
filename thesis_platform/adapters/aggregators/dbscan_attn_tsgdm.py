from __future__ import annotations

from thesis_platform.algorithms.aggregators.dbscan_core import aggregate_dbscan_critiques
from thesis_platform.models.embedding import build_embedder


class DBSCANAttnTSGDMAggregator:
    """DBSCAN-Attn aggregator with cross-round momentum memory."""

    def __init__(self, config, repo_root):
        """Build the embedder and memory hyper-parameters used by TSGD-M."""

        self.max_rules = int(config.get("max_rules", 5))
        self.cluster_eps = float(config.get("cluster_eps", 0.35))
        self.cluster_min_samples = int(config.get("cluster_min_samples", 2))
        self.momentum_beta = float(config.get("momentum_beta", 0.7))
        self.prototype_cluster_method = config.get("prototype_cluster_method", "dbscan")
        embedding_model = config.get("embedding_model")
        if not embedding_model:
            raise ValueError("dbscan_attn_tsgdm requires aggregator.embedding_model.")
        self.embedder = build_embedder(
            embedding_model,
            repo_root,
            allow_fallback=bool(config.get("allow_hashing_fallback", False)),
        )

    def aggregate(self, client_critiques, server_ctx):
        """Aggregate critiques with semantic clustering and persistent memory."""

        prompt_update, updated_memory = aggregate_dbscan_critiques(
            client_critiques,
            round_id=len(server_ctx.prompt_history) - 1,
            max_rules=self.max_rules,
            embedder=self.embedder,
            text_backend=server_ctx.text_backend,
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            use_memory=True,
            memory=server_ctx.aggregation_memory,
            momentum_beta=self.momentum_beta,
            base_prompt=server_ctx.base_prompt,
            prototype_feedbacks=list(server_ctx.prototype_feedbacks),
            personalized_mix_ratio=server_ctx.routing_state.get("personalized_mix_ratio"),
            prototype_cluster_method=self.prototype_cluster_method,
        )
        server_ctx.aggregation_memory = updated_memory
        return prompt_update
