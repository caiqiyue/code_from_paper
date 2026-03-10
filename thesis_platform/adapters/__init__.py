"""Adapter registrations."""

from thesis_platform.adapters.aggregators.dbscan_attn import DBSCANAttnAggregator
from thesis_platform.adapters.aggregators.dbscan_attn_tsgdm import DBSCANAttnTSGDMAggregator
from thesis_platform.adapters.aggregators.none import NoAggregator
from thesis_platform.adapters.aggregators.summarization import SummarizationAggregator
from thesis_platform.adapters.aggregators.uid import UIDAggregator
from thesis_platform.adapters.critics.fedtextgrad_critic import FedTextGradCritic
from thesis_platform.adapters.critics.none import NoCritic
from thesis_platform.adapters.generators.pretext_generator import PretextSeedGenerator
from thesis_platform.adapters.retrievers.knn_retriever import KNNRetriever
from thesis_platform.adapters.retrievers.label_match import LabelMatchRetriever
from thesis_platform.adapters.retrievers.none import NoRetriever
from thesis_platform.adapters.scorers.datainf_scorer import DataInfScorer
from thesis_platform.adapters.scorers.gradmm_scorer import GradMMScorer
from thesis_platform.adapters.scorers.ira_scorer import IRAScorer
from thesis_platform.adapters.scorers.pretext_histogram import PretextHistogramScorer
from thesis_platform.core.registry import register

register("generator", "pretext_seed", PretextSeedGenerator)

register("scorer", "pretext_hist", PretextHistogramScorer)
register("scorer", "datainf", DataInfScorer)
register("scorer", "gradmm", GradMMScorer)
register("scorer", "ira", IRAScorer)

register("retriever", "none", NoRetriever)
register("retriever", "knn", KNNRetriever)
register("retriever", "label_match", LabelMatchRetriever)

register("critic", "none", NoCritic)
register("critic", "fedtextgrad_qwen", FedTextGradCritic)

register("aggregator", "none", NoAggregator)
register("aggregator", "summarization", SummarizationAggregator)
register("aggregator", "uid", UIDAggregator)
register("aggregator", "dbscan_attn", DBSCANAttnAggregator)
register("aggregator", "dbscan_attn_tsgdm", DBSCANAttnTSGDMAggregator)
