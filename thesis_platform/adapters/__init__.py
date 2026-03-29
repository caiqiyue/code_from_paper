"""Adapter registrations."""

import logging

from thesis_platform.adapters.aggregators.dbscan_attn import DBSCANAttnAggregator
from thesis_platform.adapters.aggregators.dbscan_attn_tsgdm import DBSCANAttnTSGDMAggregator
from thesis_platform.adapters.aggregators.none import NoAggregator
from thesis_platform.adapters.aggregators.summarization import SummarizationAggregator
from thesis_platform.adapters.aggregators.uid import UIDAggregator
from thesis_platform.adapters.aggregators.uid_llm import UIDLLMAggregator
from thesis_platform.adapters.critics.fedtextgrad_critic import FedTextGradCritic
from thesis_platform.adapters.critics.fedtextgrad_llm import FedTextGradLLMCritic
from thesis_platform.adapters.critics.fedtextgrad_qwen_critic import FedTextGradQwenCritic
from thesis_platform.adapters.critics.none import NoCritic
from thesis_platform.adapters.generators.pretext_prompt_generator import PretextPromptLLMGenerator
from thesis_platform.adapters.generators.pretext_generator import PretextSeedGenerator
from thesis_platform.adapters.retrievers.knn_retriever import KNNRetriever
from thesis_platform.adapters.retrievers.label_match import LabelMatchRetriever
from thesis_platform.adapters.retrievers.none import NoRetriever
from thesis_platform.adapters.scorers.datainf_scorer import DataInfScorer
from thesis_platform.adapters.scorers.datainf_real_scorer import DataInfRealScorer
from thesis_platform.adapters.scorers.gradmm_scorer import GradMMScorer
from thesis_platform.adapters.scorers.gradmm_real_scorer import GradMMRealScorer
from thesis_platform.adapters.scorers.ira_scorer import IRAScorer
from thesis_platform.adapters.scorers.pretext_histogram import PretextHistogramScorer
from thesis_platform.adapters.scorers.datainf_paper_scorer import DataInfPaperScorer
from thesis_platform.adapters.scorers.gradmm_paper_scorer import GradMMPaperScorer
from thesis_platform.core.registry import register

logger = logging.getLogger(__name__)


def _try_import_optional(factory_path: str, class_name: str):
    """Import one optional adapter without breaking the whole registry."""

    try:
        module = __import__(factory_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as exc:  # pragma: no cover - exercised in dependency-light environments
        logger.debug("Skipping optional adapter %s.%s: %s", factory_path, class_name, exc)
        return None

register("generator", "pretext_seed", PretextSeedGenerator)
register("generator", "pretext_prompt_llm", PretextPromptLLMGenerator)

register("scorer", "pretext_hist", PretextHistogramScorer)
register("scorer", "datainf", DataInfScorer)
register("scorer", "datainf_real", DataInfRealScorer)
register("scorer", "datainf_paper", DataInfPaperScorer)
register("scorer", "gradmm", GradMMScorer)
register("scorer", "gradmm_real", GradMMRealScorer)
register("scorer", "gradmm_paper", GradMMPaperScorer)
register("scorer", "ira", IRAScorer)

DataInfLoRARealScorer = _try_import_optional(
    "thesis_platform.adapters.scorers.datainf_lora_scorer",
    "DataInfRealScorer",
)
if DataInfLoRARealScorer is not None:
    register("scorer", "datainf_lora", DataInfLoRARealScorer)

GradMMLORARealScorer = _try_import_optional(
    "thesis_platform.adapters.scorers.gradmm_lora_scorer",
    "GradMMRealScorer",
)
if GradMMLORARealScorer is not None:
    register("scorer", "gradmm_lora", GradMMLORARealScorer)

register("retriever", "none", NoRetriever)
register("retriever", "knn", KNNRetriever)
register("retriever", "label_match", LabelMatchRetriever)

register("critic", "none", NoCritic)
register("critic", "fedtextgrad_qwen", FedTextGradCritic)
register("critic", "fedtextgrad_qwen_model", FedTextGradQwenCritic)
register("critic", "fedtextgrad_llm", FedTextGradLLMCritic)

register("aggregator", "none", NoAggregator)
register("aggregator", "summarization", SummarizationAggregator)
register("aggregator", "uid", UIDAggregator)
register("aggregator", "uid_llm", UIDLLMAggregator)
register("aggregator", "dbscan_attn", DBSCANAttnAggregator)
register("aggregator", "dbscan_attn_tsgdm", DBSCANAttnTSGDMAggregator)
