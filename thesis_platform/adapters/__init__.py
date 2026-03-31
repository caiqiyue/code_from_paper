"""Adapter registrations."""

import logging

from thesis_platform.adapters.aggregators.none import NoAggregator
from thesis_platform.adapters.aggregators.summarization import SummarizationAggregator
from thesis_platform.adapters.aggregators.uid import UIDAggregator
from thesis_platform.adapters.critics.fedtextgrad_critic import FedTextGradCritic
from thesis_platform.adapters.critics.fedtextgrad_llm import FedTextGradLLMCritic
from thesis_platform.adapters.critics.fedtextgrad_qwen_critic import FedTextGradQwenCritic
from thesis_platform.adapters.critics.none import NoCritic
from thesis_platform.adapters.generators.pretext_prompt_generator import PretextPromptLLMGenerator
from thesis_platform.adapters.generators.pretext_generator import PretextSeedGenerator
from thesis_platform.adapters.retrievers.knn_retriever import KNNRetriever
from thesis_platform.adapters.retrievers.label_match import LabelMatchRetriever
from thesis_platform.adapters.retrievers.none import NoRetriever
from thesis_platform.adapters.scorers.ira_scorer import IRAScorer
from thesis_platform.adapters.scorers.pretext_histogram import PretextHistogramScorer
from thesis_platform.core.registry import register

logger = logging.getLogger(__name__)


def _try_import_optional(factory_path: str, class_name: str):
    """Import one optional adapter without breaking the whole registry."""

    try:
        module = __import__(factory_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as exc:  # pragma: no cover - exercised in dependency-light environments
        logger.debug("Skipping optional adapter %s.%s: %s", factory_path, class_name, exc)
        return exc


def _missing_dependency_factory(kind: str, name: str, factory_path: str, exc: Exception):
    """Build one registry entry that fails with the real import error on instantiation."""

    def _raise_missing_dependency(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            f"{kind} adapter '{name}' is unavailable because importing {factory_path} failed: {exc}"
        ) from exc

    return _raise_missing_dependency


def _register_optional(kind: str, name: str, factory_path: str, class_name: str) -> None:
    """Register one optional adapter or a precise failure stub when dependencies are missing."""

    factory_or_exc = _try_import_optional(factory_path, class_name)
    if isinstance(factory_or_exc, Exception):
        register(
            kind,
            name,
            _missing_dependency_factory(kind, name, f"{factory_path}.{class_name}", factory_or_exc),
        )
        return
    if factory_or_exc is not None:
        register(kind, name, factory_or_exc)

register("generator", "pretext_seed", PretextSeedGenerator)
register("generator", "pretext_prompt_llm", PretextPromptLLMGenerator)

register("scorer", "pretext_hist", PretextHistogramScorer)
register("scorer", "ira", IRAScorer)

_register_optional("scorer", "datainf", "thesis_platform.adapters.scorers.datainf_scorer", "DataInfScorer")
_register_optional("scorer", "datainf_real", "thesis_platform.adapters.scorers.datainf_real_scorer", "DataInfRealScorer")
_register_optional("scorer", "datainf_paper", "thesis_platform.adapters.scorers.datainf_paper_scorer", "DataInfPaperScorer")
_register_optional("scorer", "gradmm", "thesis_platform.adapters.scorers.gradmm_scorer", "GradMMScorer")
_register_optional("scorer", "gradmm_real", "thesis_platform.adapters.scorers.gradmm_real_scorer", "GradMMRealScorer")
_register_optional("scorer", "gradmm_paper", "thesis_platform.adapters.scorers.gradmm_paper_scorer", "GradMMPaperScorer")
_register_optional("scorer", "datainf_lora", "thesis_platform.adapters.scorers.datainf_lora_scorer", "DataInfRealScorer")
_register_optional("scorer", "gradmm_lora", "thesis_platform.adapters.scorers.gradmm_lora_scorer", "GradMMRealScorer")

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

_register_optional("aggregator", "uid_llm", "thesis_platform.adapters.aggregators.uid_llm", "UIDLLMAggregator")
_register_optional("aggregator", "dbscan_attn", "thesis_platform.adapters.aggregators.dbscan_attn", "DBSCANAttnAggregator")
_register_optional(
    "aggregator",
    "dbscan_attn_tsgdm",
    "thesis_platform.adapters.aggregators.dbscan_attn_tsgdm",
    "DBSCANAttnTSGDMAggregator",
)
