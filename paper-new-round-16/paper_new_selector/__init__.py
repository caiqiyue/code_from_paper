"""Paper-new selector package for the single-node PrE-Text innovation line."""

from .boundary import build_boundary_state
from .eval_bridge import prepare_eval_runtime, run_eval
from .generator_bridge import build_candidate_generator
from .pipeline import run_pipeline
from .selector import greedy_select_candidates

__all__ = [
    "build_boundary_state",
    "build_candidate_generator",
    "greedy_select_candidates",
    "prepare_eval_runtime",
    "run_eval",
    "run_pipeline",
]
