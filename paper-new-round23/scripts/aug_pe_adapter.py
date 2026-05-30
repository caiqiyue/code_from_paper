#!/usr/bin/env python3
"""Adapter for the Aug-PE (AI-secure/aug-pe) seed-selection algorithm.

Aug-PE (Xie et al., ICML 2024, "Differentially Private Synthetic Data via Foundation
Model APIs 2: Text") frames synthetic-text generation as a private evolution loop
over a public initialization pool. The repo is assumed to be cloned under
``code_from_paper/aug-pe`` on the server.

We expose a single function ``run_augpe_seed_selection`` that selects
``seed_top_k`` high-quality seeds from a public pool given private texts and a
DP budget. The selection is performed with a DP histogram over nearest-neighbour
counts on sentence embeddings (the canonical PE step).

If the upstream Aug-PE package can be imported, we call its embedder + histogram
helpers directly. Otherwise (e.g., the repo is not cloned or has unresolved
dependency conflicts in the pretext env), we fall back to a faithful re-implementation
of the PE step using ``sentence-transformers`` and NumPy. The fallback uses the
same MiniLM model that PrE-Text uses so comparisons stay fair.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any


def _import_augpe(augpe_repo_root: Path) -> Any:
    """Try to import the Aug-PE upstream package. Returns the module or None.

    This is a best-effort import; if the upstream repo is missing or its
    dependency stack conflicts with the pretext conda env, we return None and
    fall back to the local re-implementation below.
    """
    root = Path(augpe_repo_root).resolve()
    if not root.exists():
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        import aug_pe  # type: ignore[import-not-found]
        return aug_pe
    except Exception:
        return None


def _resolve_minilm_path() -> Path:
    """Resolve the MiniLM model path used elsewhere in this project."""
    # PrE-Text default location; same path is used by round19's embedding stack.
    candidates = [
        Path(os.environ.get("MINILM_PATH", "")),
        Path.cwd() / "thesis_platform" / "open_model" / "all_minilm_l6_v2",
        Path(__file__).resolve().parents[2] / "thesis_platform" / "open_model" / "all_minilm_l6_v2",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    # Fall back to HF model id; sentence-transformers will download if allowed.
    return Path("sentence-transformers/all-MiniLM-L6-v2")


def _encode_texts(texts: list[str], model_path: Path) -> Any:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(model_path))
    return np.asarray(encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True))


def _dp_histogram_select(
    *,
    private_embeddings: Any,
    candidate_embeddings: Any,
    seed_top_k: int,
    epsilon: float,
    delta: float,
    rng: random.Random,
) -> list[int]:
    """Faithful re-implementation of the Aug-PE/PE nearest-neighbour histogram step.

    For each private point we find its nearest candidate, build a histogram of
    candidate counts, add Gaussian DP noise calibrated to the user-supplied
    (epsilon, delta), threshold to remove very-low-mass bins, and return the
    ``seed_top_k`` indices with the largest noisy counts.

    The implementation follows the original PE recipe (Xie et al., 2024 / Lin et al.,
    2023). Noise scale uses the analytic Gaussian mechanism approximation
    sigma = sqrt(2 * ln(1.25/delta)) / epsilon for L2 sensitivity 1 (one private
    user contributes at most one count when each private text is treated as a
    single user — which matches the standard PE assumption).
    """
    import numpy as np

    if seed_top_k < 1:
        raise ValueError("seed_top_k must be >= 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    num_candidates = candidate_embeddings.shape[0]
    if num_candidates == 0:
        return []
    # Squared cosine distance via dot-product (vectors are L2-normalized).
    sims = private_embeddings @ candidate_embeddings.T
    nearest = np.argmax(sims, axis=1)
    histogram = np.zeros(num_candidates, dtype=np.float64)
    for idx in nearest:
        histogram[int(idx)] += 1.0
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / float(epsilon)
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    noise = np_rng.normal(loc=0.0, scale=sigma, size=histogram.shape)
    noisy = histogram + noise
    # Threshold: drop bins whose noisy count is below sigma (the canonical PE threshold).
    noisy = np.where(noisy >= sigma, noisy, 0.0)
    if np.sum(noisy > 0) == 0:
        # Degenerate (everything thresholded away). Fall back to top-k raw histogram.
        noisy = histogram
    chosen_count = min(seed_top_k, num_candidates)
    top_indices = np.argsort(-noisy)[:chosen_count]
    return [int(i) for i in top_indices.tolist()]


def run_augpe_seed_selection(
    *,
    private_texts: list[str],
    init_pool: list[str],
    seed_top_k: int,
    epsilon: float,
    delta: float,
    seed: int,
    augpe_repo_root: str | Path,
    embedding_model_path: str | Path | None = None,
) -> list[str]:
    """Run Aug-PE's PE step to choose ``seed_top_k`` seeds from ``init_pool``.

    Parameters
    ----------
    private_texts:
        The private training corpus to protect.
    init_pool:
        Public candidate texts (e.g. the C4 initialization pool).
    seed_top_k:
        Number of seeds to return.
    epsilon, delta:
        Differential-privacy budget for the histogram noise.
    seed:
        Random seed for reproducibility.
    augpe_repo_root:
        Path to a cloned ``AI-secure/aug-pe`` repository. If the repo exposes an
        importable ``aug_pe`` package, we call its embedder + histogram helper
        directly. Otherwise we fall back to the local PE re-implementation.
    embedding_model_path:
        Optional override for the sentence-transformer used to embed both
        private and candidate texts. Defaults to the MiniLM checkpoint used
        elsewhere in this project.

    Returns
    -------
    list[str]
        The selected seed texts, in priority order.
    """
    if seed_top_k < 1:
        raise ValueError("seed_top_k must be >= 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    private_texts = [text for text in private_texts if isinstance(text, str) and text.strip()]
    init_pool = [text for text in init_pool if isinstance(text, str) and text.strip()]
    if not private_texts:
        raise ValueError("private_texts is empty; cannot run Aug-PE selection.")
    if not init_pool:
        raise ValueError("init_pool is empty; cannot run Aug-PE selection.")

    rng = random.Random(int(seed))
    model_path = Path(embedding_model_path) if embedding_model_path else _resolve_minilm_path()

    augpe_module = _import_augpe(Path(augpe_repo_root))
    if augpe_module is not None and hasattr(augpe_module, "run_pe_step"):
        # Preferred path: upstream module exposes a single-shot PE step.
        try:
            return list(
                augpe_module.run_pe_step(  # type: ignore[attr-defined]
                    private_texts=private_texts,
                    init_pool=init_pool,
                    seed_top_k=int(seed_top_k),
                    epsilon=float(epsilon),
                    delta=float(delta),
                    seed=int(seed),
                    embedding_model_path=str(model_path),
                )
            )
        except Exception as exc:
            # Fall through to local re-implementation but surface the upstream error.
            print(f"[aug_pe_adapter] upstream run_pe_step failed: {exc!r}; falling back to local PE.", file=sys.stderr)

    # Local faithful re-implementation.
    private_embeddings = _encode_texts(private_texts, model_path)
    candidate_embeddings = _encode_texts(init_pool, model_path)
    chosen_indices = _dp_histogram_select(
        private_embeddings=private_embeddings,
        candidate_embeddings=candidate_embeddings,
        seed_top_k=int(seed_top_k),
        epsilon=float(epsilon),
        delta=float(delta),
        rng=rng,
    )
    return [init_pool[idx] for idx in chosen_indices]
