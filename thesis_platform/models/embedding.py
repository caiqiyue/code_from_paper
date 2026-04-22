from __future__ import annotations

import gc
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Split text into normalized alphanumeric tokens."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BaseEmbedder:
    """Abstract embedding backend used by retrievers and scorers."""

    backend_name = "base"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into dense vectors."""

        raise NotImplementedError


@dataclass(slots=True)
class HashingEmbedder(BaseEmbedder):
    """Dependency-free fallback embedder based on token hashing."""

    dim: int = 256
    backend_name: str = "hashing"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with a simple normalized hashing trick."""

        vectors = [[0.0] * self.dim for _ in texts]
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
                index = int(digest, 16) % self.dim
                vectors[row][index] += 1.0
            norm = math.sqrt(sum(value * value for value in vectors[row]))
            if norm > 0:
                vectors[row] = [value / norm for value in vectors[row]]
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer wrapper used when the local model is available."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        """Store the local sentence-transformer path and load on first use."""

        self._model_path = model_path
        self._device = device
        self._model = None
        self.backend_name = f"sentence_transformer:{model_path.name}"

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        _ensure_transformers_sentence_transformer_compatibility()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(self._model_path))
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with a real sentence-transformer model."""

        model = self._ensure_loaded()
        encode_device = None if self._device == "auto" else self._device
        embeddings = model.encode(texts, normalize_embeddings=True, device=encode_device)
        return [list(map(float, row)) for row in embeddings]

    def release(self) -> None:
        model = self._model
        self._model = None
        if model is None:
            return
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _is_sentence_transformer_dir(path: Path) -> bool:
    """Return true when a directory looks like a saved sentence-transformer model."""

    return path.is_dir() and (path / "modules.json").exists()


def _ensure_transformers_sentence_transformer_compatibility() -> None:
    """Backfill the transformers symbol expected by the installed peft package.

    The caiqiyue-vllm environment ships a transformers build that predates
    ``EncoderDecoderCache`` while the installed peft/sentence-transformers
    stack imports it unconditionally. Exposing an alias keeps the full
    retrieval/critique path working without changing the environment.
    """

    try:
        import transformers
    except Exception:
        return
    if hasattr(transformers, "EncoderDecoderCache"):
        return
    cache_cls = getattr(transformers, "Cache", None)
    if cache_cls is None:
        return
    transformers.EncoderDecoderCache = cache_cls


def resolve_sentence_transformer_path(candidate: Path) -> Path:
    """Resolve a direct model path or a Hugging Face cache root to a loadable model directory."""

    if _is_sentence_transformer_dir(candidate):
        return candidate

    snapshots_dir = candidate / "snapshots"
    refs_dir = candidate / "refs"
    if not snapshots_dir.is_dir():
        return candidate

    preferred_snapshots: list[Path] = []
    main_ref = refs_dir / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if revision:
            preferred_snapshots.append(snapshots_dir / revision)

    snapshot_dirs = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    preferred_snapshots.extend(snapshot_dirs)

    for snapshot in preferred_snapshots:
        if _is_sentence_transformer_dir(snapshot):
            return snapshot
    return candidate


def build_embedder(
    model_name_or_path: str | None,
    repo_root: Path,
    *,
    allow_fallback: bool = True,
    device: str = "cpu",
) -> BaseEmbedder:
    """Build the best available embedding backend for the current environment."""

    if model_name_or_path:
        # On Windows, repo_root may have garbled non-ASCII chars in its Path string
        # representation, causing join/resolve to produce wrong paths.
        # Use cwd-based resolution: find the correct project root by navigating up.
        import os
        raw_candidate = repo_root / model_name_or_path
        # Try using os.getcwd() to find the correct base, then build the correct path
        cwd = os.getcwd()
        # Navigate up from cwd to find the project root containing thesis_platform
        candidate_dir = Path(cwd)
        candidate = None
        found = False
        for _ in range(10):
            if (candidate_dir / "thesis_platform").is_dir():
                # Found the project root - use it as base
                correct_candidate = candidate_dir / model_name_or_path
                if correct_candidate.exists():
                    candidate = correct_candidate.resolve()
                    found = True
                    break
            parent = candidate_dir.parent
            if parent == candidate_dir:
                break
            candidate_dir = parent
        if not found:
            candidate = raw_candidate.resolve()
        if candidate and candidate.exists():
            try:
                return SentenceTransformerEmbedder(
                    resolve_sentence_transformer_path(candidate),
                    device=device,
                )
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(
                        f"Failed to initialize SentenceTransformer embedder from {candidate}. "
                        "Install sentence-transformers in the active environment or allow hashing fallback."
                    ) from exc
        elif not allow_fallback:
            raise FileNotFoundError(f"Embedding model path does not exist: {candidate}")
    if not allow_fallback:
        raise RuntimeError("No embedding model configured and hashing fallback is disabled.")
    return HashingEmbedder()
