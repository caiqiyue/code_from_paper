from __future__ import annotations

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

    def __init__(self, model_path: Path):
        """Load a local sentence-transformer model from disk."""

        from sentence_transformers import SentenceTransformer

        self._model_path = model_path
        self._model = SentenceTransformer(str(model_path))
        self.backend_name = f"sentence_transformer:{model_path.name}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with a real sentence-transformer model."""

        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


def _is_sentence_transformer_dir(path: Path) -> bool:
    """Return true when a directory looks like a saved sentence-transformer model."""

    return path.is_dir() and (path / "modules.json").exists()


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
) -> BaseEmbedder:
    """Build the best available embedding backend for the current environment."""

    if model_name_or_path:
        candidate = (repo_root / model_name_or_path).resolve()
        if candidate.exists():
            try:
                return SentenceTransformerEmbedder(resolve_sentence_transformer_path(candidate))
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
