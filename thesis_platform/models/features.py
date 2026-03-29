from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FEATURE_MODEL_CACHE: dict[tuple[str, str, int], tuple[Any, Any, Any]] = {}


class BaseFeatureEncoder:
    """Abstract text feature encoder used by the v3 real scorers."""

    backend_name = "base_feature_encoder"

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into dense feature vectors."""

        raise NotImplementedError


@dataclass(slots=True)
class HashingFeatureEncoder(BaseFeatureEncoder):
    """Dependency-light fallback feature encoder."""

    dim: int = 256
    backend_name: str = "hashing_feature"

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with the hashing embedder implementation."""

        from thesis_platform.models.embedding import HashingEmbedder

        return HashingEmbedder(dim=self.dim).embed_texts(texts)


class TransformerFeatureEncoder(BaseFeatureEncoder):
    """Hidden-state mean-pooling encoder backed by a local transformers model."""

    def __init__(self, *, model_path: Path, device: str = "auto", max_length: int = 256):
        self._model_path = model_path
        self._device = device
        self._max_length = max_length
        self.backend_name = f"transformers_feature:{model_path.name}"

        cache_key = (str(model_path), device, max_length)
        if cache_key not in _FEATURE_MODEL_CACHE:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except Exception as exc:  # pragma: no cover - dependency failures are environment-specific
                raise RuntimeError(
                    "transformers and torch are required for transformer feature encoding."
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
            if device != "auto":
                model.to(device)
                load_device = device
            else:
                load_device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(load_device)
            model.eval()
            _FEATURE_MODEL_CACHE[cache_key] = (tokenizer, model, load_device)

        self._tokenizer, self._model, self._load_device = _FEATURE_MODEL_CACHE[cache_key]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Mean-pool the last hidden state under the attention mask."""

        if not texts:
            return []

        import torch
        import torch.nn.functional as functional

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._load_device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self._model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().tolist()


def build_feature_encoder(
    model_name_or_path: str | None,
    repo_root: Path,
    *,
    allow_fallback: bool = False,
    max_length: int = 256,
    device: str = "auto",
) -> BaseFeatureEncoder:
    """Build the best available feature encoder for real scorer execution."""

    if model_name_or_path:
        candidate = (repo_root / model_name_or_path).resolve()
        if candidate.exists():
            try:
                return TransformerFeatureEncoder(model_path=candidate, device=device, max_length=max_length)
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(f"Failed to initialize feature encoder from {candidate}.") from exc
        elif not allow_fallback:
            raise FileNotFoundError(f"Feature model path does not exist: {candidate}")
    if not allow_fallback:
        raise RuntimeError("No feature model configured and hashing fallback is disabled.")
    return HashingFeatureEncoder()
