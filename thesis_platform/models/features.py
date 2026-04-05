from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        self._tokenizer = None
        self._model = None
        self._load_device = None
        self.backend_name = f"transformers_feature:{model_path.name}"

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None and self._load_device is not None:
            return self._tokenizer, self._model, self._load_device
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency failures are environment-specific
            raise RuntimeError(
                "transformers and torch are required for transformer feature encoding."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(self._model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(self._model_path), local_files_only=True)
        if self._device != "auto":
            model.to(self._device)
            load_device = self._device
        else:
            load_device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(load_device)
        model.eval()
        self._tokenizer, self._model, self._load_device = tokenizer, model, load_device
        return tokenizer, model, load_device

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Mean-pool the last hidden state under the attention mask."""

        if not texts:
            return []

        import torch
        import torch.nn.functional as functional

        tokenizer, model, load_device = self._ensure_loaded()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(load_device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().tolist()

    def release(self) -> None:
        tokenizer = self._tokenizer
        model = self._model
        self._tokenizer = None
        self._model = None
        self._load_device = None
        del tokenizer, model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


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
