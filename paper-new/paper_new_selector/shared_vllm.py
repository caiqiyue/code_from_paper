from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import SharedSessionContract


@dataclass(slots=True)
class SharedVllmSession:
    """Lazy shared Stage 1/Stage 2 vLLM session wrapper."""

    backend: Any
    contract: SharedSessionContract

    def ensure_session(self) -> tuple[Any, Any]:
        return self.backend.ensure_session()

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float | None = None,
    ) -> list[str]:
        return self.backend.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.contract.to_dict()


def build_shared_vllm_session(text_backend: Any) -> SharedVllmSession | None:
    """Create a lazy shared-session wrapper when the backend supports vLLM reuse."""

    if not hasattr(text_backend, "ensure_session") or not hasattr(text_backend, "generate_batch"):
        return None
    contract = SharedSessionContract(
        llm_backend="vllm",
        model_path=str(getattr(text_backend, "_model_path")),
        max_model_len=int(getattr(text_backend, "_max_model_len")),
        gpu_memory_utilization=float(getattr(text_backend, "_gpu_memory_utilization")),
        tensor_parallel_size=int(getattr(text_backend, "_tensor_parallel_size")),
        enforce_eager=bool(getattr(text_backend, "_enforce_eager")),
    )
    return SharedVllmSession(
        backend=text_backend,
        contract=contract,
    )
