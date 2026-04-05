from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

class BaseTextBackend:
    """Abstract text generation backend."""

    backend_name = "base"

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
        """Generate text from a prompt."""

        raise NotImplementedError

    def negative_log_likelihood(self, prompt: str, completion: str) -> float:
        """Score a completion under the current model; lower is better."""

        raise NotImplementedError


@dataclass(slots=True)
class HeuristicTextBackend(BaseTextBackend):
    """Fallback backend kept for MVP compatibility only."""

    backend_name: str = "heuristic"

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
        """Return the prompt itself as a deterministic fallback output."""

        del max_new_tokens, temperature
        return prompt.strip()

    def negative_log_likelihood(self, prompt: str, completion: str) -> float:
        """Use a simple lexical-overlap heuristic when no LM backend is available."""

        prompt_tokens = {token.lower() for token in prompt.split()}
        completion_tokens = [token.lower() for token in completion.split()]
        if not completion_tokens:
            return 0.0
        overlap = sum(1 for token in completion_tokens if token in prompt_tokens)
        return float(len(completion_tokens) - overlap) / float(len(completion_tokens))


@dataclass(slots=True)
class MockTextBackend(BaseTextBackend):
    """Deterministic backend used by tests and light-weight validation configs."""

    role: str = "mock"
    backend_name: str = "mock"

    def _digest(self, prompt: str) -> str:
        return hashlib.sha1(f"{self.role}:{prompt}".encode("utf-8")).hexdigest()[:12]

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
        """Return deterministic JSON or text based on the requested task."""

        del max_new_tokens, temperature
        lower = prompt.lower()
        digest = self._digest(prompt)
        if "return json" in lower and "memory_summary" in lower:
            payload = {
                "rules": [
                    f"Preserve high-signal guidance {digest[:4]}.",
                    f"Retain repeated domain constraints {digest[4:8]}.",
                ],
                "memory_summary": f"Carry forward cluster memory {digest[8:12]}.",
            }
            return json.dumps(payload, ensure_ascii=False)
        if "return json" in lower and "rules" in lower:
            payload = {
                "rules": [
                    f"Increase domain-specific precision for {digest[:4]}.",
                    f"Remove generic phrasing associated with {digest[4:8]}.",
                ]
            }
            return json.dumps(payload, ensure_ascii=False)
        return f"<sample>mock-{self.role}-{digest}</sample>"

    def negative_log_likelihood(self, prompt: str, completion: str) -> float:
        """Assign lower loss when completion tokens overlap prompt tokens."""

        prompt_tokens = {token.lower() for token in prompt.split()}
        completion_tokens = [token.lower() for token in completion.split()]
        if not completion_tokens:
            return 0.0
        overlap = sum(1 for token in completion_tokens if token in prompt_tokens)
        return float(len(completion_tokens) - overlap + 1) / float(len(completion_tokens) + 1)


class TransformersTextBackend(BaseTextBackend):
    """Transformers-backed causal LM used by the research-mode pipeline."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        use_chat_template: bool = False,
        use_fast: bool | None = None,
    ):
        """Store one local causal LM path and load it lazily on first use."""

        self._model_path = model_path
        self._device = device
        self._dtype = dtype
        self._default_temperature = temperature
        self._default_max_new_tokens = max_new_tokens
        self._use_chat_template = use_chat_template
        self._use_fast = use_fast
        self._tokenizer = None
        self._model = None
        self._load_device = None
        self.backend_name = f"transformers:{model_path.name}"

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None and self._load_device is not None:
            return self._tokenizer, self._model, self._load_device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised in dependency-missing environments
            raise RuntimeError(
                "transformers/torch are required for research-mode text backends. "
                "Install thesis_platform/requirements.txt in the active environment."
            ) from exc

        torch_dtype = self._resolve_dtype(torch, self._dtype, self._device)
        tokenizer = self._load_tokenizer(AutoTokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "torch_dtype": torch_dtype,
        }
        if self._device == "auto":
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(str(self._model_path), **load_kwargs)
            load_device = "auto"
        else:
            model = AutoModelForCausalLM.from_pretrained(str(self._model_path), **load_kwargs)
            model.to(self._device)
            load_device = self._device
        model.eval()
        self._tokenizer, self._model, self._load_device = tokenizer, model, load_device
        return tokenizer, model, load_device

    def _load_tokenizer(self, auto_tokenizer_cls: Any):
        if self._use_fast is None:
            try:
                return auto_tokenizer_cls.from_pretrained(str(self._model_path), local_files_only=True)
            except Exception:
                return auto_tokenizer_cls.from_pretrained(
                    str(self._model_path),
                    local_files_only=True,
                    use_fast=False,
                )
        try:
            return auto_tokenizer_cls.from_pretrained(
                str(self._model_path),
                local_files_only=True,
                use_fast=self._use_fast,
            )
        except Exception:
            if self._use_fast:
                return auto_tokenizer_cls.from_pretrained(
                    str(self._model_path),
                    local_files_only=True,
                    use_fast=False,
                )
            raise

    @staticmethod
    def _resolve_dtype(torch: Any, dtype: str, device: str) -> Any:
        if dtype == "float32":
            return torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if device in {"cuda", "auto"} and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _format_prompt(self, prompt: str) -> str:
        tokenizer, _, _ = self._ensure_loaded()
        if self._use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _resolve_input_device(self) -> Any:
        _, model, load_device = self._ensure_loaded()
        if load_device != "auto":
            return load_device
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return None

    def _tokenize(self, text: str) -> Any:
        tokenizer, _, _ = self._ensure_loaded()
        inputs = tokenizer(text, return_tensors="pt")
        input_device = self._resolve_input_device()
        if input_device is not None:
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
        return inputs

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
        """Generate text from a prompt with greedy or sampled decoding."""

        import torch

        tokenizer, model, _ = self._ensure_loaded()
        formatted_prompt = self._format_prompt(prompt)
        inputs = self._tokenize(formatted_prompt)
        effective_temperature = temperature if temperature is not None else self._default_temperature
        do_sample = effective_temperature > 0
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens or self._default_max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = effective_temperature
        generation = model.generate(**generate_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            new_tokens = generation[0][prompt_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def negative_log_likelihood(self, prompt: str, completion: str) -> float:
        """Compute the mean token-level NLL of completion conditioned on prompt."""

        import torch

        if not completion:
            return 0.0
        tokenizer, model, _ = self._ensure_loaded()
        formatted_prompt = self._format_prompt(prompt)
        prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([prompt_ids["input_ids"], completion_ids["input_ids"]], dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, : prompt_ids["input_ids"].shape[1]] = -100
        input_device = self._resolve_input_device()
        if input_device is not None:
            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)
            labels = labels.to(input_device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return float(outputs.loss.detach().cpu().item())

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


def build_text_backend(
    engine_or_config: str | dict[str, Any] | None,
    model_name_or_path: str | None = None,
    repo_root: Path | None = None,
    **overrides: Any,
) -> BaseTextBackend:
    """Select a text backend based on config and local model availability."""

    repo_root_path = Path(repo_root) if repo_root is not None else Path(".")
    if isinstance(engine_or_config, dict):
        config = dict(engine_or_config)
        config.update(overrides)
    else:
        config = {"engine": engine_or_config, "model_name_or_path": model_name_or_path}
        config.update(overrides)

    engine = str(config.get("engine", "heuristic") or "heuristic").lower()
    role = str(config.get("role", "text"))

    if engine == "heuristic":
        return HeuristicTextBackend()
    if engine == "mock":
        return MockTextBackend(role=role)

    raw_path = config.get("model_name_or_path")
    if not raw_path:
        raise ValueError(f"Text backend engine '{engine}' requires model_name_or_path.")
    model_path = (repo_root_path / str(raw_path)).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Configured model path does not exist: {model_path}")
    if engine != "transformers":
        raise ValueError(f"Unsupported text backend engine '{engine}'.")
    return TransformersTextBackend(
        model_path=model_path,
        device=str(config.get("device", "auto")),
        dtype=str(config.get("dtype", "auto")),
        temperature=float(config.get("temperature", 0.2)),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        use_chat_template=bool(config.get("use_chat_template", False)),
        use_fast=(
            bool(config.get("use_fast"))
            if "use_fast" in config
            else None
        ),
    )
