from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, str]] = {}


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
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """Load a causal LM lazily and keep it cached by model/device/dtype."""

        self._model_path = model_path
        self._device = device
        self._dtype = dtype
        self._default_temperature = temperature
        self._default_max_new_tokens = max_new_tokens
        self._use_chat_template = use_chat_template
        self._use_fast = use_fast
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self.backend_name = f"transformers:{model_path.name}"

        cache_key = (str(model_path), device, dtype, str(load_in_4bit), str(load_in_8bit))
        if cache_key not in _MODEL_CACHE:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:  # pragma: no cover - exercised in dependency-missing environments
                raise RuntimeError(
                    "transformers/torch are required for research-mode text backends. "
                    "Install thesis_platform/requirements.txt in the active environment."
                ) from exc

            torch_dtype = self._resolve_dtype(torch, dtype, device)
            if self._use_fast is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(model_path),
                        use_fast=self._use_fast,
                    )
                except Exception:
                    if self._use_fast:
                        tokenizer = AutoTokenizer.from_pretrained(
                            str(model_path),
                            use_fast=False,
                        )
                    else:
                        raise
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs = dict(
                low_cpu_mem_usage=True,
            )
            if self._load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    compute_dtype = torch_dtype if torch_dtype != torch.float32 else torch.float16
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    torch_dtype = torch.float16  # force to float16 for 4bit
                except Exception as e:
                    raise RuntimeError(f"4-bit quantization requested but failed to configure: {e}") from e
            elif self._load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    torch_dtype = torch.float16
                except Exception as e:
                    raise RuntimeError(f"8-bit quantization requested but failed to configure: {e}") from e

            if device == "auto":
                load_kwargs["dtype"] = torch_dtype
                load_kwargs["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
                load_device = "auto"
            else:
                load_kwargs["dtype"] = torch_dtype
                model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
                model.to(device)
                load_device = device
            model.eval()
            _MODEL_CACHE[cache_key] = (tokenizer, model, load_device)

        self._tokenizer, self._model, self._load_device = _MODEL_CACHE[cache_key]

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
        if self._use_chat_template and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _resolve_input_device(self) -> Any:
        if self._load_device != "auto":
            return self._load_device
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return None

    def _tokenize(self, text: str) -> Any:
        inputs = self._tokenizer(text, return_tensors="pt")
        input_device = self._resolve_input_device()
        if input_device is not None:
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
        return inputs

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float | None = None) -> str:
        """Generate text from a prompt with greedy or sampled decoding."""

        import torch

        formatted_prompt = self._format_prompt(prompt)
        inputs = self._tokenize(formatted_prompt)
        effective_temperature = temperature if temperature is not None else self._default_temperature
        do_sample = effective_temperature > 0
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens or self._default_max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = effective_temperature
        generation = self._model.generate(**generate_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            new_tokens = generation[0][prompt_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def negative_log_likelihood(self, prompt: str, completion: str) -> float:
        """Compute the mean token-level NLL of completion conditioned on prompt."""

        import torch

        if not completion:
            return 0.0
        formatted_prompt = self._format_prompt(prompt)
        prompt_ids = self._tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        completion_ids = self._tokenizer(completion, return_tensors="pt", add_special_tokens=False)
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
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return float(outputs.loss.detach().cpu().item())


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
        load_in_4bit=bool(config.get("load_in_4bit", False)),
        load_in_8bit=bool(config.get("load_in_8bit", False)),
    )
