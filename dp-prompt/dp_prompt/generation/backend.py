from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def build_generation_request(prompt: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "temperature": cfg.get("temperature", 1.0),
        "max_new_tokens": cfg.get("max_generated_tokens", 128),
        "stop_sequences": list(cfg.get("stop_sequences", [])),
        "logits_clipping": {
            "enabled": bool(cfg.get("logits_clipping", {}).get("enabled", False)),
            "lower_bound": cfg.get("logits_clipping", {}).get("lower_bound"),
            "upper_bound": cfg.get("logits_clipping", {}).get("upper_bound"),
        },
        "do_sample": True,
        "top_k": 0,
        "top_p": 1.0,
    }


def _trim_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    trimmed = text
    for stop_sequence in stop_sequences:
        if stop_sequence and stop_sequence in trimmed:
            trimmed = trimmed.split(stop_sequence, 1)[0]
    return trimmed.strip()


@dataclass
class LocalTransformersGenerator:
    model_path: str
    tokenizer_path: str | None
    torch_dtype: str | None
    batch_size: int
    max_context_length: int | None = None

    _tokenizer: Any = None
    _model: Any = None

    @staticmethod
    def _normalize_optional_path(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or stripped.startswith("${"):
            return None
        return stripped

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> "LocalTransformersGenerator":
        return cls(
            model_path=model_cfg["model_path"],
            tokenizer_path=cls._normalize_optional_path(model_cfg.get("tokenizer_path")),
            torch_dtype=model_cfg.get("torch_dtype"),
            batch_size=int(model_cfg.get("batch_size", 4)),
            max_context_length=model_cfg.get("max_context_length"),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Running the local generation backend requires `transformers` and `torch`."
            ) from exc

        tokenizer_path = self.tokenizer_path or self.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dtype = getattr(torch, self.torch_dtype) if self.torch_dtype else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=dtype,
        )

    def _build_logits_processor(self, request_cfg: dict[str, Any]) -> Any:
        clipping_cfg = request_cfg.get("logits_clipping", {})
        if not clipping_cfg.get("enabled", False):
            return None
        lower_bound = clipping_cfg.get("lower_bound")
        upper_bound = clipping_cfg.get("upper_bound")
        if lower_bound is None or upper_bound is None:
            raise ValueError(
                "logits_clipping is enabled but lower_bound/upper_bound are missing."
            )
        try:
            from transformers import LogitsProcessor, LogitsProcessorList
        except ImportError as exc:
            raise ImportError(
                "logits clipping requires `transformers` to provide a logits processor."
            ) from exc

        class StaticLogitsClippingProcessor(LogitsProcessor):
            def __init__(self, lower: float, upper: float):
                self.lower = lower
                self.upper = upper

            def __call__(self, input_ids, scores):
                return scores.clamp(min=self.lower, max=self.upper)

        return LogitsProcessorList(
            [StaticLogitsClippingProcessor(lower=float(lower_bound), upper=float(upper_bound))]
        )

    def generate_batch(
        self,
        prompts: list[str],
        privacy_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()
        request_cfg = build_generation_request("", privacy_cfg)
        tokenizer = self._tokenizer
        model = self._model
        outputs: list[dict[str, Any]] = []

        for start in range(0, len(prompts), self.batch_size):
            prompt_batch = prompts[start : start + self.batch_size]
            model_inputs = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=self.max_context_length is not None,
                max_length=self.max_context_length,
            ).to(model.device)
            generated = model.generate(
                **model_inputs,
                temperature=request_cfg["temperature"],
                max_new_tokens=request_cfg["max_new_tokens"],
                do_sample=request_cfg["do_sample"],
                top_k=request_cfg["top_k"],
                top_p=request_cfg["top_p"],
                logits_processor=self._build_logits_processor(request_cfg),
            )
            decoded = tokenizer.batch_decode(
                generated[:, model_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for prompt, text in zip(prompt_batch, decoded, strict=True):
                outputs.append(
                    {
                        "prompt": prompt,
                        "sanitized_text": _trim_stop_sequences(
                            text,
                            request_cfg["stop_sequences"],
                        ),
                        "generation_request": request_cfg,
                    }
                )

        return outputs
