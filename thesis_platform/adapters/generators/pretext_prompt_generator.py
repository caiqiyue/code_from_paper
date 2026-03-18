from __future__ import annotations

import re

from thesis_platform.core.schemas import Sample

SAMPLE_RE = re.compile(r"<sample>(.*?)</sample>", re.DOTALL | re.IGNORECASE)


class PretextPromptLLMGenerator:
    """Research-mode generator that lets the server LLM consume prompt updates directly."""

    def __init__(self, config, repo_root):
        """Read generation hyper-parameters from config."""

        del repo_root
        self.generated_per_round = int(config.get("generated_per_round", 16))
        self.exemplars_per_prompt = int(config.get("exemplars_per_prompt", 2))
        self.max_new_tokens = int(config.get("max_new_tokens", 192))
        self.temperature = float(config.get("temperature", 0.2))

    @staticmethod
    def _parse_sample_text(raw_text: str) -> str:
        match = SAMPLE_RE.search(raw_text)
        if match:
            return match.group(1).strip()
        cleaned = raw_text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned
        return cleaned

    def _build_prompt(self, *, round_ctx, source_samples: list[Sample]) -> str:
        exemplar_block = "\n\n".join(
            f"Example {idx + 1}:\n{sample.rendered_text()}" for idx, sample in enumerate(source_samples)
        )
        return (
            "You are the server-side synthetic data generator for federated prompt optimization.\n"
            "Use the instruction and guidance below to generate one new synthetic sample.\n"
            "Stay close to the domain and structure implied by the public seed examples.\n"
            "Return only one synthetic sample enclosed by <sample> and </sample>.\n\n"
            f"{round_ctx.prompt_text}\n\n"
            f"{exemplar_block}\n"
        )

    def generate(self, round_ctx):
        """Generate one synthetic sample batch for the current round."""

        if not round_ctx.public_seed_samples:
            raise ValueError("pretext_prompt_llm requires public_seed_samples in the round context.")
        if round_ctx.text_backend is None:
            raise ValueError("pretext_prompt_llm requires a server text backend.")

        pool = round_ctx.public_seed_samples
        generated: list[Sample] = []
        requests = round_ctx.runtime_artifacts.setdefault("generation_requests", [])
        responses = round_ctx.runtime_artifacts.setdefault("generation_responses", [])
        for idx in range(self.generated_per_round):
            start = (idx * self.exemplars_per_prompt) % len(pool)
            source_samples = [pool[(start + offset) % len(pool)] for offset in range(self.exemplars_per_prompt)]
            prompt = self._build_prompt(round_ctx=round_ctx, source_samples=source_samples)
            raw_text = round_ctx.text_backend.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            sample_text = self._parse_sample_text(raw_text)
            requests.append(
                {
                    "round_id": round_ctx.round_id,
                    "request_id": f"gen_req_{round_ctx.round_id}_{idx}",
                    "prompt": prompt,
                    "seed_sample_ids": [sample.sample_id for sample in source_samples],
                }
            )
            responses.append(
                {
                    "round_id": round_ctx.round_id,
                    "response_id": f"gen_res_{round_ctx.round_id}_{idx}",
                    "raw_text": raw_text,
                    "parsed_text": sample_text,
                }
            )
            generated.append(
                Sample(
                    sample_id=f"syn_r{round_ctx.round_id}_{idx}",
                    client_id="server",
                    round_id=round_ctx.round_id,
                    source="synthetic",
                    dataset_name=source_samples[0].dataset_name,
                    task_type=source_samples[0].task_type,
                    text=sample_text,
                    meta={
                        "seed_sample_ids": [sample.sample_id for sample in source_samples],
                        "prompt": round_ctx.prompt_text,
                        "backend_name": getattr(round_ctx.text_backend, "backend_name", type(round_ctx.text_backend).__name__),
                    },
                )
            )
        return generated
