from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path used in constrained environments
    yaml = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, with override values taking precedence."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file, using a stdlib-only fallback parser when PyYAML is unavailable."""

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} must decode to a mapping.")
        return data
    return _load_yaml_without_dependency(text)


def _parse_scalar(raw: str) -> Any:
    """Parse a scalar YAML value into a Python primitive."""

    value = raw.strip()
    if value == "":
        return ""
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value.replace("'", '"'))
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _load_yaml_without_dependency(text: str) -> dict[str, Any]:
    """Parse a restricted subset of YAML used by the MVP configs."""

    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        """Parse one indentation block into either a mapping or a sequence."""

        mapping: dict[str, Any] = {}
        sequence: list[Any] | None = None
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if content.startswith("- "):
                if sequence is None:
                    sequence = []
                item = content[2:].strip()
                if item:
                    sequence.append(_parse_scalar(item))
                    index += 1
                else:
                    child, index = parse_block(index + 1, current_indent + 2)
                    sequence.append(child)
                continue

            key, _, raw_value = content.partition(":")
            if not _:
                raise ValueError(f"Invalid YAML line: {content}")
            if raw_value.strip():
                mapping[key.strip()] = _parse_scalar(raw_value)
                index += 1
            else:
                child, index = parse_block(index + 1, current_indent + 2)
                mapping[key.strip()] = child
        return (sequence if sequence is not None else mapping), index

    parsed, _ = parse_block(0, 0)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML object must be a mapping.")
    return parsed


def _load_with_includes(path: Path) -> dict[str, Any]:
    """Load a YAML file and recursively resolve its inherited config fragments."""

    data = _load_yaml(path)
    includes = data.pop("inherits", []) or []
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (path.parent / include).resolve()
        merged = _deep_merge(merged, _load_with_includes(include_path))
    return _deep_merge(merged, data)


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _normalize_config_path(value: str | Path) -> str:
    """Normalize user-configured paths so the same config works on Windows and Linux."""

    raw = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if _WINDOWS_DRIVE_RE.match(raw) or raw.startswith("\\\\"):
        return raw
    return raw.replace("\\", "/")


@dataclass(slots=True)
class ExperimentConfig:
    """Typed wrapper around the raw experiment configuration mapping."""

    path: Path
    raw: dict[str, Any]

    @property
    def meta(self) -> dict[str, Any]:
        """Return the experiment metadata section."""

        return self.raw.get("meta", {})

    @property
    def paths(self) -> dict[str, Any]:
        """Return the path section."""

        return self.raw.get("paths", {})

    @property
    def data(self) -> dict[str, Any]:
        """Return the data section."""

        return self.raw.get("data", {})

    @property
    def federation(self) -> dict[str, Any]:
        """Return the federation runtime section."""

        return self.raw.get("federation", {})

    @property
    def generator(self) -> dict[str, Any]:
        """Return the generator config section."""

        return self.raw.get("generator", {})

    @property
    def scorer(self) -> dict[str, Any]:
        """Return the scorer config section."""

        return self.raw.get("scorer", {})

    @property
    def retriever(self) -> dict[str, Any]:
        """Return the retriever config section."""

        return self.raw.get("retriever", {})

    @property
    def critic(self) -> dict[str, Any]:
        """Return the critic config section."""

        return self.raw.get("critic", {})

    @property
    def aggregator(self) -> dict[str, Any]:
        """Return the aggregator config section."""

        return self.raw.get("aggregator", {})

    @property
    def prototype(self) -> dict[str, Any]:
        """Return the prototype extraction config with v3 defaults."""

        return _deep_merge(
            {
                "name": "minilm_mean",
                "embedding_model": "thesis_platform/open_model/all_minilm_l6_v2",
                "allow_hashing_fallback": False,
            },
            self.raw.get("prototype", {}),
        )

    @property
    def routing(self) -> dict[str, Any]:
        """Return the routing config with v3 defaults."""

        return _deep_merge(
            {
                "enabled": False,
                "personalized_mix_ratio": 0.7,
                "cluster_eps": 0.35,
                "cluster_min_samples": 2,
            },
            self.raw.get("routing", {}),
        )

    @property
    def privacy(self) -> dict[str, Any]:
        """Return the privacy config with paper-aligned defaults."""

        return _deep_merge(
            {
                "enabled": False,
                "mode": "sample_critique_upload_proxy",
                "epsilon": 1.29,
                "delta": 3e-6,
                "sample_cost": 0.0,
                "critique_cost": 0.0,
                "upload_token_cost": 0.0,
                "enforce_budget": False,
            },
            self.raw.get("privacy", {}),
        )

    @property
    def downstream_eval(self) -> dict[str, Any]:
        """Return the downstream-eval config with v3 defaults."""

        config = _deep_merge(
            {
                "enabled": False,
                "kind": "none",
                "export_filename": "llama7b_text_syn.json",
                "run_large_eval": False,
                "run_small_eval": False,
                "large_eval_mode": "auto",
                "windows_large_eval_mode": "peft_lora",
                "linux_large_eval_mode": "peft_lora",
                "small_eval_mode": "auto",
                "windows_small_eval_mode": "gpt2",
                "linux_small_eval_mode": "distilgpt2",
                "baseline_summary_paths": [],
            },
            self.raw.get("downstream_eval", {}),
        )
        if "run_large_eval" not in self.raw.get("downstream_eval", {}):
            config["run_large_eval"] = config.get("kind") == "pretext_large_eval"
        return config

    @property
    def evaluation(self) -> dict[str, Any]:
        """Return the evaluation config section."""

        return self.raw.get("evaluation", {})

    @property
    def cross_domain_eval(self) -> dict[str, Any]:
        """Return the cross-domain evaluation config for transfer learning experiments."""

        return self.raw.get("cross_domain_eval", {})

    @property
    def llm(self) -> dict[str, Any]:
        """Return the shared client/server text-backend config section."""

        return self.raw.get("llm", {})

    @property
    def runtime(self) -> dict[str, Any]:
        """Return the runtime config section."""

        return self.raw.get("runtime", {})

    @property
    def stage_a(self) -> dict[str, Any]:
        """Return the Stage A config section."""

        return self.raw.get("stage_a", {})

    @property
    def stage_b(self) -> dict[str, Any]:
        """Return the Stage B config section."""

        return self.raw.get("stage_b", {})

    @property
    def stage_c(self) -> dict[str, Any]:
        """Return the Stage C config section."""

        return self.raw.get("stage_c", {})

    def repo_root(self) -> Path:
        """Resolve the repository root relative to the current config file."""
        raw_repo_root = _normalize_config_path(self.paths.get("repo_root", "."))
        return (self.path.parent / raw_repo_root).resolve()

    def resolve_path(self, value: str | Path | None) -> Path | None:
        """Resolve a configured path relative to the repository root.

        On Windows, absolute paths with non-ASCII characters may be garbled in Path
        string representations. Use cwd-based reconstruction for such paths.
        """
        import os
        if value in (None, ""):
            return None
        path = Path(_normalize_config_path(value))
        if path.is_absolute():
            str_path = str(path)
            # Check if path contains garbled chars (non-ASCII chars appear as replacement chars)
            if '�' in str_path or not os.path.exists(path):
                # Try to reconstruct from cwd
                cwd = os.getcwd()
                parts = str_path.split(os.sep)
                # Find 'thesis_platform' in path and reconstruct from cwd
                for i, part in enumerate(parts):
                    if 'thesis_platform' in part.lower():
                        base = Path(cwd)
                        for _ in range(10):
                            if (base / "thesis_platform").is_dir():
                                break
                            base = base.parent
                        reconstructed = base.joinpath(*parts[i:])
                        if reconstructed.exists():
                            return reconstructed.resolve()
                # Also try with 'datasets' as marker
                for i, part in enumerate(parts):
                    if 'datasets' in part.lower():
                        base = Path(cwd)
                        for _ in range(10):
                            if (base / "datasets").is_dir():
                                return base.resolve()
                            base = base.parent
                # Try cwd as base
                base = Path(cwd)
                for _ in range(10):
                    tp = base / "thesis_platform"
                    if tp.is_dir():
                        reconstructed = tp.joinpath(*parts[i+1:])
                        if reconstructed.exists():
                            return reconstructed.resolve()
                    base = base.parent
            return path.resolve() if path.exists() else path
        return (self.repo_root() / path).resolve()

    def output_root(self) -> Path:
        """Resolve the configured experiment output root."""

        output_root = self.resolve_path(self.paths.get("output_root", "./outputs/thesis_platform"))
        if output_root is None:
            raise ValueError("output_root must be configured.")
        return output_root

    def cache_root(self) -> Path:
        """Resolve the configured cache root."""

        cache_root = self.resolve_path(self.paths.get("cache_root", "./thesis_platform/workspace/cache"))
        if cache_root is None:
            raise ValueError("cache_root must be configured.")
        return cache_root


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one experiment config file into an ExperimentConfig object."""

    resolved = Path(path).resolve()
    return ExperimentConfig(path=resolved, raw=_load_with_includes(resolved))
