from __future__ import annotations

from dataclasses import dataclass
import json
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
    def evaluation(self) -> dict[str, Any]:
        """Return the evaluation config section."""

        return self.raw.get("evaluation", {})

    @property
    def llm(self) -> dict[str, Any]:
        """Return the shared client/server text-backend config section."""

        return self.raw.get("llm", {})

    @property
    def runtime(self) -> dict[str, Any]:
        """Return the runtime config section."""

        return self.raw.get("runtime", {})

    def repo_root(self) -> Path:
        """Resolve the repository root relative to the current config file."""

        raw_repo_root = self.paths.get("repo_root", ".")
        return (self.path.parent / raw_repo_root).resolve()

    def resolve_path(self, value: str | Path | None) -> Path | None:
        """Resolve a configured path relative to the repository root."""

        if value in (None, ""):
            return None
        path = Path(value)
        if path.is_absolute():
            return path
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
