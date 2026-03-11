from __future__ import annotations

import argparse
import json

from thesis_platform.model_downloaders import download_models, list_model_downloaders
from thesis_platform.model_downloaders.common import models_root, to_package_relative


def parse_repo_overrides(entries: list[str] | None) -> dict[str, str]:
    """Parse repeated --repo-override arguments."""

    if not entries:
        return {}

    overrides: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --repo-override value: {entry}. Expected <model_name>=<huggingface_repo_id>.")
        name, repo_id = entry.split("=", 1)
        name = name.strip()
        repo_id = repo_id.strip()
        if not name or not repo_id:
            raise ValueError(f"Invalid --repo-override value: {entry}. Expected <model_name>=<huggingface_repo_id>.")
        overrides[name] = repo_id
    return overrides


def main() -> None:
    """Download thesis-platform models or print the available model list."""

    parser = argparse.ArgumentParser(description="Download thesis-platform open models.")
    parser.add_argument("--list", action="store_true", help="Print the registered model downloaders.")
    parser.add_argument("--names", nargs="+", help="Only download the named models.")
    parser.add_argument("--force", action="store_true", help="Redownload models even when artifacts already exist.")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional models up to 15B in the default download set.",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include models larger than 15B in the default download set.",
    )
    parser.add_argument(
        "--repo-override",
        action="append",
        help="Override one model source as <model_name>=<huggingface_repo_id>. Can be repeated.",
    )
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_model_downloaders(), ensure_ascii=False, indent=2))
        return

    print(f"Downloading models into {to_package_relative(models_root())}")
    summary = download_models(
        names=args.names,
        force=args.force,
        include_optional=args.include_optional,
        include_large=args.include_large,
        repo_overrides=parse_repo_overrides(args.repo_override),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
