from __future__ import annotations

import argparse
import json

from thesis_platform.dataset_downloaders import download_datasets, list_dataset_downloaders
from thesis_platform.dataset_downloaders.common import datasets_root, to_package_relative


def main() -> None:
    """Download thesis-platform datasets or print the available dataset list."""

    parser = argparse.ArgumentParser(description="Download thesis-platform datasets.")
    parser.add_argument("--list", action="store_true", help="Print the registered dataset downloaders.")
    parser.add_argument("--names", nargs="+", help="Only download the named datasets.")
    parser.add_argument("--force", action="store_true", help="Redownload datasets even when artifacts already exist.")
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_dataset_downloaders(), ensure_ascii=False, indent=2))
        return

    print(f"Downloading datasets into {to_package_relative(datasets_root())}")
    summary = download_datasets(names=args.names, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
