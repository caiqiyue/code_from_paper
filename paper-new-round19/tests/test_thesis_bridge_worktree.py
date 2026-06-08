from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from paper_new_selector import thesis_bridge


def test_resolve_repo_root_falls_back_to_git_common_checkout(tmp_path):
    worktree_root = tmp_path / "worktree"
    shared_root = tmp_path / "shared_checkout"
    module_path = worktree_root / "paper-new-round19" / "paper_new_selector" / "thesis_bridge.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# stub\n", encoding="utf-8")

    (shared_root / "thesis_platform" / "datasets").mkdir(parents=True, exist_ok=True)
    (shared_root / "thesis_platform" / "open_model").mkdir(parents=True, exist_ok=True)
    (shared_root / "PrE-Text").mkdir(parents=True, exist_ok=True)

    with patch.object(thesis_bridge, "__file__", str(module_path)):
        with patch.object(thesis_bridge, "_resolve_git_common_checkout_root", lambda _: shared_root):
            assert thesis_bridge.resolve_repo_root().samefile(shared_root.resolve())
