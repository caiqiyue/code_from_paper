from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from thesis_platform.models.embedding import resolve_sentence_transformer_path


class EmbeddingPathTests(unittest.TestCase):
    """Validate local embedder path resolution behavior."""

    def test_direct_sentence_transformer_dir_is_returned_as_is(self) -> None:
        """A directory with modules.json should be treated as a loadable model root."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "model"
            model_dir.mkdir()
            (model_dir / "modules.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_sentence_transformer_path(model_dir), model_dir)

    def test_hf_cache_root_resolves_to_snapshot_from_main_ref(self) -> None:
        """A Hugging Face cache root should resolve to the referenced snapshot directory."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "all-MiniLM-L6-v2"
            snapshot_dir = cache_root / "snapshots" / "revision-123"
            refs_dir = cache_root / "refs"
            snapshot_dir.mkdir(parents=True)
            refs_dir.mkdir(parents=True)
            (refs_dir / "main").write_text("revision-123", encoding="utf-8")
            (snapshot_dir / "modules.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_sentence_transformer_path(cache_root), snapshot_dir)


if __name__ == "__main__":
    unittest.main()
