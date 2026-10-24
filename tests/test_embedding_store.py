"""Tests: the §45/§46 whole-item embedding cache (embeddings.embedding_store).

Mechanics: cache hit does not recompute, version/item changes invalidate,
deterministic layout on disk. No torch, no audio.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.embeddings.embedding_store import (
    cached_embeddings,
    checkpoint_version,
)


class TestEmbeddingStore(unittest.TestCase):
    def test_second_call_hits_cache_without_recompute(self):
        calls = {"n": 0}

        def compute(i):
            calls["n"] += 1
            return np.full(4, i, dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            a = cached_embeddings(
                "arm", ["x", "y"], compute, version_key="v1", cache_dir=td, log=lambda *_: None
            )
            b = cached_embeddings(
                "arm", ["x", "y"], compute, version_key="v1", cache_dir=td, log=lambda *_: None
            )
        np.testing.assert_array_equal(a, b)
        self.assertEqual(calls["n"], 2)  # computed once per item, never again

    def test_version_change_invalidates(self):
        with tempfile.TemporaryDirectory() as td:
            a = cached_embeddings(
                "arm",
                ["x"],
                lambda i: np.ones(3),
                version_key="ckpt:aaa",
                cache_dir=td,
                log=lambda *_: None,
            )
            b = cached_embeddings(
                "arm",
                ["x"],
                lambda i: np.full(3, 9.0),
                version_key="ckpt:bbb",
                cache_dir=td,
                log=lambda *_: None,
            )
        self.assertEqual(float(a[0, 0]), 1.0)
        self.assertEqual(float(b[0, 0]), 9.0)  # retrained -> recomputed

    def test_item_list_change_invalidates(self):
        with tempfile.TemporaryDirectory() as td:
            cached_embeddings(
                "arm",
                ["x", "y"],
                lambda i: np.ones(3),
                version_key="v1",
                cache_dir=td,
                log=lambda *_: None,
            )
            c = cached_embeddings(
                "arm",
                ["x", "z"],
                lambda i: np.full(3, 5.0),
                version_key="v1",
                cache_dir=td,
                log=lambda *_: None,
            )
        self.assertEqual(float(c[1, 0]), 5.0)  # different items -> recomputed

    def test_checkpoint_version_tracks_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(checkpoint_version(td), "none")
            (Path(td) / "best_model.pt").write_bytes(b"weights")
            v1 = checkpoint_version(td)
            (Path(td) / "best_model.pt").write_bytes(b"weights2")
            self.assertNotEqual(checkpoint_version(td), v1)


if __name__ == "__main__":
    unittest.main()
