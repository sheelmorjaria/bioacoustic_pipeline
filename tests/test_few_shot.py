"""Tests: few-shot caller instruments (diagnostics.few_shot).

Mechanics only: support/query file disjointness, determinism, chance baselines
on synthetic embeddings with planted caller structure. No torch, no network.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.few_shot import (
    knn_few_shot,
    prototype_accuracy,
)


def _fixture(n_callers=5, files_per_caller=8, windows_per_file=3, dim=12, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_callers, dim) * 5.0
    embs, callers, files = [], [], []
    for c in range(n_callers):
        for f in range(files_per_caller):
            for _ in range(windows_per_file):
                embs.append(centers[c] + 0.1 * rng.randn(dim))
                callers.append(c)
                files.append(f"c{c}_f{f:02d}.wav")
    return np.asarray(embs), np.asarray(callers), np.asarray(files)


class TestKnnFewShot(unittest.TestCase):
    def test_perfect_clustering_gives_full_accuracy(self):
        r = knn_few_shot(*_fixture(), n_support=3, k=1, seed=0)
        self.assertEqual(r.n_callers, 5)
        self.assertEqual(r.chance, 0.2)
        self.assertGreater(r.accuracy, 0.99)
        self.assertGreater(r.n_query, 0)

    def test_support_and_query_files_disjoint(self):
        r = knn_few_shot(*_fixture(), n_support=3, k=1, seed=0)
        for caller, sfiles in r.support_files.items():
            for f in sfiles:
                self.assertNotIn(f, r.query_files[caller])

    def test_random_embeddings_near_chance(self):
        rng = np.random.RandomState(3)
        embs = rng.randn(400, 10)
        callers = np.repeat(np.arange(5), 80)
        files = np.array([f"c{c}_f{f:02d}" for c in range(5) for f in range(8) for _ in range(10)])
        r = knn_few_shot(embs, callers, files, n_support=3, k=1, seed=0)
        self.assertLess(r.accuracy, r.chance * 1.6)

    def test_deterministic_under_seed(self):
        args = _fixture()
        a = knn_few_shot(*args, n_support=2, k=1, seed=4)
        b = knn_few_shot(*args, n_support=2, k=1, seed=4)
        self.assertEqual(a.accuracy, b.accuracy)
        self.assertEqual(a.support_files, b.support_files)

    def test_insufficient_callers_skipped(self):
        # caller 0 has only 4 files: 3 support + <2 query -> skipped; caller 1 ok
        rng = np.random.RandomState(0)
        embs = rng.randn(40, 6)
        callers = np.array([0] * 12 + [1] * 28)
        files = np.array(
            [f"c0_f{f}" for f in range(4) for _ in range(3)]
            + [f"c1_f{f}" for f in range(7) for _ in range(4)]
        )
        r = knn_few_shot(embs, callers, files, n_support=3, k=1, seed=0)
        self.assertEqual(r.n_callers, 1)
        self.assertEqual(r.chance, 1.0)


class TestPrototypeAccuracy(unittest.TestCase):
    def test_perfect_clustering_gives_full_accuracy(self):
        r = prototype_accuracy(*_fixture(), n_support=2, seed=0)
        self.assertGreater(r.accuracy, 0.99)
        self.assertEqual(r.n_callers, 5)

    def test_random_embeddings_near_chance(self):
        rng = np.random.RandomState(5)
        embs = rng.randn(400, 10)
        callers = np.repeat(np.arange(5), 80)
        files = np.array([f"c{c}_f{f:02d}" for c in range(5) for f in range(8) for _ in range(10)])
        r = prototype_accuracy(embs, callers, files, n_support=2, seed=0)
        self.assertLess(r.accuracy, r.chance * 1.6)

    def test_support_query_disjoint_and_deterministic(self):
        args = _fixture()
        a = prototype_accuracy(*args, n_support=2, seed=1)
        b = prototype_accuracy(*args, n_support=2, seed=1)
        self.assertEqual(a.accuracy, b.accuracy)
        for caller, sfiles in a.support_files.items():
            for f in sfiles:
                self.assertNotIn(f, a.query_files[caller])


if __name__ == "__main__":
    unittest.main()
