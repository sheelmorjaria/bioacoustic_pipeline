"""Tests: cross-taxa silhouette + permutation null (44-D sweep instrument).

Silhouette WITHOUT a null overstates separation under class imbalance; the
utility returns the observed score and a label-permutation null
(Phipson–Smyth corrected p) so the sweep reports magnitude + significance,
per the repo's statistical-honesty conventions.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.cross_taxa_metrics import (
    compute_silhouette_by_class,
    silhouette_with_null,
)


class TestSilhouette(unittest.TestCase):
    def test_well_separated_classes_score_high(self):
        rng = np.random.RandomState(0)
        X = np.vstack(
            [rng.randn(10, 44) * 0.5, rng.randn(10, 44) * 0.5 + 10, rng.randn(10, 44) * 0.5 - 10]
        )
        y = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        self.assertGreater(compute_silhouette_by_class(X, y), 0.5)

    def test_single_class_returns_zero(self):
        X = np.random.RandomState(1).randn(10, 44)
        self.assertEqual(compute_silhouette_by_class(X, np.array(["A"] * 10)), 0.0)

    def test_null_gives_large_p_for_random_structure(self):
        rng = np.random.RandomState(2)
        X = rng.randn(60, 44)  # no structure
        y = rng.choice(["A", "B", "C"], 60)
        obs, p = silhouette_with_null(X, y, n_permutations=50, seed=0)
        self.assertGreater(p, 0.05)

    def test_separated_classes_get_small_p(self):
        rng = np.random.RandomState(3)
        X = np.vstack([rng.randn(20, 44) * 0.3, rng.randn(20, 44) * 0.3 + 10])
        y = np.array(["A"] * 20 + ["B"] * 20)
        obs, p = silhouette_with_null(X, y, n_permutations=50, seed=0)
        self.assertLessEqual(p, 1.0 / 51)
        self.assertGreater(obs, 0.5)


if __name__ == "__main__":
    unittest.main()
