"""Tests: opportunity-aware concentration (§46 Stage 3, unit-presence endpoint).

Per codebook entry: is its BOUT-LEVEL PRESENCE concentrated among callers
beyond what their sampling opportunity (bouts examined) explains? The null
draws each caller's presence count from Binomial(E_i, pi_m) — exposure
structure preserved exactly — and concentration is the Herfindahl of
per-caller presence RATES. Saturated entries (present in ~every bout) are
flagged uninformative-for-context-specificity, never "shared".

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.opportunity_concentration import (
    bout_presence_matrix,
    excess_concentration,
    herfindahl_of_rates,
)


class TestBoutPresenceMatrix(unittest.TestCase):
    def test_binary_presence_per_bout(self):
        # 4 frames -> 2 bouts (2 frames each); entry 0 in bout 1 only
        labels = np.array([0, 0, 1, 1])
        offsets = np.array([0, 2, 4])
        P = bout_presence_matrix(labels, offsets, n_clusters=2)
        self.assertEqual(P.shape, (2, 2))
        np.testing.assert_array_equal(P, [[1, 0], [0, 1]])

    def test_empty_bout_row_is_zero(self):
        labels = np.array([5, 5])
        offsets = np.array([0, 0, 2])  # bout 0 empty
        P = bout_presence_matrix(labels, offsets, n_clusters=8)
        np.testing.assert_array_equal(P[0], np.zeros(8))


class TestHerfindahlOfRates(unittest.TestCase):
    def test_concentrated_vs_uniform(self):
        conc = herfindahl_of_rates(np.array([10, 0, 0]), np.array([10, 10, 10]))
        unif = herfindahl_of_rates(np.array([10, 10, 10]), np.array([10, 10, 10]))
        self.assertAlmostEqual(float(conc), 1.0, places=6)
        self.assertAlmostEqual(float(unif), 1 / 3, places=6)


class TestExcessConcentration(unittest.TestCase):
    def test_concentrated_entry_gets_small_p(self):
        # caller 1 shows the entry in every bout; others never — far beyond
        # a Binomial(E_i, pi) null with pi ~ 0.25
        n_i = np.array([10, 0, 0, 0])  # presence bouts per caller
        E_i = np.array([10, 10, 10, 10])  # exposure per caller
        res = excess_concentration(n_i, E_i, n_permutations=200, seed=0)
        self.assertLessEqual(res.p_value, 1.0 / 201)
        self.assertGreater(res.observed, res.null_mean)

    def test_uniform_entry_gets_large_p(self):
        n_i = np.array([2, 3, 2, 3])
        E_i = np.array([10, 10, 10, 10])
        res = excess_concentration(n_i, E_i, n_permutations=200, seed=0)
        self.assertGreater(res.p_value, 0.05)

    def test_null_preserves_exposure(self):
        # exposure of 1 bout vs 100: the null must not equalize them
        n_i = np.array([1, 50])
        E_i = np.array([1, 100])
        res = excess_concentration(n_i, E_i, n_permutations=50, seed=0)
        self.assertTrue(res.exposures.tolist() == [1, 100])

    def test_saturation_flagged(self):
        n_i = np.array([100, 100, 100, 100])
        E_i = np.array([100, 100, 100, 100])
        res = excess_concentration(n_i, E_i, n_permutations=50, seed=0)
        self.assertTrue(res.saturated)
        self.assertIn("uninformative", res.interpretation)

    def test_deterministic(self):
        a = excess_concentration(
            np.array([5, 0, 1]), np.array([10, 10, 10]), n_permutations=100, seed=3
        )
        b = excess_concentration(
            np.array([5, 0, 1]), np.array([10, 10, 10]), n_permutations=100, seed=3
        )
        self.assertEqual(a.p_value, b.p_value)


if __name__ == "__main__":
    unittest.main()
