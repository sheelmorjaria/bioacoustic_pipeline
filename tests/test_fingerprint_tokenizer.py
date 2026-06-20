"""Tests for the shared cross-species FingerprintTokenizer.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (
    HZ_SCALE_PHYSICS_INDICES,
    FingerprintTokenizer,
    cross_species_mask,
)


class TestFingerprintTokenizer(unittest.TestCase):
    def test_mask_drops_only_hz_physics_features(self):
        # combined dim 44 (physics 15 + macro + micro); 5 Hz physics dims dropped.
        m = cross_species_mask(44)
        self.assertEqual(int(m.sum()), 44 - len(HZ_SCALE_PHYSICS_INDICES))
        for i in HZ_SCALE_PHYSICS_INDICES:
            self.assertFalse(m[i])
        self.assertTrue(m[0])  # dur_ms kept

    def test_fit_and_tokenize_shared_codebook(self):
        rng = np.random.RandomState(0)
        # two well-separated clusters of 44-D vectors
        a = rng.normal(loc=0.0, scale=0.3, size=(50, 44))
        b = rng.normal(loc=3.0, scale=0.3, size=(50, 44))
        X = np.vstack([a, b])
        tok = FingerprintTokenizer.fit(X, k=2)
        self.assertEqual(tok.k, 2)
        toks_a = {tok.tokenize(v) for v in a}
        toks_b = {tok.tokenize(v) for v in b}
        # the two clusters map to different codes (shared codebook separates them)
        self.assertTrue(toks_a.isdisjoint(toks_b))

    def test_tokenize_silent_segment_is_minus_one(self):
        rng = np.random.RandomState(1)
        tok = FingerprintTokenizer.fit(rng.normal(size=(30, 44)), k=4)
        self.assertEqual(tok.tokenize(np.zeros(44)), -1)

    def test_utilization_report(self):
        rng = np.random.RandomState(2)
        X = rng.normal(size=(200, 44))
        tok = FingerprintTokenizer.fit(X, k=8)
        u = tok.utilization(X)
        self.assertLessEqual(u["used"], u["k"])
        self.assertEqual(u["n"], 200)


if __name__ == "__main__":
    unittest.main()
