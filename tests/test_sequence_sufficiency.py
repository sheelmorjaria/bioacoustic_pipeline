"""Tests for the sequence-sufficiency gate (diagnostic abstention).

The gate's whole purpose: stop a sequence diagnostic emitting a biological null
when the token sequences are too sparse to test. These tests pin that the three
reason codes fire in the right order and that the verdict abstains (not_evaluable)
instead of returning "absent".

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

from bioacoustic_pipeline.diagnostics.sequence_sufficiency import (
    REASON_INSUFFICIENT_BIGRAMS,
    REASON_INSUFFICIENT_TOKENS,
    REASON_SEGMENTATION_COLLAPSE,
    evaluate_sufficiency,
)
from bioacoustic_pipeline.verdict_engine import build_report, is_valid


class TestSufficiencyGate(unittest.TestCase):
    def test_segmentation_collapse_when_most_bouts_empty(self):
        # 56% empty bouts, rest single-token -> mirrors the bat holophrastic profile
        seqs = [[1]] * 200 + [[]] * 260
        r = evaluate_sufficiency(seqs)
        self.assertFalse(r.evaluable)
        self.assertEqual(r.reason, REASON_SEGMENTATION_COLLAPSE)
        self.assertAlmostEqual(r.nonzero_bout_rate, 200 / 460)
        self.assertLess(r.nonzero_bout_rate, r.thresholds["min_nonzero_bout_rate"])

    def test_insufficient_tokens_when_bouts_too_short(self):
        # every bout non-empty but median length 1 -> too short on average
        seqs = [[i % 3] for i in range(500)]  # all length 1
        r = evaluate_sufficiency(seqs)
        self.assertFalse(r.evaluable)
        self.assertEqual(r.reason, REASON_INSUFFICIENT_TOKENS)
        self.assertEqual(r.median_tokens_per_bout, 1.0)

    def test_insufficient_bigrams_when_support_too_low(self):
        # long-ish bouts but very few of them -> bigrams under the floor
        seqs = [list(range(10))] * 3
        r = evaluate_sufficiency(seqs, min_total_bigrams=500)
        self.assertFalse(r.evaluable)
        self.assertEqual(r.reason, REASON_INSUFFICIENT_BIGRAMS)

    def test_evaluable_when_dense(self):
        seqs = [list(range(i % 5, (i % 5) + 8)) for i in range(500)]
        r = evaluate_sufficiency(seqs)
        self.assertTrue(r.evaluable)
        self.assertIsNone(r.reason)
        self.assertGreaterEqual(r.total_bigrams, r.thresholds["min_total_bigrams"])

    def test_empty_input_is_not_evaluable(self):
        r = evaluate_sufficiency([])
        self.assertFalse(r.evaluable)


class TestVerdictAbstention(unittest.TestCase):
    def test_sparse_sequences_abstain_on_three_axes(self):
        sparse = evaluate_sufficiency([[1]] * 200 + [[]] * 260)
        rep = build_report(
            "bat", lz_normalized=5.47, alias_density=0.0, alias_pvalue=1.0,
            n_hard=3000, n_soft=2966, n_total_boundaries=5966, sufficiency=sparse,
        )
        self.assertTrue(is_valid(rep))
        for ax in ("sequence_recombination", "functional_aliasing", "motif_structure"):
            self.assertEqual(rep["axes"][ax]["level"], "not_evaluable", ax)
            self.assertIn("NOT EVALUABLE", rep["axes"][ax]["evidence"])

    def test_abstaining_axes_make_no_positive_claim(self):
        sparse = evaluate_sufficiency([[1]] * 200 + [[]] * 260)
        rep = build_report(
            "bat", lz_normalized=5.47, alias_density=0.0, alias_pvalue=1.0,
            n_hard=3000, n_soft=2966, n_total_boundaries=5966, sufficiency=sparse,
        )
        positive = {e["axis"] for e in rep["falsification"] if e["makes_positive_claim"]}
        # The three gated axes must not assert structure just because they abstained.
        self.assertNotIn("functional_aliasing", positive)
        self.assertNotIn("motif_structure", positive)
        self.assertNotIn("sequence_recombination", positive)

    def test_dense_sequences_still_score_normally(self):
        dense = evaluate_sufficiency([list(range(i % 5, (i % 5) + 8)) for i in range(500)])
        rep = build_report(
            "human", lz_normalized=3.57, alias_density=0.04, alias_pvalue=0.010,
            n_hard=25000, n_soft=14000, n_total_boundaries=39000, sufficiency=dense,
        )
        self.assertTrue(is_valid(rep))
        self.assertNotEqual(rep["axes"]["functional_aliasing"]["level"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
