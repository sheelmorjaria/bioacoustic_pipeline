"""Tests for the token<->word association statistics (diagnostics.token_word).

Pure-function checks: overlap math, the three assignment policies, and the
consistency metrics. No audio / no torch.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

from bioacoustic_pipeline.diagnostics.token_word import (
    assign_word_by_overlap,
    build_token_word_table,
    consistency_by_class,
    global_consistency,
    is_function_word,
    macro_word_consistency,
    overlap_ms,
    token_purity,
    word_consistency,
)


class TestOverlap(unittest.TestCase):
    def test_overlap_basic(self):
        self.assertAlmostEqual(overlap_ms(0, 10, 5, 15), 5.0)
        self.assertEqual(overlap_ms(0, 10, 20, 30), 0.0)
        self.assertAlmostEqual(overlap_ms(0, 10, 0, 10), 10.0)

    def test_strict_policy_unambiguous(self):
        spans = [(0, 100, "THE"), (100, 200, "BOY")]
        # segment fully inside BOY -> assigned
        self.assertEqual(assign_word_by_overlap(120, 180, spans, policy="strict")[0], "BOY")
        # segment split 50/50 across the boundary -> genuinely ambiguous -> None
        self.assertIsNone(assign_word_by_overlap(80, 120, spans, policy="strict", min_ratio=0.5))

    def test_max_overlap_always_assigns(self):
        spans = [(0, 100, "THE"), (100, 200, "BOY")]
        w, _r = assign_word_by_overlap(80, 150, spans, policy="max_overlap")
        self.assertEqual(w, "BOY")  # 50ms overlap with BOY vs 20ms with THE

    def test_midpoint_policy(self):
        # non-adjacent spans so a real gap exists between them
        spans = [(0, 90, "THE"), (110, 200, "BOY")]
        self.assertEqual(assign_word_by_overlap(40, 60, spans, policy="midpoint")[0], "THE")
        self.assertEqual(assign_word_by_overlap(140, 160, spans, policy="midpoint")[0], "BOY")
        self.assertIsNone(assign_word_by_overlap(95, 105, spans, policy="midpoint"))  # gap at mid=100


class TestConsistency(unittest.TestCase):
    def setUp(self):
        # token 42 -> TOM (12), PIG (8); token 17 -> THE (20), A (5)
        self.records = (
            [{"token": 42, "word": "TOM"}] * 12
            + [{"token": 42, "word": "PIG"}] * 8
            + [{"token": 17, "word": "THE"}] * 20
            + [{"token": 17, "word": "A"}] * 5
            + [{"token": 99, "word": None}] * 3  # unlabeled -> ignored
        )

    def test_token_purity(self):
        pur = token_purity(build_token_word_table(self.records))
        self.assertAlmostEqual(pur[42]["purity"], 12 / 20)
        self.assertEqual(pur[42]["top_word"], "TOM")
        self.assertEqual(pur[17]["top_word"], "THE")

    def test_word_consistency(self):
        wc = word_consistency(self.records)
        self.assertAlmostEqual(wc["TOM"]["consistency"], 1.0)  # all TOM -> token 42
        self.assertEqual(wc["THE"]["top_token"], 17)

    def test_global_vs_macro(self):
        # global is micro (dominated by frequent THE); macro averages words equally
        g = global_consistency(self.records)
        m = macro_word_consistency(self.records, min_count=5)
        self.assertGreater(g, 0.0)
        self.assertGreater(m, 0.0)

    def test_content_function_split(self):
        by = consistency_by_class(self.records, min_count=5)
        # THE/A are function words; TOM/PIG content. Both classes present here.
        self.assertIn("content", by)
        self.assertIn("function", by)

    def test_function_word_detection(self):
        self.assertTrue(is_function_word("THE"))
        self.assertTrue(is_function_word("don't"))
        self.assertFalse(is_function_word("TOM"))


if __name__ == "__main__":
    unittest.main()
