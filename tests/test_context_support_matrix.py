"""Tests: caller-context support matrix (§46 context-conditioned Stage 1).

Identifiability BEFORE inference: which (caller, context-contrast) cells have
enough bouts to support a caller-controlled design, and which are flagged
"not evaluable" so silent confounds cannot sneak into later stages. Works on
any annotation objects exposing .emitter/.context (CallAnnotation for bat).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.ingestion.context_eligibility import (
    build_caller_context_support_matrix,
    check_evaluability,
)
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


def _ann(emitter, context, name):
    return CallAnnotation(name, emitter, 0, context, 0, 0, 0, 0)


class TestCallerContextSupportMatrix(unittest.TestCase):
    def setUp(self):
        self.annotations = [
            _ann(1, 1, "f1_b1"),
            _ann(1, 1, "f1_b2"),
            _ann(1, 2, "f2_b3"),
            _ann(2, 1, "f3_b4"),
            # caller 2 has NO bouts in context 2
        ]

    def test_matrix_counts_bouts_per_caller_per_context(self):
        m = build_caller_context_support_matrix(self.annotations)
        self.assertEqual(m[(1, 1)], 2)
        self.assertEqual(m[(1, 2)], 1)
        self.assertEqual(m[(2, 1)], 1)
        self.assertEqual(m[(2, 2)], 0)

    def test_evaluability_flag_for_missing_contexts(self):
        m = build_caller_context_support_matrix(self.annotations)
        ok = check_evaluability(m, caller=1, c1=1, c2=2, min_bouts=1)
        self.assertTrue(ok["evaluable"])
        self.assertEqual(ok["reason"], "adequate_support")
        bad = check_evaluability(m, caller=2, c1=1, c2=2, min_bouts=1)
        self.assertFalse(bad["evaluable"])
        self.assertEqual(bad["reason"], "missing_context_data")
        self.assertEqual(bad["n_c2"], 0)

    def test_min_bouts_threshold_respected(self):
        m = build_caller_context_support_matrix(self.annotations)
        thin = check_evaluability(m, caller=1, c1=1, c2=2, min_bouts=2)
        self.assertFalse(thin["evaluable"])
        self.assertEqual(thin["reason"], "missing_context_data")

    def test_nonexistent_context_is_not_evaluable(self):
        m = build_caller_context_support_matrix(self.annotations)
        res = check_evaluability(m, caller=1, c1=1, c2=99, min_bouts=1)
        self.assertFalse(res["evaluable"])
        self.assertEqual(res["n_c2"], 0)


if __name__ == "__main__":
    unittest.main()
