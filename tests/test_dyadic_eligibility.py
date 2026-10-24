"""Tests: dyadic (emitter-addressee) eligibility audit (§46 Priority 2).

Does the corpus support RECIPIENT-conditioned analysis without confounding
recipient with context? The support table is emitter x recipient x context x
recording; an emitter is evaluable for a recipient contrast only when both
contexts have support AND the recipient sets overlap (disjoint sets =
perfect recipient-context confound — the three dyadic questions of the
roadmap stay separable only then).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.ingestion.dyadic_eligibility import (
    audit_dyadic_support,
    build_dyadic_support_matrix,
    check_recipient_evaluability,
)
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


def _ann(e, r, c, f):
    return CallAnnotation(f, e, r, c, 0, 0, 0, 0)


class TestDyadicEligibility(unittest.TestCase):
    def setUp(self):
        self.annotations = [
            _ann(1, 2, 11, "f1"),
            _ann(1, 2, 11, "f1"),
            _ann(1, 3, 12, "f2"),  # A: B in C11, C in C12 -> confounded
            _ann(4, 2, 11, "f3"),  # D: C11 only -> missing context
            _ann(5, 2, 11, "f4"),  # E: B in both contexts -> evaluable
            _ann(5, 2, 12, "f5"),
            _ann(6, 0, 11, "f6"),  # sentinel addressee: excluded
        ]

    def test_matrix_counts_per_cell(self):
        m = build_dyadic_support_matrix(self.annotations)
        self.assertEqual(m[(1, 2, 11, "f1")], 2)
        self.assertEqual(m[(4, 2, 11, "f3")], 1)
        self.assertNotIn((6, 0, 11, "f6"), m)  # sentinel addressee dropped

    def test_confound_flagged(self):
        m = build_dyadic_support_matrix(self.annotations)
        res = check_recipient_evaluability(m, emitter=1, c1=11, c2=12, min_bouts=1)
        self.assertFalse(res["evaluable"])
        self.assertEqual(res["reason"], "recipient_context_confound")

    def test_missing_context_flagged(self):
        m = build_dyadic_support_matrix(self.annotations)
        res = check_recipient_evaluability(m, emitter=4, c1=11, c2=12, min_bouts=1)
        self.assertFalse(res["evaluable"])
        self.assertEqual(res["reason"], "missing_context_data")

    def test_overlapping_recipients_evaluable(self):
        m = build_dyadic_support_matrix(self.annotations)
        res = check_recipient_evaluability(m, emitter=5, c1=11, c2=12, min_bouts=1)
        self.assertTrue(res["evaluable"])
        self.assertEqual(res["reason"], "adequate_support")

    def test_min_bouts_respected(self):
        m = build_dyadic_support_matrix(self.annotations)
        res = check_recipient_evaluability(m, emitter=5, c1=11, c2=12, min_bouts=2)
        self.assertFalse(res["evaluable"])

    def test_audit_covers_all_emitters(self):
        audit = audit_dyadic_support(self.annotations, c1=11, c2=12, min_bouts=1)
        reasons = {r["emitter"]: r["reason"] for r in audit["results"]}
        self.assertEqual(reasons[1], "recipient_context_confound")
        self.assertEqual(reasons[4], "missing_context_data")
        self.assertEqual(reasons[5], "adequate_support")
        self.assertNotIn(6, reasons)  # sentinel-only emitter absent


if __name__ == "__main__":
    unittest.main()
