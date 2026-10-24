"""Tests: exposure overlap coefficient O_e (dyadic audit refinement).

Identity overlap (shared recipient IDs) is NOT exposure overlap: an emitter
addressing recipient R in 99% of C11 bouts but 1% of C12 bouts shares the
recipient ID yet has almost no controlled exposure. O_e =
sum_r min[P(r|e,C11), P(r|e,C12)] quantifies SUBSTANTIAL shared exposure —
the eligibility quantity the recipient-conditioned design actually needs.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.ingestion.dyadic_eligibility import (
    exposure_overlap,
)
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


def _ann(e, r, c):
    return CallAnnotation(f"f{e}_{r}_{c}", e, r, c, 0, 0, 0, 0)


class TestExposureOverlap(unittest.TestCase):
    def test_perfect_overlap_gives_one(self):
        anns = [_ann(1, 2, 11)] * 50 + [_ann(1, 2, 12)] * 50
        oe, (p11, p12) = exposure_overlap(anns, emitter=1, c1=11, c2=12)
        self.assertAlmostEqual(oe, 1.0, places=6)

    def test_disjoint_gives_zero(self):
        anns = [_ann(1, 2, 11)] * 50 + [_ann(1, 3, 12)] * 50
        oe, _ = exposure_overlap(anns, emitter=1, c1=11, c2=12)
        self.assertAlmostEqual(oe, 0.0, places=6)

    def test_ninetynine_one_asymmetry_is_small(self):
        # the illustrative confound: R in 99% of C11, 1% of C12
        anns = [_ann(1, 2, 11)] * 99 + [_ann(1, 3, 11)] + [_ann(1, 2, 12)] + [_ann(1, 3, 12)] * 99
        oe, _ = exposure_overlap(anns, emitter=1, c1=11, c2=12)
        self.assertAlmostEqual(oe, 0.02, places=6)  # min(0.99, 0.01)+min(0.01,0.99)

    def test_uniform_over_many_recipients(self):
        anns = []
        for r in range(5):
            anns += [_ann(1, r, 11)] * 10 + [_ann(1, r, 12)] * 10
        oe, _ = exposure_overlap(anns, emitter=1, c1=11, c2=12)
        self.assertAlmostEqual(oe, 1.0, places=6)

    def test_missing_context_returns_none(self):
        anns = [_ann(1, 2, 11)] * 50
        oe, _ = exposure_overlap(anns, emitter=1, c1=11, c2=12)
        self.assertIsNone(oe)

    def test_sentinel_addressee_excluded(self):
        anns = [_ann(1, 0, 11)] * 50 + [_ann(1, 0, 12)] * 50 + [_ann(1, 2, 11)] * 5
        oe, _ = exposure_overlap(anns, emitter=1, c1=11, c2=12)
        self.assertIsNone(oe)  # C12 has only sentinels -> no context support


if __name__ == "__main__":
    unittest.main()
