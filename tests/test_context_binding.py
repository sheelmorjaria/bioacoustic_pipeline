"""Tests for annotation-grounded context binding.

Pins: real token<->label association is detected (low p, high NMI); chance is
not (high p); and a speaker-confounded association that looks significant under
the global null is KILLED by the within-Emitter null.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.context_binding import (
    association_test,
    benjamini_hochberg,
    mutual_information,
)


class TestContextBinding(unittest.TestCase):
    def test_strong_association_detected(self):
        rng = np.random.RandomState(0)
        n = 600
        ctx = rng.randint(0, 4, n)
        toks = [[int(rng.choice([2 * c, 2 * c + 1])) for _ in range(8)] for c in ctx]
        r = association_test(toks, ctx.tolist(), n_permutations=200, seed=0)
        self.assertGreater(r.observed_nmi, 0.5)
        self.assertLess(r.p_value, 0.05)

    def test_chance_not_detected(self):
        rng = np.random.RandomState(1)
        n = 600
        ctx = rng.randint(0, 4, n)
        toks = [[int(rng.randint(0, 16)) for _ in range(8)] for _ in range(n)]
        r = association_test(toks, ctx.tolist(), n_permutations=200, seed=0)
        self.assertLess(r.observed_nmi, 0.05)
        self.assertGreater(r.p_value, 0.2)

    def test_within_emitter_null_kills_speaker_confound(self):
        # emitter -> token AND emitter -> context, but token indep of context given emitter
        rng = np.random.RandomState(2)
        n = 600
        emit = rng.randint(0, 6, n)
        ctx = np.array([emit[i] % 3 for i in range(n)])
        toks = [[int(emit[i] * 5 + rng.randint(0, 3)) for _ in range(6)] for i in range(n)]
        global_r = association_test(toks, ctx.tolist(), n_permutations=200, seed=0, null_type="global")
        within_r = association_test(
            toks, ctx.tolist(), n_permutations=200, seed=0,
            group=emit.tolist(), null_type="within_group",
        )
        # Spuriously significant globally, destroyed within-emitter.
        self.assertLess(global_r.p_value, 0.1)
        self.assertGreater(within_r.p_value, 0.5)

    def test_mutual_information_zero_for_independent_table(self):
        table = np.full((4, 4), 10.0)  # uniform independent -> MI ~ 0
        mi, nmi = mutual_information(table)
        self.assertLess(mi, 1e-9)
        self.assertLess(nmi, 1e-9)

    def test_mutual_information_high_for_diagonal_table(self):
        table = np.eye(4) * 10.0
        _mi, nmi = mutual_information(table)
        self.assertGreater(nmi, 0.9)

    def test_benjamini_hochberg(self):
        pvals = [0.001, 0.04, 0.5, 0.6]
        qs = benjamini_hochberg(pvals)
        self.assertAlmostEqual(qs[0], 0.004, places=3)
        self.assertLessEqual(max(qs), 1.0)
        # q-values are monotone non-decreasing when ordered by p.
        qs_in_p_order = [q for _p, q in sorted(zip(pvals, qs))]
        self.assertEqual(qs_in_p_order, sorted(qs_in_p_order))


if __name__ == "__main__":
    unittest.main()
