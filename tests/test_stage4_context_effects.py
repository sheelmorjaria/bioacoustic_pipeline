"""Tests: within-caller context effects (§46 Stage 4, CMH stratified design).

logit-style hierarchy replaced by caller-stratified Cochran–Mantel–Haenszel +
Mantel–Haenszel common odds ratio: stratification BY DESIGN attributes
within-caller context changes to context, not to caller baselines, and is
immune to the context-mixture confound (a_i·pi_c1 + (1-a_i)·pi_c2 pooling).
numpy/scipy only — statsmodels is not a repo dependency and its mixedlm is
a LINEAR mixed model (wrong likelihood for binary outcomes).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import math
import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.context_effects import (
    cmh_test,
    fit_hierarchical_context_model,
    pooled_vs_stratified,
)


def _rows(rates_by_caller, bouts=40, seed=42):
    """rates_by_caller: {caller: (rate_c11, rate_c12)} -> bout rows."""
    rng = np.random.RandomState(seed)
    rows = []
    for caller, (r11, r12) in rates_by_caller.items():
        for _ in range(bouts // 2):
            rows.append({"caller": caller, "context": "C11", "present": int(rng.rand() < r11)})
        for _ in range(bouts // 2):
            rows.append({"caller": caller, "context": "C12", "present": int(rng.rand() < r12)})
    return rows


class TestCMH(unittest.TestCase):
    def test_detects_within_caller_context_effect(self):
        rows = _rows({"A": (0.2, 0.8), "B": (0.5, 0.5), "C": (0.1, 0.4)})
        res = fit_hierarchical_context_model(rows, c1="C11", c2="C12")
        self.assertLess(res["p_value"], 0.05)
        self.assertGreater(res["log_odds_ratio"], 0)
        # true within-caller log-OR ~ log(4) = 1.386 (callers A and C)
        self.assertGreater(res["log_odds_ratio"], 0.5)

    def test_no_effect_gives_large_p(self):
        rows = _rows({"A": (0.3, 0.3), "B": (0.5, 0.5), "C": (0.7, 0.7)})
        res = fit_hierarchical_context_model(rows, c1="C11", c2="C12")
        self.assertGreater(res["p_value"], 0.05)
        self.assertLess(abs(res["log_odds_ratio"]), 0.5)

    def test_mixture_confound_does_not_create_effect(self):
        # callers differ ONLY in context exposure + baseline; within-caller
        # rates identical -> pooled association is spurious, CMH must not flag
        rng = np.random.RandomState(0)
        rows = []
        for caller, (r, frac_c11) in {"A": (0.25, 0.7), "B": (0.75, 0.3)}.items():
            n11, n12 = int(200 * frac_c11), 200 - int(200 * frac_c11)
            rows += [
                {"caller": caller, "context": "C11", "present": int(rng.rand() < r)}
                for _ in range(n11)
            ]
            rows += [
                {"caller": caller, "context": "C12", "present": int(rng.rand() < r)}
                for _ in range(n12)
            ]
        pooled, strat = pooled_vs_stratified(rows, c1="C11", c2="C12")
        self.assertGreater(strat["p_value"], 0.05)  # stratified: no within effect
        self.assertGreaterEqual(strat["n_callers_used"], 2)
        self.assertGreater(abs(pooled["log_odds_ratio"]), abs(strat["log_odds_ratio"]) + 0.2)

    def test_cmh_statistic_known_small_case(self):
        # [[2,1],[1,2]] has diagonal concentration (OR 4); an OPPOSITE pair
        # of strata cancels -> common OR 1, log-OR 0
        stat, p, lor = cmh_test([np.array([[2, 1], [1, 2]])])
        self.assertAlmostEqual(lor, math.log(4.0), places=6)
        stat2, p2, lor2 = cmh_test([np.array([[2, 1], [1, 2]]), np.array([[1, 2], [2, 1]])])
        self.assertAlmostEqual(lor2, 0.0, places=6)
        self.assertGreater(stat2, 0)
        self.assertLess(p2, 1.0)

    def test_callers_without_variation_are_skipped_not_crashing(self):
        rows = [
            {"caller": "A", "context": "C11", "present": 1},
            {"caller": "A", "context": "C12", "present": 1},  # A never absent
            {"caller": "B", "context": "C11", "present": 0},
            {"caller": "B", "context": "C12", "present": 1},
        ] * 10
        res = fit_hierarchical_context_model(rows, c1="C11", c2="C12")
        self.assertIn("p_value", res)
        self.assertGreaterEqual(res["n_callers_used"], 0)


if __name__ == "__main__":
    unittest.main()
