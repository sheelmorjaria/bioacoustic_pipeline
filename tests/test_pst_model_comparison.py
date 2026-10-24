"""Tests: PST M0/M1/M2 nested model comparison (§46 triangulation of T_y≈0).

Mechanics: bout-safe counting, suffix-backoff with finite smoothed
probabilities, depth-as-cap (contexts retained only when they beat their
suffix), relabeling invariance, opposite-cycle discrimination, held-out NLL
endpoint, Phipson–Smyth p.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.pst_models import (
    _perm_p_value,
    fit_caller_transition_model,
    fit_shared_pst,
    fit_static_bias_model,
    score_heldout_nll,
)


class TestPSTMechanics(unittest.TestCase):
    def test_constant_sequence_finite_and_no_credit(self):
        pst = fit_shared_pst([[0] * 100], max_depth=4, n_clusters=24)
        self.assertGreaterEqual(pst.entropy_rate(), 0.0)
        self.assertTrue(np.isfinite(pst.entropy_rate()))
        self.assertEqual(pst.n_contexts_improving_over_suffix(), 0)

    def test_first_order_cycle_suffices(self):
        pst = fit_shared_pst([[1, 2, 3] * 50], max_depth=4, n_clusters=24)
        self.assertEqual(pst.n_contexts_improving_over_suffix(depth_min=2), 0)
        self.assertGreater(pst.n_contexts_improving_over_suffix(depth_min=1), 0)

    def test_second_order_pattern_needs_depth_2(self):
        pst = fit_shared_pst([[0, 0, 1, 1] * 50], max_depth=4, n_clusters=24)
        self.assertGreater(pst.n_contexts_improving_over_suffix(depth_min=2), 0)

    def test_relabeling_preserves_complexity(self):
        a = fit_shared_pst([[1, 2, 3] * 50], max_depth=4, n_clusters=24)
        b = fit_shared_pst([[4, 5, 6] * 50], max_depth=4, n_clusters=24)
        self.assertEqual(a.n_significant_nodes, b.n_significant_nodes)

    def test_opposite_cycles_same_complexity_poor_cross_prediction(self):
        pop = fit_shared_pst([[1, 2, 3] * 50], max_depth=4, n_clusters=24)
        caller = fit_shared_pst([[1, 3, 2] * 50], max_depth=4, n_clusters=24)
        self.assertEqual(pop.n_significant_nodes, caller.n_significant_nodes)
        self.assertGreater(
            score_heldout_nll(pop, [[1, 3, 2] * 20]), score_heldout_nll(caller, [[1, 3, 2] * 20])
        )

    def test_bout_boundaries_no_invented_transitions(self):
        pst = fit_shared_pst([[0, 1], [2, 3]], max_depth=4, n_clusters=24)
        self.assertNotIn((1, 2), pst.all_observed_transitions())

    def test_unseen_context_backs_off_finitely(self):
        pst = fit_shared_pst([[1, 2, 3] * 50], max_depth=4, n_clusters=24)
        p = pst.predict_proba(context=[5, 5], next_token=1)
        self.assertTrue(np.isfinite(p))
        self.assertGreater(p, 0.0)


class TestNestedModels(unittest.TestCase):
    def test_m1_beats_m0_under_token_bias(self):
        rng = np.random.RandomState(42)
        pop = [list(rng.randint(0, 3, 100)) for _ in range(10)]
        caller_train = [list(rng.choice([0, 0, 0, 1, 2], 50))]
        caller_test = [list(rng.choice([0, 0, 0, 1, 2], 50))]
        m0 = fit_shared_pst(pop, max_depth=2, n_clusters=3)
        m1 = fit_static_bias_model(m0, caller_train)
        self.assertGreater(
            score_heldout_nll(m0, caller_test) - score_heldout_nll(m1, caller_test), 0.0
        )

    def test_m2_beats_m1_under_caller_transitions(self):
        rng = np.random.RandomState(42)
        pop = [list(rng.randint(0, 3, 100)) for _ in range(10)]
        caller_train = [[0, 1, 2] * 30]
        caller_test = [[0, 1, 2] * 20]
        m0 = fit_shared_pst(pop, max_depth=2, n_clusters=3)
        m1 = fit_static_bias_model(m0, caller_train)
        m2 = fit_caller_transition_model(m0, caller_train, max_depth=2)
        self.assertGreater(
            score_heldout_nll(m1, caller_test) - score_heldout_nll(m2, caller_test), 0.0
        )

    def test_m2_no_better_when_caller_is_population_like(self):
        rng = np.random.RandomState(7)
        pop = [list(rng.randint(0, 3, 100)) for _ in range(10)]
        caller_train = [list(rng.randint(0, 3, 100))]
        caller_test = [list(rng.randint(0, 3, 100))]
        m0 = fit_shared_pst(pop, max_depth=2, n_clusters=3)
        m1 = fit_static_bias_model(m0, caller_train)
        m2 = fit_caller_transition_model(m0, caller_train, max_depth=2)
        # noisy caller: M2's in-sample edge must not generalize to a large win
        self.assertLess(
            score_heldout_nll(m1, caller_test) - score_heldout_nll(m2, caller_test), 0.5
        )


class TestPermP(unittest.TestCase):
    def test_known_array(self):
        p = _perm_p_value(5.0, [3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertAlmostEqual(p, 4 / 6, places=6)


if __name__ == "__main__":
    unittest.main()


class TestM1ContractChecks(unittest.TestCase):
    """M1 must be an exact no-op under uniform bias, normalize, and
    sharpen toward a known bias direction."""

    def _m0(self):
        rng = np.random.RandomState(0)
        pop = [list(rng.randint(0, 3, 100)) for _ in range(6)]
        return fit_shared_pst(pop, max_depth=2, n_clusters=3)

    def test_uniform_bias_is_exact_noop(self):
        from bioacoustic_pipeline.diagnostics.pst_models import BiasedPST

        m0 = self._m0()
        m1 = BiasedPST(m0, bias={t: 1 / 3 for t in range(3)})
        for ctx in ([], [0], [1, 2]):
            for tok in range(3):
                self.assertAlmostEqual(
                    m1.predict_proba(context=ctx, next_token=tok),
                    m0.predict_proba(context=ctx, next_token=tok),
                    places=12,
                )

    def test_q1_normalizes(self):
        from bioacoustic_pipeline.diagnostics.pst_models import BiasedPST

        m0 = self._m0()
        m1 = BiasedPST(m0, bias={0: 0.7, 1: 0.2, 2: 0.1})
        for ctx in ([], [2], [0, 1]):
            total = sum(m1.predict_proba(context=ctx, next_token=t) for t in range(3))
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_known_bias_sharpens_toward_target(self):
        from bioacoustic_pipeline.diagnostics.pst_models import BiasedPST

        m0 = self._m0()
        m1 = BiasedPST(m0, bias={0: 0.9, 1: 0.05, 2: 0.05})
        # on a context where M0 is nearly uniform, bias must move mass to 0
        ctx = [0]
        self.assertGreater(
            m1.predict_proba(context=ctx, next_token=0)
            / max(m0.predict_proba(context=ctx, next_token=0), 1e-9),
            1.0,
        )
