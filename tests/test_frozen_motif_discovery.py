"""Tests: frozen motif discovery + assignment (§46 Stage 2).

The frozen-vocabulary contract: discovery (fit) touches TRAIN-split
trajectories only; assignment on held-out data never refits — so eval-split
motif usage is a measurement of a FIXED vocabulary, not a moving target.
Reuses the Stage-1 eligibility matrix to filter callers; the clustering core
is a KMeans surrogate for the DUSTED family discovery (swappable via
fit/assign; documented, not hidden).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.ingestion.frozen_dusted import (
    FrozenDUSTEDModel,
    filter_eligible_intervals,
)
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


def _ann(emitter, context, name):
    return CallAnnotation(name, emitter, 0, context, 0, 0, 0, 0)


class TestFilterEligibleIntervals(unittest.TestCase):
    def setUp(self):
        self.intervals = [
            _ann(1, 11, "f1"),
            _ann(1, 12, "f2"),
            _ann(2, 11, "f3"),  # caller 2 has no C12 -> ineligible
        ]
        self.matrix = {(1, 11): 25, (1, 12): 22, (2, 11): 30, (2, 12): 0}

    def test_only_eligible_callers_retained(self):
        out = filter_eligible_intervals(
            self.intervals,
            contexts=(11, 12),
            min_bouts=20,
            eligibility_matrix=self.matrix,
        )
        callers = {a.emitter for a in out}
        self.assertEqual(callers, {1})
        self.assertTrue(all(a.context in (11, 12) for a in out))

    def test_other_contexts_dropped(self):
        self.intervals.append(_ann(1, 3, "f9"))
        out = filter_eligible_intervals(
            self.intervals,
            contexts=(11, 12),
            min_bouts=20,
            eligibility_matrix=self.matrix,
        )
        self.assertNotIn(3, {a.context for a in out})


class TestFrozenDUSTEDModel(unittest.TestCase):
    def test_fit_and_assign_ranges_and_determinism(self):
        rng = np.random.RandomState(0)
        train = rng.randn(500, 8)
        held_out = rng.randn(60, 8)
        model = FrozenDUSTEDModel(n_clusters=5, random_state=0)
        model.fit(train)
        train_labels = model.assign(train)
        eval_labels = model.assign(held_out)
        self.assertTrue(np.all(train_labels < 5) and np.all(train_labels >= 0))
        self.assertTrue(np.all(eval_labels < 5) and np.all(eval_labels >= 0))
        np.testing.assert_array_equal(eval_labels, model.assign(held_out))

    def test_assign_before_fit_raises(self):
        m = FrozenDUSTEDModel(n_clusters=3)
        with self.assertRaises(RuntimeError):
            m.assign(np.zeros((4, 8)))

    def test_frozen_centers_do_not_move_on_assign(self):
        rng = np.random.RandomState(1)
        train = rng.randn(300, 6)
        m = FrozenDUSTEDModel(n_clusters=4, random_state=0).fit(train)
        before = m.centers_.copy()
        m.assign(rng.randn(1000, 6))
        np.testing.assert_array_equal(before, m.centers_)

    def test_save_and_reload_round_trip(self):
        import tempfile
        from pathlib import Path

        rng = np.random.RandomState(2)
        train = rng.randn(200, 5)
        m = FrozenDUSTEDModel(n_clusters=3, random_state=0).fit(train)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.npz"
            m.save(path)
            m2 = FrozenDUSTEDModel(n_clusters=3).load(path)
            np.testing.assert_array_equal(m.assign(train), m2.assign(train))


if __name__ == "__main__":
    unittest.main()
