"""Tests: continuous recurrence discovery (§46, corrected design).

Regression tests for the mechanics: z-normalisation translation invariance
(the identity that makes caller-mean subtraction a no-op), boundary-safe
windows, eligible-neighbour restriction before the minimum, frozen global
transform, implanted-motif recovery, constant rejection, duplicate-episode
suppression, exposure-normalised concentration, source-index preservation.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.continuous_recurrence import (
    _z_normalise,
    boundary_safe_windows,
    closest_eligible_neighbour,
    compute_exposure_normalized_concentration,
    discover_cross_caller_motifs,
    suppress_duplicate_episodes,
    FixedGlobalTransform,
)


class TestInvariances(unittest.TestCase):
    def test_z_normalised_translation_invariance(self):
        rng = np.random.default_rng(42)
        window = rng.standard_normal((10, 4))
        offset = np.array([5.0, -3.0, 2.0, 1.0])
        np.testing.assert_allclose(_z_normalise(window), _z_normalise(window + offset), atol=1e-10)

    def test_within_caller_euclidean_translation_invariant(self):
        rng = np.random.default_rng(42)
        z_t, z_s, mu = rng.standard_normal(44), rng.standard_normal(44), rng.standard_normal(44)
        self.assertAlmostEqual(
            float(np.linalg.norm(z_t - z_s)),
            float(np.linalg.norm((z_t - mu) - (z_s - mu))),
            places=10,
        )

    def test_constant_coordinate_guarded(self):
        w = np.ones((5, 3))
        z = _z_normalise(w)
        self.assertTrue(np.all(np.isfinite(z)))


class TestWindowsAndNeighbours(unittest.TestCase):
    def test_boundary_safe_windows_never_cross(self):
        windows = boundary_safe_windows(
            total_frames=300, window_len=10, bout_boundaries=[0, 100, 200, 300]
        )
        self.assertGreater(len(windows), 0)
        for start, end in windows:
            crossed = [b for b in (0, 100, 200, 300) if start < b < end]
            self.assertEqual(len(crossed), 0, f"window [{start},{end}] crosses {crossed}")

    def test_closest_eligible_excludes_same_caller_before_min(self):
        rng = np.random.default_rng(0)
        windows = rng.standard_normal((5, 10, 4))
        # make window 4 an EXACT copy of window 0 — if same-caller windows
        # were eligible it would win; caller B copy must be chosen over the
        # same-caller near-identical window 1
        windows[4] = windows[0] * 1.0
        windows[1] = windows[0] + 0.001
        caller_ids = ["A", "A", "A", "B", "B"]
        result = closest_eligible_neighbour(
            query_idx=0, windows=windows, caller_ids=caller_ids, exclude_same_caller=True
        )
        self.assertEqual(result["neighbour_idx"], 4)

    def test_global_transform_frozen(self):
        rng = np.random.default_rng(42)
        train = rng.standard_normal((500, 44))
        ev = rng.standard_normal((50, 44))
        t = FixedGlobalTransform(n_components=10).fit(train)
        np.testing.assert_array_equal(t.transform(ev), t.transform(ev))
        self.assertEqual(t.transform(ev).shape, (50, 10))


class TestDiscovery(unittest.TestCase):
    def test_implanted_motif_recovered_across_bouts(self):
        rng = np.random.default_rng(42)
        motif = rng.standard_normal((20, 4))
        b1 = np.vstack([rng.standard_normal((50, 4)), motif, rng.standard_normal((50, 4))])
        b2 = np.vstack([rng.standard_normal((50, 4)), motif, rng.standard_normal((50, 4))])
        results = discover_cross_caller_motifs(
            [b1, b2], ["A", "B"], window_len=20, n_families=1, exclude_same_caller=True
        )
        self.assertGreaterEqual(len(results), 1)
        m = results[0]
        self.assertEqual(sorted(m["bouts"]), [0, 1])
        p0 = m["positions"][m["bouts"].index(0)]
        p1 = m["positions"][m["bouts"].index(1)]
        self.assertAlmostEqual(p0, 50, delta=5)
        self.assertAlmostEqual(p1, 50, delta=5)

    def test_constant_segment_not_reported(self):
        rng = np.random.default_rng(1)
        flat = np.ones((20, 4)) * 0.5
        noise = rng.standard_normal((80, 4))
        bout = np.vstack([noise, flat, noise])
        results = discover_cross_caller_motifs(
            [bout, bout + 1.0],
            ["A", "B"],
            window_len=20,
            n_families=5,
            exclude_same_caller=True,
        )
        for r in results:
            self.assertFalse(r.get("is_constant", False))

    def test_duplicate_episode_suppression(self):
        cands = [
            {"start": 50, "end": 70, "distance": 0.1},
            {"start": 51, "end": 71, "distance": 0.11},
            {"start": 52, "end": 72, "distance": 0.12},
            {"start": 200, "end": 220, "distance": 0.15},
        ]
        eps = suppress_duplicate_episodes(cands, min_gap=10)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["start"], 50)  # best of the overlapping trio kept

    def test_exposure_normalized_concentration(self):
        occ = [{"caller": "A", "bout_id": f"a{i}"} for i in range(10)] + [
            {"caller": "B", "bout_id": "b0"}
        ]
        res = compute_exposure_normalized_concentration(occ, {"A": 100, "B": 10})
        self.assertLess(res["herfindahl_H"], 0.6)
        self.assertAlmostEqual(res["rates"]["A"], res["rates"]["B"])

    def test_source_indices_preserved(self):
        rng = np.random.default_rng(42)
        bouts = [rng.standard_normal((100, 4)) for _ in range(3)]
        results = discover_cross_caller_motifs(
            bouts, ["A", "B", "C"], window_len=10, n_families=3, exclude_same_caller=True
        )
        for r in results:
            self.assertIn("bouts", r)
            self.assertIn("positions", r)
            self.assertIn("caller_ids", r)
            self.assertIn("distance", r)
            for b, c in zip(r["bouts"], r["caller_ids"]):
                self.assertEqual(c, ["A", "B", "C"][b])  # window->bout->caller chain


if __name__ == "__main__":
    unittest.main()


class TestRawEuclideanCounterexample(unittest.TestCase):
    """Cross-caller centering is NOT invariant under raw Euclidean — the
    synthetic counterexample (||X-Y||=sqrt(300), centered=0) must be DETECTABLE
    by any implementation that measures it; z-normalised matching IS invariant.
    This pins the distinction the centering ablation depends on."""

    def test_counterexample_detected(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3))
        mu_A = rng.standard_normal(3)
        mu_B = mu_A - np.array([10.0, 10.0, 10.0])  # ||muA - muB|| = sqrt(300)
        Y = X - mu_A + mu_B  # => centered X == centered Y exactly
        d_raw = np.linalg.norm(X - Y)
        d_centered = np.linalg.norm((X - mu_A) - (Y - mu_B))
        # per-row difference is [10,10,10]; Frobenius over 10 rows = sqrt(10*300)
        self.assertAlmostEqual(d_raw, np.sqrt(300 * X.shape[0]), places=6)
        self.assertAlmostEqual(d_centered, 0.0, places=6)
        self.assertNotAlmostEqual(d_raw, d_centered)  # the change is visible

    def test_z_norm_invariant_but_raw_not(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((10, 4))
        Y = rng.standard_normal((10, 4))
        muA, muB = rng.standard_normal(4), rng.standard_normal(4)
        from bioacoustic_pipeline.diagnostics.continuous_recurrence import _z_normalise

        self.assertLess(
            np.abs(_z_normalise(X) - _z_normalise(X - muA)).max(), 1e-10
        )  # z: invariant
        d_raw = np.linalg.norm(X - Y)
        d_ctr = np.linalg.norm((X - muA) - (Y - muB))
        # raw: cross-caller centering changes the distance (not an identity)
        self.assertGreater(abs(d_raw - d_ctr), 1e-3)


class TestTwoSidedEligibility(unittest.TestCase):
    def test_invalid_neighbour_cannot_win(self):
        rng = np.random.default_rng(0)
        good = rng.standard_normal((20, 4))
        bout = np.vstack([good, np.zeros((20, 4)), good + 0.01])  # padding mid
        other = rng.standard_normal((60, 4))
        results = discover_cross_caller_motifs(
            [bout, other],
            ["A", "B"],
            window_len=20,
            n_families=5,
            exclude_same_caller=True,
            hop=5,
        )
        for r in results:
            # no match may reference a window that is all-zero or constant
            self.assertFalse(r.get("is_constant", False))
            self.assertGreater(r["distance"], 0.0)

    def test_unavailable_query_gets_no_match_not_zero(self):
        # single valid window caller A vs caller B with only padding: A's
        # query has no eligible neighbour -> excluded, never distance-0
        rng = np.random.default_rng(1)
        a = rng.standard_normal((30, 4))
        b = np.zeros((30, 4))
        results = discover_cross_caller_motifs(
            [a, b],
            ["A", "B"],
            window_len=20,
            n_families=3,
            exclude_same_caller=True,
            hop=5,
        )
        self.assertEqual(len(results), 0)  # B's windows are all padding

    def test_frame_level_padding_mask(self):
        from bioacoustic_pipeline.diagnostics.continuous_recurrence import (
            valid_frame_mask,
        )

        traj = np.vstack([np.ones((5, 4)), np.zeros((5, 4)), np.ones((3, 4))])
        mask = valid_frame_mask(traj)
        self.assertEqual(mask.tolist(), [True] * 5 + [False] * 5 + [True] * 3)


class TestRegressionInvariant(unittest.TestCase):
    """Padding-append invariance + gap preservation (the quarantine contract)."""

    def _traj(self):
        rng = np.random.default_rng(0)
        t = np.vstack(
            [rng.standard_normal((20, 4)), np.zeros((10, 4)), rng.standard_normal((15, 4))]
        )
        return t

    def test_padding_append_changes_no_valid_results(self):
        from bioacoustic_pipeline.diagnostics.continuous_recurrence import (
            valid_frame_sequences,
        )

        t = self._traj()
        runs_before = valid_frame_sequences(t)
        padded = np.vstack([t, np.zeros((50, 4))])  # artificial tail padding
        runs_after = valid_frame_sequences(padded)
        self.assertEqual(len(runs_before), len(runs_after))  # no new occurrences
        for (s1, e1, f1), (s2, e2, f2) in zip(runs_before, runs_after):
            self.assertEqual((s1, e1), (s2, e2))  # positions unchanged
            np.testing.assert_array_equal(f1, f2)  # features unchanged

    def test_gap_preserved_no_invented_adjacency(self):
        from bioacoustic_pipeline.diagnostics.continuous_recurrence import (
            valid_frame_sequences,
        )

        t = self._traj()
        runs = valid_frame_sequences(t)
        self.assertEqual(len(runs), 2)
        # the gap (frames 20-29) is NOT bridged: run 1 ends at 20, run 2 starts at 30
        self.assertEqual(runs[0][1], 20)
        self.assertEqual(runs[1][0], 30)
        # original timestamps retained
        self.assertEqual(runs[1][2].shape[0], 15)
