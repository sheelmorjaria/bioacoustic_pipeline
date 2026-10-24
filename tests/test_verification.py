"""Tests: caller-verification instruments (diagnostics.verification).

EER math is pinned to known answers; pair builders are checked for leakage
(no same-file genuine pairs); permutation/bootstrap helpers are exercised on
synthetic embeddings with known structure. No torch, no checkpoints, no network.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.verification import (
    build_genuine_impostor_pairs,
    calculate_eer,
    cluster_bootstrap_files,
    label_permutation_null,
    permute_labels_at_file_level,
)


class TestCalculateEer(unittest.TestCase):
    def test_separable_distributions_give_eer_near_zero(self):
        rng = np.random.RandomState(0)
        pos = rng.normal(10.0, 1.0, 2000)
        neg = rng.normal(-10.0, 1.0, 2000)
        r = calculate_eer(pos, neg)
        self.assertLess(r.eer, 0.01)
        self.assertEqual(r.n_pos, 2000)
        self.assertEqual(r.n_neg, 2000)
        self.assertGreaterEqual(r.threshold, neg.max())
        self.assertLessEqual(r.threshold, pos.min())

    def test_identical_distributions_give_eer_half(self):
        rng = np.random.RandomState(1)
        scores = rng.normal(0.0, 1.0, 4000)
        r = calculate_eer(scores[:2000], scores[2000:])
        self.assertAlmostEqual(r.eer, 0.5, delta=0.02)

    def test_known_overlap_matches_analytic_value(self):
        # Equal-variance gaussians d' apart: EER = Phi(-d'/2). d'=2 -> 0.1587.
        rng = np.random.RandomState(2)
        pos = rng.normal(1.0, 1.0, 20000)
        neg = rng.normal(-1.0, 1.0, 20000)
        r = calculate_eer(pos, neg)
        self.assertAlmostEqual(r.eer, 0.1587, delta=0.02)

    def test_monotonicity_shrinking_separation_raises_eer(self):
        rng = np.random.RandomState(3)
        eers = []
        for shift in (4.0, 2.0, 0.5):
            pos = rng.normal(shift, 1.0, 4000)
            neg = rng.normal(0.0, 1.0, 4000)
            eers.append(calculate_eer(pos, neg).eer)
        self.assertLess(eers[0], eers[1])
        self.assertLess(eers[1], eers[2])

    def test_eer_role_swap_complements_and_sign_flip_matches(self):
        # EER is defined for an oriented score (higher = genuine). Swapping the
        # class roles without flipping the score yields the exact complement;
        # swapping roles AND negating the score yields the same EER.
        rng = np.random.RandomState(4)
        pos = rng.normal(1.0, 1.0, 4000)
        neg = rng.normal(-1.0, 1.0, 4000)
        a = calculate_eer(pos, neg).eer
        b = calculate_eer(neg, pos).eer
        self.assertAlmostEqual(a + b, 1.0, delta=1e-9)
        c = calculate_eer(-neg, -pos).eer
        self.assertAlmostEqual(a, c, delta=1e-9)

    def test_empty_inputs_raise(self):
        with self.assertRaises(ValueError):
            calculate_eer([], [0.1, 0.2])
        with self.assertRaises(ValueError):
            calculate_eer([0.1], [])


class TestBuildGenuineImpostorPairs(unittest.TestCase):
    def _fixture(self, n_callers=8, files_per_caller=5, dim=16, seed=7):
        """Caller-clustered embeddings, one file per caller offset cluster."""
        rng = np.random.RandomState(seed)
        embs, callers, files = [], [], []
        centers = rng.randn(n_callers, dim) * 6.0
        for c in range(n_callers):
            for f in range(files_per_caller):
                for _ in range(3):  # 3 windows per file
                    embs.append(centers[c] + 0.1 * rng.randn(dim))
                    callers.append(c)
                    files.append(f"caller{c}_file{f}.wav")
        return np.asarray(embs), np.asarray(callers), np.asarray(files)

    def test_genuine_pairs_same_caller_different_file(self):
        embs, callers, files = self._fixture()
        sample = build_genuine_impostor_pairs(embs, callers, files, n_pairs=200, seed=0)
        self.assertEqual(len(sample.labels), 200)
        self.assertEqual(sample.n_genuine + sample.n_impostor, 200)
        for i, j, lab in zip(sample.pair_i, sample.pair_j, sample.labels):
            same_caller = callers[i] == callers[j]
            same_file = files[i] == files[j]
            if lab == 1:
                self.assertTrue(same_caller)
                self.assertFalse(same_file)  # anti-leakage: never same file
            else:
                self.assertFalse(same_caller)

    def test_balanced_and_capped(self):
        embs, callers, files = self._fixture()
        sample = build_genuine_impostor_pairs(
            embs, callers, files, n_pairs=100, max_per_caller=4, seed=0
        )
        self.assertEqual(sample.n_genuine, sample.n_impostor)
        # per-caller cap on genuine side
        genuine_callers = [callers[i] for i, lab in zip(sample.pair_i, sample.labels) if lab == 1]
        _, counts = np.unique(genuine_callers, return_counts=True)
        self.assertLessEqual(counts.max(), 4)

    def test_deterministic_under_seed(self):
        embs, callers, files = self._fixture()
        a = build_genuine_impostor_pairs(embs, callers, files, n_pairs=60, seed=3)
        b = build_genuine_impostor_pairs(embs, callers, files, n_pairs=60, seed=3)
        np.testing.assert_array_equal(a.pair_i, b.pair_i)
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_allclose(a.scores, b.scores)

    def test_scores_higher_for_genuine_in_clustered_fixture(self):
        embs, callers, files = self._fixture()
        sample = build_genuine_impostor_pairs(embs, callers, files, n_pairs=400, seed=0)
        g = sample.scores[sample.labels == 1]
        i = sample.scores[sample.labels == 0]
        self.assertGreater(g.mean(), i.mean())


class TestFileLevelPermutation(unittest.TestCase):
    def test_labels_permuted_whole_file_at_a_time(self):
        rng = np.random.RandomState(0)
        files = np.array([f"f{i}.wav" for i in range(10) for _ in range(4)])
        callers = np.array([i % 3 for i in range(10) for _ in range(4)])
        perm = permute_labels_at_file_level(callers, files, rng)
        # every file now carries one consistent (possibly changed) label
        for f in np.unique(files):
            vals = np.unique(perm[files == f])
            self.assertEqual(len(vals), 1)
        # the multiset of file->label assignments is a shuffle of the original
        orig_map = {f: callers[files == f][0] for f in np.unique(files)}
        perm_map = {f: perm[files == f][0] for f in np.unique(files)}
        self.assertEqual(sorted(orig_map.values()), sorted(perm_map.values()))

    def test_deterministic(self):
        files = np.array(["a", "a", "b", "b", "c", "c"])
        callers = np.array([0, 0, 1, 1, 2, 2])
        p1 = permute_labels_at_file_level(callers, files, np.random.RandomState(5))
        p2 = permute_labels_at_file_level(callers, files, np.random.RandomState(5))
        np.testing.assert_array_equal(p1, p2)


class TestPermutationNull(unittest.TestCase):
    def _eer_stat(self, embeddings, caller_ids, file_ids):
        sample = build_genuine_impostor_pairs(embeddings, caller_ids, file_ids, n_pairs=200, seed=0)
        return calculate_eer(
            sample.scores[sample.labels == 1], sample.scores[sample.labels == 0]
        ).eer

    def test_signal_gives_minimal_p(self):
        rng = np.random.RandomState(0)
        centers = rng.randn(6, 8) * 5.0
        embs, callers, files = [], [], []
        for c in range(6):
            for f in range(6):
                for _ in range(3):
                    embs.append(centers[c] + 0.05 * rng.randn(8))
                    callers.append(c)
                    files.append(f"c{c}_f{f}")
        res = label_permutation_null(
            self._eer_stat,
            np.asarray(embs),
            np.asarray(callers),
            np.asarray(files),
            n_permutations=50,
            seed=0,
            direction="lower",
        )
        self.assertAlmostEqual(res.p_value, 1.0 / 51)
        self.assertLess(res.observed, res.null_mean)

    def test_noise_gives_non_significant_p(self):
        rng = np.random.RandomState(1)
        embs = rng.randn(300, 8)
        callers = rng.randint(0, 5, 300)
        files = np.array([f"f{i // 5}" for i in range(300)])
        res = label_permutation_null(
            self._eer_stat,
            embs,
            callers,
            files,
            n_permutations=50,
            seed=0,
            direction="lower",
        )
        self.assertGreater(res.p_value, 0.05)


class TestClusterBootstrapFiles(unittest.TestCase):
    def _eer_stat(self, embeddings, caller_ids, file_ids):
        sample = build_genuine_impostor_pairs(embeddings, caller_ids, file_ids, n_pairs=200, seed=0)
        return calculate_eer(
            sample.scores[sample.labels == 1], sample.scores[sample.labels == 0]
        ).eer

    def test_ci_brackets_and_deterministic(self):
        rng = np.random.RandomState(2)
        centers = rng.randn(5, 8) * 4.0
        embs, callers, files = [], [], []
        for c in range(5):
            for f in range(8):
                embs.append(centers[c] + 0.2 * rng.randn(8))
                callers.append(c)
                files.append(f"c{c}_f{f}")
        args = (np.asarray(embs), np.asarray(callers), np.asarray(files))
        r1 = cluster_bootstrap_files(self._eer_stat, *args, n_bootstrap=40, seed=1)
        r2 = cluster_bootstrap_files(self._eer_stat, *args, n_bootstrap=40, seed=1)
        self.assertEqual(r1.ci_lo, r2.ci_lo)
        self.assertLessEqual(r1.ci_lo, r1.observed + 1e-9)
        self.assertGreaterEqual(r1.ci_hi, r1.observed - 1e-9)
        self.assertEqual(r1.n_bootstrap, 40)


if __name__ == "__main__":
    unittest.main()


class TestPairedBootstrapDelta(unittest.TestCase):
    def test_paired_delta_shares_resamples_and_brackets(self):
        from bioacoustic_pipeline.diagnostics.verification import (
            paired_bootstrap_delta,
        )

        rng = np.random.RandomState(0)
        centers = rng.randn(6, 8) * 4.0
        embs_a, embs_b, callers, files = [], [], [], []
        for c in range(6):
            for f in range(8):
                embs_a.append(centers[c] + 0.05 * rng.randn(8))
                embs_b.append(centers[c] + 3.5 * rng.randn(8))  # materially noisier arm
                callers.append(c)
                files.append(f"c{c}_f{f}")
        a, b = np.asarray(embs_a), np.asarray(embs_b)
        ca, fa = np.asarray(callers), np.asarray(files)

        def eer_of(emb):
            def stat(X, y, z):
                s = build_genuine_impostor_pairs(X, y, z, n_pairs=200, seed=0)
                return calculate_eer(s.scores[s.labels == 1], s.scores[s.labels == 0]).eer

            return stat

        r = paired_bootstrap_delta(eer_of(a), eer_of(b), a, b, ca, fa, n_bootstrap=40, seed=1)
        # arm B (noisier) should have higher EER: delta = b - a > 0
        self.assertGreater(r.observed, 0.0)
        self.assertGreater(r.ci_lo, 0.0)
        r2 = paired_bootstrap_delta(eer_of(a), eer_of(b), a, b, ca, fa, n_bootstrap=40, seed=1)
        self.assertEqual(r.ci_lo, r2.ci_lo)  # deterministic

    def test_identical_arms_give_zero_delta(self):
        from bioacoustic_pipeline.diagnostics.verification import (
            paired_bootstrap_delta,
        )

        rng = np.random.RandomState(1)
        centers = rng.randn(5, 8) * 4.0
        embs, callers, files = [], [], []
        for c in range(5):
            for f in range(6):
                embs.append(centers[c] + 0.2 * rng.randn(8))
                callers.append(c)
                files.append(f"c{c}_f{f}")
        e, ca, fa = np.asarray(embs), np.asarray(callers), np.asarray(files)

        def stat(X, y, z):
            s = build_genuine_impostor_pairs(X, y, z, n_pairs=150, seed=0)
            return calculate_eer(s.scores[s.labels == 1], s.scores[s.labels == 0]).eer

        r = paired_bootstrap_delta(stat, stat, e, e, ca, fa, n_bootstrap=20, seed=0)
        self.assertEqual(r.observed, 0.0)
        self.assertEqual(r.ci_lo, 0.0)
        self.assertEqual(r.ci_hi, 0.0)


class TestImpostorScopeStratification(unittest.TestCase):
    """§46 twin confound: impostor pairs stratified by group (twin family)."""

    def _fixture(self):
        rng = np.random.RandomState(0)
        embs, callers, files, groups = [], [], [], []
        for g in range(4):  # 4 twin families
            for c in range(2 * g, 2 * g + 2):  # 2 callers per family
                for f in range(6):
                    embs.append(rng.randn(8))
                    callers.append(c)
                    files.append(f"g{g}_c{c}_f{f}")
                    groups.append(g)
        return (
            np.asarray(embs),
            np.asarray(callers),
            np.asarray(files),
            np.asarray(groups),
        )

    def test_within_group_impostors_share_twin_family(self):
        from bioacoustic_pipeline.diagnostics.verification import (
            build_genuine_impostor_pairs,
        )

        e, c, f, g = self._fixture()
        s = build_genuine_impostor_pairs(
            e,
            c,
            f,
            group_ids=g,
            impostor_scope="within_group",
            n_pairs=100,
            seed=0,
        )
        imposter = s.labels == 0
        self.assertGreater(int(imposter.sum()), 0)
        for i, j in zip(s.pair_i[imposter], s.pair_j[imposter]):
            self.assertEqual(g[i], g[j])  # same twin family
            self.assertNotEqual(c[i], c[j])  # different callers

    def test_cross_group_impostors_differ_in_family(self):
        from bioacoustic_pipeline.diagnostics.verification import (
            build_genuine_impostor_pairs,
        )

        e, c, f, g = self._fixture()
        s = build_genuine_impostor_pairs(
            e,
            c,
            f,
            group_ids=g,
            impostor_scope="cross_group",
            n_pairs=100,
            seed=0,
        )
        imposter = s.labels == 0
        self.assertGreater(int(imposter.sum()), 0)
        for i, j in zip(s.pair_i[imposter], s.pair_j[imposter]):
            self.assertNotEqual(g[i], g[j])

    def test_default_scope_ignores_groups(self):
        from bioacoustic_pipeline.diagnostics.verification import (
            build_genuine_impostor_pairs,
        )

        e, c, f, g = self._fixture()
        a = build_genuine_impostor_pairs(e, c, f, n_pairs=100, seed=0)
        b = build_genuine_impostor_pairs(e, c, f, group_ids=g, n_pairs=100, seed=0)
        np.testing.assert_array_equal(a.pair_i, b.pair_i)

    def test_bad_scope_raises(self):
        from bioacoustic_pipeline.diagnostics.verification import (
            build_genuine_impostor_pairs,
        )

        e, c, f, g = self._fixture()
        with self.assertRaises(ValueError):
            build_genuine_impostor_pairs(e, c, f, group_ids=g, impostor_scope="nope")
