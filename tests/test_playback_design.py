"""Playback-design (Phase 1) tests — offline, synthetic, torch-free.

Covers the leaf modules, the exemplar/stimulus pipeline, alias substitution, affect
descriptors, spatial routing, naturalness preflight, and the manifest assembler.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from bioacoustic_pipeline.experiments.playback_design import (
    affect_morphing,
    alias_substitution,
    candidate_selector,
    motif_variants,
    naturalness_preflight,
    playback_manifest,
    spatial_router,
    stimulus_builder,
)


class TestMotifVariants(unittest.TestCase):
    def test_variants_are_deterministic_and_sensible(self):
        p = [1, 2, 3]
        self.assertEqual(motif_variants.reverse(p), [3, 2, 1])
        self.assertEqual(motif_variants.shuffle(p, seed=0), motif_variants.shuffle(p, seed=0))
        self.assertEqual(len(motif_variants.token_delete(p, seed=0)), 2)
        ins = motif_variants.token_insert(p, vocab={9}, seed=0)
        self.assertEqual(len(ins), 4)
        self.assertIn(9, ins)
        rc = motif_variants.random_control({1, 2, 3}, 5, seed=0)
        self.assertEqual(len(rc), 5)


class TestCandidateSelector(unittest.TestCase):
    def _annotated(self):
        return [
            {"token": 0, "audio": np.zeros(8, dtype=np.float32), "sr": 16000, "segment_id": "a"},
            {"token": 0, "audio": np.zeros(8, dtype=np.float32), "sr": 16000, "segment_id": "b"},
            {"token": 1, "audio": np.zeros(8, dtype=np.float32), "sr": 16000, "segment_id": "c"},
            {
                "token": -1,
                "audio": np.zeros(8, dtype=np.float32),
                "sr": 16000,
                "segment_id": "drop",
            },
        ]

    def test_pool_groups_and_drops_unmapped(self):
        pool = candidate_selector.build_exemplar_pool(self._annotated())
        self.assertEqual(set(pool), {0, 1})
        self.assertEqual(len(pool[0]), 2)

    def test_select_candidates_filters_by_exemplar_and_support(self):
        @dataclass
        class M:
            pattern: tuple
            support: int
            score: float = 0.5
            motif_id: str = "m"

        pool = candidate_selector.build_exemplar_pool(self._annotated())
        typical = candidate_selector.typical_exemplars(pool)
        motifs = [
            M((0, 1), 5),
            M((0, 2), 5),
            M((0,), 9),
        ]  # 2nd has token w/o exemplar; 3rd too short
        sel, vocab = candidate_selector.select_candidates(
            motifs, typical, n_motifs=5, min_support=2
        )
        self.assertEqual([tuple(m.pattern) for m in sel], [(0, 1)])
        self.assertEqual(vocab, {0, 1})


class TestStimulusBuilder(unittest.TestCase):
    def _pool(self):
        segs = [
            {
                "token": t,
                "audio": np.ones(1600, dtype=np.float32) * (t + 1) * 0.1,
                "sr": 16000,
                "segment_id": f"tok{t}",
            }
            for t in range(3)
        ]
        return candidate_selector.typical_exemplars(candidate_selector.build_exemplar_pool(segs))

    def test_concatenates_and_provenance(self):
        pool = self._pool()
        audio, prov = stimulus_builder.build_stimulus_audio(
            [0, 1, 2], pool, 16000, isi_ms=50, seed=0
        )
        self.assertGreater(audio.size, 0)
        self.assertEqual(len(prov), 3)

    def test_missing_token_raises(self):
        pool = self._pool()
        with self.assertRaises(KeyError):
            stimulus_builder.build_stimulus_audio([0, 99], pool, 16000, seed=0)


@dataclass
class _FakeGroup:
    group_id: int
    member_tokens: list


@dataclass
class _FakeAliasReport:
    groups: list
    _by_token: dict

    def group_of(self, token):
        return self._by_token.get(int(token))


class TestAliasSubstitution(unittest.TestCase):
    def _report(self):
        # token 1 and 2 are aliases (group 0); token 5 is a singleton in another group
        g0 = _FakeGroup(0, [1, 2])
        g1 = _FakeGroup(1, [5])
        return _FakeAliasReport([g0, g1], {1: g0, 2: g0, 5: g1})

    def test_alias_picks_same_group_member(self):
        rep = self._report()
        new, info = alias_substitution.alias_substitute_pattern([1, 3, 4], rep, seed=0)
        self.assertIsNotNone(new)
        self.assertEqual(
            rep.group_of(info["substituted"]).group_id, rep.group_of(info["original"]).group_id
        )

    def test_nonalias_picks_different_group(self):
        rep = self._report()
        new, info = alias_substitution.nonalias_control_pattern(
            [1, 3, 4], rep, {1, 2, 3, 4, 5}, seed=0
        )
        self.assertIsNotNone(new)
        orig_g = rep.group_of(info["original"])
        new_g = rep.group_of(info["substituted"])
        self.assertTrue(new_g is None or new_g.group_id != (orig_g.group_id if orig_g else None))


class TestAffectMorphing(unittest.TestCase):
    def test_no_affect_model_marks_not_available(self):
        d = affect_morphing.morph_descriptors([])
        self.assertFalse(d["available"])
        self.assertEqual(d["note"], affect_morphing.NOT_AVAILABLE)

    def test_trajectory_descriptors_when_affect_present(self):
        @dataclass
        class Seg:
            affect_mu: np.ndarray

        segs = [Seg(np.array([float(i)], dtype=np.float32)) for i in range(4)]
        d = affect_morphing.morph_descriptors(segs)
        self.assertTrue(d["available"])
        self.assertEqual(len(d["natural"]), 4)
        self.assertEqual(len(d["flattened"]), 4)
        self.assertEqual(list(d["reversed"]), list(reversed(d["natural"])))


class TestSpatialRouter(unittest.TestCase):
    def test_vbap_two_speakers_normalize_to_one(self):
        gains = spatial_router.vbap_gains(45.0, [0.0, 90.0, 180.0, 270.0])
        self.assertLessEqual(len(gains), 2)
        self.assertAlmostEqual(sum(gains.values()), 1.0, places=4)

    def test_spatial_condition_labels(self):
        self.assertEqual(
            spatial_router.spatial_condition(True, "unicast"), spatial_router.CORRECT_DIRECTION
        )
        self.assertEqual(
            spatial_router.spatial_condition(False, "unicast"), spatial_router.UNICAST_FROM_WRONG
        )


class TestNaturalnessPreflight(unittest.TestCase):
    def test_identical_sequence_passes_disjoint_fails(self):
        fp = np.ones(10, dtype=np.float64)
        same = naturalness_preflight.fingerprint_sequence_distance([fp, fp], [fp, fp])
        self.assertGreater(same["mean_cosine"], 0.99)
        other = np.array([-c for c in fp], dtype=np.float64)
        diff = naturalness_preflight.fingerprint_sequence_distance([fp], [other])
        self.assertLess(diff["mean_cosine"], 0.1)
        self.assertGreaterEqual(diff["dtw"], 0.0)

    def test_preflight_pass_flag(self):
        sr = 16000
        nat = np.zeros(sr, dtype=np.float32)
        pf_seq = [np.ones(8)]
        r = naturalness_preflight.preflight_stimulus(nat, nat, sr, pf_seq, pf_seq)
        self.assertTrue(r["pass"])
        r2 = naturalness_preflight.preflight_stimulus(nat, nat, sr, [-np.ones(8)], [np.ones(8)])
        self.assertFalse(r2["pass"])


class TestPlaybackManifest(unittest.TestCase):
    def test_shape_caveats_predictions(self):
        m = playback_manifest.build_manifest(
            "motif_disruption",
            "bat",
            [{"stimulus_id": "x_natural", "condition": "natural", "token_sequence": [1, 2]}],
        )
        self.assertEqual(m["experiment"], "motif_disruption")
        self.assertEqual(m["n_stimuli"], 1)
        self.assertTrue(m["caveats"])
        self.assertIn("natural_vs_disrupted", m["predicted_behavior"])
        self.assertIn("response_measures", m)

    def test_unknown_experiment_rejected(self):
        with self.assertRaises(ValueError):
            playback_manifest.build_manifest("turing_test", "bat", [])


if __name__ == "__main__":
    unittest.main()
