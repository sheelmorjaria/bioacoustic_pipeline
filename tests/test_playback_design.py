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
    audio_splicer,
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


class TestAudioSplicer(unittest.TestCase):
    """Compositional + standard DSP splice conditions (exemplar-based, deterministic)."""

    def _units(self):
        # distinct amplitudes so isolated/reversed/natural are distinguishable
        return [np.ones(2000, dtype=np.float32) * 0.1, np.ones(2000, dtype=np.float32) * 0.5]

    def test_extract_unit_audio_slices_segments(self):
        audio = np.arange(16000, dtype=np.float32)
        units = audio_splicer.extract_unit_audio(0.0, 1.0, [(0.0, 0.25), (0.25, 0.5)], audio, 16000)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].size, int(0.25 * 16000))

    def test_generate_all_counterfactuals_has_five_conditions(self):
        donor = [np.ones(2000, dtype=np.float32) * 0.3]  # donor from another caller
        res = audio_splicer.generate_all_counterfactuals(
            self._units(), 16000, donor_pool=donor, seed=0
        )
        conds = [r.condition for r in res]
        self.assertEqual(
            set(conds),
            {
                "natural",
                "order_shuffled",
                "onset_offset_swap",
                "caller_swapped",
                "amplitude_matched",
            },
        )

    def test_generate_all_counterfactuals_caller_swapped_falls_back_without_donor(self):
        # documented behaviour: empty donor_pool -> caller_swapped becomes natural
        res = audio_splicer.generate_all_counterfactuals(self._units(), 16000, seed=0)
        self.assertNotIn("caller_swapped", [r.condition for r in res])

    def test_isolated_unit_renders_one_unit_and_clamps_index(self):
        units = self._units()
        iso = audio_splicer.splice_isolated_unit(units, 16000, idx=1)
        self.assertEqual(iso.condition, "isolated_unit")
        self.assertEqual(iso.n_units, 1)
        self.assertEqual(iso.provenance["isolated_index"], 1)
        # out-of-range index clamps to 0, not an error
        iso0 = audio_splicer.splice_isolated_unit(units, 16000, idx=99)
        self.assertEqual(iso0.provenance["isolated_index"], 0)
        # empty units → empty audio, no crash
        empty = audio_splicer.splice_isolated_unit([], 16000, idx=0)
        self.assertEqual(empty.audio.size, 0)

    def test_reversed_is_ba_not_random(self):
        units = self._units()
        rev = audio_splicer.splice_reversed(units, 16000)
        self.assertEqual(rev.condition, "reversed")
        self.assertEqual(rev.provenance["order"], "reversed")
        # reversed concatenation should differ from natural (A then B vs B then A)
        nat = audio_splicer.splice_natural(units, 16000)
        self.assertFalse(np.allclose(rev.audio[:2000], nat.audio[:2000]))
        # single unit → falls back to natural (nothing to reverse)
        single = audio_splicer.splice_reversed([units[0]], 16000)
        self.assertEqual(single.condition, "natural")

    def test_compositional_set_is_five_distinct_conditions(self):
        res = audio_splicer.generate_compositional_counterfactuals(self._units(), 16000, 0, 1)
        conds = [r.condition for r in res]
        self.assertEqual(
            conds,
            ["natural", "isolated_unit", "isolated_unit", "reversed", "amplitude_matched"],
        )
        # isolated A is unit 0 (amplitude 0.1), isolated B is unit 1 (amplitude 0.5)
        iso_a, iso_b = res[1], res[2]
        self.assertLess(np.max(np.abs(iso_a.audio)), np.max(np.abs(iso_b.audio)))


if __name__ == "__main__":
    unittest.main()
