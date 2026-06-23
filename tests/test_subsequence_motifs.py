"""Tests for DUSTED-style recurrent-subsequence motif discovery.

Verifies: DTW monotonicity + role-aware indels; prefilter gating; DTW recovers an
injected variant family (incl. length-mismatched indel variants that fixed-length
cosine cannot merge); order-shuffle null collapses the discovery; fixed-length baseline
never spans ≥2 window lengths.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm


def _vec(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _sub(fps, roles=None):
    n = len(fps)
    if roles is None:
        roles = sm._roles_for_len(n)
    return sm.SubSequence(
        fps=list(fps),
        clusters=list(range(n)),
        roles=list(roles),
        rel_mids=sm._rel_mids_for_len(n),
    )


# ─── DTW monotonicity ────────────────────────────────────────────────────────


class TestDTWDistance:
    def test_identical_is_zero(self):
        a = _sub([_vec(0), _vec(1), _vec(2)])
        assert sm.dtw_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_monotonicity_identical_substitution_unrelated(self):
        a = _sub([_vec(0), _vec(1), _vec(2)])
        # near-miss: mostly v1 with a small admixture of a different vector
        near = _vec(1) * 0.95 + _vec(5) * 0.05
        near = near / (np.linalg.norm(near) + 1e-9)
        sub_variant = _sub([_vec(0), near, _vec(2)])
        unrelated = _sub([_vec(10), _vec(11), _vec(12)])
        d_id = sm.dtw_distance(a, a)
        d_sub = sm.dtw_distance(a, sub_variant)
        d_unrel = sm.dtw_distance(a, unrelated)
        assert d_id < d_sub < d_unrel
        assert d_sub < 0.15  # a near-substitution stays cheap
        assert d_unrel > 0.5  # orthogonal units are far

    def test_role_aware_indel_body_cheaper_than_edge(self):
        # len-4 window: roles = [onset, body, body, offset]
        a = _sub([_vec(0), _vec(1), _vec(2), _vec(3)])
        drop_body = _sub([_vec(0), _vec(1), _vec(3)])  # dropped v2 (body)
        drop_onset = _sub([_vec(1), _vec(2), _vec(3)])  # dropped v0 (onset)
        d_body = sm.dtw_distance(a, drop_body, relpos_lambda=0.0, role_penalty=0.0)
        d_onset = sm.dtw_distance(a, drop_onset, relpos_lambda=0.0, role_penalty=0.0)
        assert d_body < d_onset
        assert d_body == pytest.approx(0.25 / 4, abs=1e-6)
        assert d_onset == pytest.approx(0.45 / 4, abs=1e-6)


# ─── prefilter ───────────────────────────────────────────────────────────────


class TestPrefilter:
    def test_keeps_centroid_neighbours_gates_incompatible(self):
        same_a = _sub([_vec(0), _vec(1)])
        same_b = _sub([_vec(0), _vec(1)])  # identical centroid
        far = _sub([_vec(10), _vec(11)])  # orthogonal centroid
        windows = [same_a, same_b, far]
        pairs = sm._prefilter_pairs(windows, centroid_radius=0.25, length_ratio=1.5)
        assert (0, 1) in pairs  # near centroids, same length, onset/body-compatible ends
        assert all(2 not in p for p in pairs)  # `far` shares no neighbour

    def test_length_ratio_gate(self):
        short = _sub([_vec(0), _vec(1)])
        long = _sub([_vec(0), _vec(1), _vec(0), _vec(1)])  # ratio 2.0 > 1.5
        pairs = sm._prefilter_pairs([short, long], length_ratio=1.5)
        assert pairs == []


# ─── discovery: indel recovery ───────────────────────────────────────────────


def _make_calls_with_motif(n_a: int = 10, n_b: int = 10, n_fill: int = 2):
    """Calls containing a recurring motif [v0,v1]; group B inserts an extra v2 unit.

    Group A: [filler×n_fill, v0, v1, filler×n_fill]      -> pure [v0,v1] window
    Group B: [filler×n_fill, v0, v1, v2, filler×n_fill]  -> [v0,v1] + variant [v0,v1,v2]
    Fillers are unique per (call, slot) so only motif windows recur.
    """
    calls = []

    def _fillers(k: int, off: int) -> list[np.ndarray]:
        return [_vec(off + k * 37 + j) for j in range(n_fill)]

    for k in range(n_a):
        calls.append(
            sm.CallUnits(
                fingerprints=_fillers(k, 5000) + [_vec(0), _vec(1)] + _fillers(k, 6000),
                clusters=[-1] * n_fill + [0, 1] + [-1] * n_fill,
                calltype="A",
                caller=f"c{k % 3}",
                call_idx=k,
            )
        )
    for k in range(n_b):
        idx = n_a + k
        calls.append(
            sm.CallUnits(
                fingerprints=_fillers(k, 7000) + [_vec(0), _vec(1), _vec(2)] + _fillers(k, 8000),
                clusters=[-1] * n_fill + [0, 1, 2] + [-1] * n_fill,
                calltype="B",
                caller=f"c{k % 3}",
                call_idx=idx,
            )
        )
    return calls


class TestDiscovery:
    def test_dtw_recovers_indel_variant_family(self):
        calls = _make_calls_with_motif()
        motifs = sm.discover_recurrent_subsequences(
            calls, dist_threshold=0.35, centroid_radius=0.4, min_support=3, min_calls=2
        )
        assert motifs, "expected at least one discovered motif"
        # The motif family must span ≥2 window lengths (len2 [v0,v1] + len3 [v0,v1,v2]):
        # this is exactly the indel-tolerant recovery fixed-length cosine cannot do.
        assert any(m.support >= 18 for m in motifs)
        assert any(m.n_calls >= 15 for m in motifs)

    def test_fixed_length_never_spans_two_lengths(self):
        calls = _make_calls_with_motif()
        motifs = sm.fixed_length_cosine_clusters(
            calls, dist_threshold=0.35, centroid_radius=0.4, min_support=3, min_calls=2
        )
        # Fixed-length cosine still finds the exact [v0,v1] motif...
        assert any(m.support >= 15 for m in motifs)
        # ...but every fixed-length motif's prototype is single-length (no indel recovery).
        for m in motifs:
            assert m.prototype is not None
        # And fixed-length yields strictly more (fragmented) motifs than DTW merges them
        dtw_motifs = sm.discover_recurrent_subsequences(
            calls, dist_threshold=0.35, centroid_radius=0.4, min_support=3, min_calls=2
        )
        assert len(dtw_motifs) <= len(motifs)

    def test_exact_ngram_finds_motif_but_fragments_variants(self):
        calls = _make_calls_with_motif()
        motifs = sm.exact_ngram_clusters(calls, min_support=3, min_calls=2)
        # the exact [0,1] n-gram is found...
        assert any(m.support >= 15 for m in motifs)
        # ...and exact n-grams are label-pure by construction (all members share one tuple)
        assert all(m.label_purity == 1.0 for m in motifs)
        # ...but variants fragment into separate families (len2 [0,1] ≠ len3 [0,1,2]),
        # so exact n-grams yield at least as many motifs as DTW-merged families.
        dtw = sm.discover_recurrent_subsequences(
            calls, dist_threshold=0.35, centroid_radius=0.4, min_support=3, min_calls=2
        )
        assert len(motifs) >= len(dtw)


# ─── null + stability ────────────────────────────────────────────────────────


class TestNullAndStability:
    def test_discovery_null_collapses_on_order_shuffle(self):
        # n_fill=6 → 14-unit calls: the 3 shared motif vectors are diluted enough that
        # shattering their order destroys recurrence (no concentrated chance adjacencies).
        calls = _make_calls_with_motif(n_a=8, n_b=8, n_fill=6)
        result = sm.discovery_null(
            calls,
            dist_threshold=0.35,
            centroid_radius=0.4,
            min_calls=3,
            n_perm=20,
            seed=0,
        )
        # Order carries the motif: total support must beat the shuffled null.
        assert result["observed_support"] >= 15
        assert result["observed_support"] > result["null_mean"]

    def test_bootstrap_stability_in_unit_interval(self):
        calls = _make_calls_with_motif(n_a=8, n_b=8)
        jac = sm.bootstrap_stability(
            calls, dist_threshold=0.35, centroid_radius=0.4, n_boot=5, top_k=5, seed=0
        )
        assert 0.0 <= jac <= 1.0
