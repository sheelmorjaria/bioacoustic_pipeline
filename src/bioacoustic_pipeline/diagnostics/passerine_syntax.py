"""Passerine combinatorial syntax — mechanics for the Parus track (§46).

Transition-safe by construction: codebook assignments are grouped per FILE
and bigram counts never cross file boundaries (a boundary is not a vocal
transition). The permutation null shuffles assignments WITHIN files, so the
unigram content of every file is preserved and only the order structure is
tested. Ceiling: a non-random transition is a T2 structural association —
never a T4 functional/semantic claim; only Track C biology can elevate it.

The passerine analysis fits a FRESH codebook on a Parus train split (the
Stage-2 frozen-vocabulary pattern); the Stage-2 bat codebook is a transfer
ablation, not the primary vocabulary.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

PARUS_SPECIES = ("Parus major", "Cyanistes caeruleus")


def filter_parus_recordings(manifest_rows) -> list[dict]:
    """Keep Parus major + Cyanistes caeruleus rows."""
    return [r for r in manifest_rows if r.get("species_scientific") in PARUS_SPECIES]


def build_transition_safe_intervals(intervals: list[dict]) -> list[dict]:
    """Group intervals by file (stable order) — transitions never cross."""
    by_file: dict = {}
    for iv in intervals:
        by_file.setdefault(iv["file_id"], []).append(iv)
    out: list[dict] = []
    for _f, ivs in by_file.items():
        out.extend(ivs)
    return out


def build_transition_matrix(sequences, *, n_clusters: int) -> np.ndarray:
    """Bigram counts over per-file codebook sequences (no cross-file pairs)."""
    m = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for seq in sequences:
        s = np.asarray(seq)
        if len(s) < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            m[int(a), int(b)] += 1
    return m


def _max_bigram_count(sequences) -> float:
    """Statistic: the count of the single most frequent bigram.

    Calibrated against a within-file shuffle that preserves unigrams — so
    unigram concentration alone cannot drive significance.
    """
    counts: dict = {}
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    return float(max(counts.values())) if counts else 0.0


def test_transition_significance(
    sequences, *, n_clusters: int, n_perm: int = 200, seed: int = 0
) -> dict:
    """Permutation null on transition structure (within-file shuffles)."""
    rng = np.random.RandomState(seed)
    observed = _max_bigram_count(sequences)
    exceed = 0
    for _ in range(n_perm):
        shuffled = [list(rng.permutation(s)) for s in sequences]
        if _max_bigram_count(shuffled) >= observed:
            exceed += 1
    return {
        "statistic": observed,
        "p_value": (1.0 + exceed) / (n_perm + 1.0),
        "n_permutations": n_perm,
        "null": "within-file assignment shuffle (unigram-preserving)",
    }


def evaluate_compositionality(*, transition, p_value: float, is_non_random: bool) -> dict:
    """Tier labeling for a codebook transition — T2 structural at most."""
    if not is_non_random or p_value >= 0.05:
        return {
            "transition": list(transition),
            "evidence_tier": [],
            "semantic_claim": "no_structural_claim",
        }
    return {
        "transition": list(transition),
        "evidence_tier": ["T2"],
        "semantic_claim": "structural_association_not_function",
        "ceiling": "T4 (causal function) requires Track C playback biology",
    }
