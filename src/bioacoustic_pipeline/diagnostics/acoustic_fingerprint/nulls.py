"""Fingerprint-aware null tests (token-coherence + alias-divergence).

One-sided permutation tests: does the *observed* fingerprint coherence (or alias
divergence) exceed what you get by randomly reassigning labels? Reuses
``diagnostics.null_models.NullTestResult`` and its Phipson-Smyth ``+1/+1`` p-value so
the result is uniform with the rest of the diagnostic suite. (The permutation is
implemented inline because the fingerprint statistic is computed over token/alias maps,
not a token stream — but the convention is shared.)

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from bioacoustic_pipeline.diagnostics.null_models import NullTestResult

from . import coherence, similarity


def _mean_cohesion(token_to_vecs: dict) -> float:
    flat = [v for vecs in token_to_vecs.values() for v in vecs]
    if len(flat) < 2:
        return 0.0
    norm = similarity.normalize_batch(np.asarray(flat, dtype=np.float64))
    # rebuild normalized groups by walking the same order
    i = 0
    coh = []
    for vecs in token_to_vecs.values():
        n = len(vecs)
        if n >= 2:
            coh.append(coherence.token_fingerprint_coherence({0: norm[i : i + n]})[0]["cohesion"])
        i += n
    return float(np.mean(coh)) if coh else 0.0


def token_coherence_null(token_to_fingerprints: dict, *, n_permutations: int = 200,
                         alpha: float = 0.05, seed: int = 0) -> NullTestResult:
    """Shuffle token labels across segments; is real token cohesion > chance?"""
    labels, flat = [], []
    for tok, vecs in token_to_fingerprints.items():
        for v in vecs:
            labels.append(tok)
            flat.append(np.asarray(v, dtype=np.float64))
    labels = np.array(labels)
    flat = np.asarray(flat, dtype=np.float64)
    if len(flat) < 2:
        return NullTestResult(0.0, 0.0, 0.0, 1.0, 0.0, n_permutations, "token_coherence", alpha)

    def cohesion_of(lbls):
        groups = {}
        for lbl, vec in zip(lbls, flat):
            groups.setdefault(lbl, []).append(vec)
        return _mean_cohesion(groups)

    rng = np.random.RandomState(seed)
    observed = cohesion_of(labels)
    ge = 0
    samples = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(labels)
        val = cohesion_of(perm)
        samples[b] = val
        if val >= observed:
            ge += 1
    p = (1 + ge) / (n_permutations + 1)
    return NullTestResult(observed, float(samples.mean()), float(samples.std()), p,
                          (observed - samples.mean()) / (samples.std() + 1e-12),
                          n_permutations, "token_coherence", alpha)


def _mean_divergence(alias_groups: dict, token_centroids: dict) -> float:
    res = coherence.alias_acoustic_divergence(alias_groups, token_centroids)
    vals = [g["divergence"] for g in res.values() if g["n"] >= 2]
    return float(np.mean(vals)) if vals else 0.0


def alias_divergence_null(alias_groups: dict, token_centroids: dict, *,
                          n_permutations: int = 200, alpha: float = 0.05,
                          seed: int = 0) -> NullTestResult:
    """Shuffle token->centroid assignment (preserve token set); is real alias
    divergence > chance?"""
    tokens = list(token_centroids.keys())
    if len(tokens) < 2 or not alias_groups:
        return NullTestResult(0.0, 0.0, 0.0, 1.0, 0.0, n_permutations, "alias_divergence", alpha)
    rng = np.random.RandomState(seed)
    observed = _mean_divergence(alias_groups, token_centroids)
    ge = 0
    samples = np.empty(n_permutations)
    for b in range(n_permutations):
        shuffled = rng.permutation(tokens)
        remap = {t: token_centroids[s] for t, s in zip(tokens, shuffled)}
        val = _mean_divergence(alias_groups, remap)
        samples[b] = val
        if val >= observed:
            ge += 1
    p = (1 + ge) / (n_permutations + 1)
    return NullTestResult(observed, float(samples.mean()), float(samples.std()), p,
                          (observed - samples.mean()) / (samples.std() + 1e-12),
                          n_permutations, "alias_divergence", alpha)


def _mean_pairwise_distance(rows: np.ndarray) -> float:
    """Mean cosine distance (1 - cosine) over all pairs in rows (normalized)."""
    if rows.ndim != 2 or rows.shape[0] < 2:
        return 0.0
    n = rows.shape[0]
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - similarity.cosine(rows[i], rows[j])
            cnt += 1
    return total / max(cnt, 1)


def species_distance_null(group_a, group_b, *, n_permutations: int = 200,
                          alpha: float = 0.05, seed: int = 0) -> NullTestResult:
    """Within- vs cross-species fingerprint separation.

    Statistic = (mean cross-species distance) - (mean within-species distance).
    Observed > null => the two groups are acoustically more separated than a random
    label split — i.e. the claimed similarity/difference is species-specific, not
    generic. Permutation shuffles species labels (preserving group sizes).
    """
    A = np.asarray(group_a, dtype=np.float64)
    B = np.asarray(group_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] < 2 or B.shape[0] < 2:
        return NullTestResult(0.0, 0.0, 0.0, 1.0, 0.0, n_permutations, "species_distance", alpha)
    allv = np.vstack([A, B])
    labels = np.array([0] * A.shape[0] + [1] * B.shape[0])
    norm = similarity.normalize_batch(allv)

    def separation(lbls):
        a = norm[lbls == 0]
        b = norm[lbls == 1]
        if a.shape[0] < 2 or b.shape[0] < 2:
            return 0.0
        cross = float(np.mean([1.0 - similarity.cosine(x, y) for x in a for y in b]))
        within = 0.5 * (_mean_pairwise_distance(a) + _mean_pairwise_distance(b))
        return cross - within

    rng = np.random.RandomState(seed)
    observed = separation(labels)
    ge = 0
    samples = np.empty(n_permutations)
    for b in range(n_permutations):
        val = separation(rng.permutation(labels))
        samples[b] = val
        if val >= observed:
            ge += 1
    p = (1 + ge) / (n_permutations + 1)
    return NullTestResult(observed, float(samples.mean()), float(samples.std()), p,
                          (observed - samples.mean()) / (samples.std() + 1e-12),
                          n_permutations, "species_distance", alpha)
