"""Cross-stream diagnostics — the relationship between VQ syntax and β-VAE affect.

These metrics test whether the two streams carry independent information or are
redundant. The key question: does the discrete token (what categorical unit
occurred) predict the continuous affect (how did graded state vary), and vice versa?

Pure numpy/stdlib. No torch.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def affect_within_token_variance(segments: list) -> dict[int, dict]:
    """For each VQ token: variance of the affect vector across all its instances.

    High variance = "same categorical call form, different graded state" — the
    token is acoustically consistent but affectively flexible (biologically
    interesting: the same call type carries different arousal/prosody).

    Returns {token_id: {n, affect_var_mean, affect_intensity_var, affect_centroid}}.
    """
    token_affects: dict[int, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        tok = getattr(seg, "vq_token", -1)
        mu = getattr(seg, "affect_mu", None)
        if tok != -1 and mu is not None and len(mu) > 0:
            token_affects[tok].append(mu)

    result: dict[int, dict] = {}
    for tok, mus in token_affects.items():
        mus_arr = np.array(mus)
        centroid = mus_arr.mean(axis=0)
        # per-dimension variance, averaged
        var_per_dim = mus_arr.var(axis=0)
        result[tok] = {
            "n": len(mus),
            "affect_var_mean": round(float(var_per_dim.mean()), 6),
            "affect_intensity_var": round(float(np.var([np.linalg.norm(m) for m in mus])), 6),
            "affect_centroid_norm": round(float(np.linalg.norm(centroid)), 4),
        }
    return result


def token_affect_mutual_information(segments: list) -> float:
    """MI between VQ token and discretized affect cluster.

    Discretizes affect into K clusters (k-means or norm-quantile), then computes
    MI(token, affect_cluster). High MI = tokens predict affect; low MI = affect
    carries information beyond the token (the streams are complementary).

    Returns normalized MI in [0, 1].
    """
    tokens = []
    affects = []
    for seg in segments:
        tok = getattr(seg, "vq_token", -1)
        mu = getattr(seg, "affect_mu", None)
        if tok != -1 and mu is not None and len(mu) > 0:
            tokens.append(tok)
            affects.append(mu)
    if len(tokens) < 20:
        return 0.0

    tokens = np.array(tokens)
    A = np.array(affects)

    # discretize affect into K = min(8, sqrt(n)) norm-based clusters
    norms = np.linalg.norm(A, axis=1)
    k = min(8, max(2, int(np.sqrt(len(norms)))))
    clusters = np.digitize(norms, np.linspace(norms.min(), norms.max() + 1e-9, k)[:-1])

    # MI(token, affect_cluster)
    from bioacoustic_pipeline.diagnostics.context_binding import mutual_information  # noqa: PLC0415

    # build contingency table
    table = np.zeros((len(set(tokens.tolist())), k), dtype=np.float64)
    tok_idx = {t: i for i, t in enumerate(sorted(set(tokens.tolist())))}
    for t, c in zip(tokens, clusters):
        table[tok_idx[int(t)], int(c)] += 1.0
    mi, nmi = mutual_information(table)
    return nmi


def alias_affect_equivalence(
    alias_groups: list[set[int]],
    segments: list,
) -> list[dict]:
    """For each functional alias group: are the member tokens affectively equivalent?

    Cases:
      same function + different sound + same affect → strong functional alias
      same function + different sound + different affect → syntactic alias with affective distinction
    """
    token_affects: dict[int, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        tok = getattr(seg, "vq_token", -1)
        mu = getattr(seg, "affect_mu", None)
        if tok != -1 and mu is not None and len(mu) > 0:
            token_affects[tok].append(mu)

    results = []
    for gi, group in enumerate(alias_groups):
        centroids = {}
        for tok in group:
            mus = token_affects.get(tok, [])
            if mus:
                centroids[tok] = np.mean(mus, axis=0)

        if len(centroids) < 2:
            results.append({"group": gi, "members": sorted(group), "affect_equivalence": None,
                            "note": "insufficient affect data"})
            continue

        # pairwise cosine between token centroids within the alias group
        toks = sorted(centroids.keys())
        cents = np.array([centroids[t] for t in toks])
        norms = np.linalg.norm(cents, axis=1, keepdims=True)
        unit = cents / (norms + 1e-9)
        cos_matrix = unit @ unit.T
        # mean off-diagonal cosine (1.0 = perfectly equivalent affect, 0.0 = orthogonal)
        off_diag = cos_matrix[np.triu_indices_from(cos_matrix, k=1)]
        mean_cos = float(off_diag.mean()) if len(off_diag) > 0 else 0.0

        results.append({
            "group": gi,
            "members": toks,
            "affect_equivalence": round(mean_cos, 4),
            "interpretation": (
                "affectively_equivalent" if mean_cos > 0.7
                else "affectively_distinct" if mean_cos < 0.3
                else "partial"
            ),
        })
    return results


def motif_affect_signature(
    motif_pattern: list[int],
    segments: list,
) -> dict:
    """Affect trajectory for a token motif: how does affect_norm change across the pattern?

    Example: motif [12, 44, 91] → affect_norm trajectory [0.2, 0.6, 1.1] = escalation.
    """
    token_norms: dict[int, list[float]] = defaultdict(list)
    for seg in segments:
        tok = getattr(seg, "vq_token", -1)
        intensity = getattr(seg, "affect_intensity", 0.0)
        if tok != -1:
            token_norms[tok].append(intensity)

    trajectory = []
    for tok in motif_pattern:
        norms = token_norms.get(tok, [])
        trajectory.append(float(np.mean(norms)) if norms else 0.0)

    trajectory_arr = np.array(trajectory)
    return {
        "pattern": motif_pattern,
        "mean_trajectory": trajectory,
        "affect_slope": round(float(np.polyfit(range(len(trajectory)), trajectory, 1)[0]), 4)
            if len(trajectory) >= 2 else 0.0,
        "affect_peak_position": int(np.argmax(trajectory)) if len(trajectory) > 0 else 0,
        "affect_range": round(float(trajectory_arr.max() - trajectory_arr.min()), 4)
            if len(trajectory) > 0 else 0.0,
    }
