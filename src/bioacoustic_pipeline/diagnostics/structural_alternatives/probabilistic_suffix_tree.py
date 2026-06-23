"""Probabilistic Suffix Tree (PST) — variable-order Markov syntax on token sequences.

Architecture v1.0 "alternative structure" technique. Where DUSTED finds *static* fuzzy matches, a
PST models the transitional-probability structure: which token follows which context, at variable
order. It keeps only contexts where the next-token distribution is significantly more predictive
than the shorter suffix — i.e. genuine long-range syntax rather than chance. This directly quantifies
the "grammar" of a vocalization; on Bengalese finch song (known Markovian syntax) it lets the pipeline
recover syntax rules blindly. Requires DISCRETE tokens (k-means cluster IDs from the fingerprint
trajectory), so it suits discrete repertoires and is weak on graded ones.

Pure stdlib (collections). Firewall-clean: operates on acoustic-similarity cluster IDs, not learned
semantics. NOT meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def _count_contexts(sequences: list[list[int]], max_depth: int):
    """Count every context (tuple) of length 1..max_depth+1 and its follower distribution.

    Returns: context_counts[context_tuple] -> total count; follow_counts[context_tuple][next] -> count.
    """
    context_counts: dict[tuple, int] = Counter()
    follow_counts: dict[tuple, Counter] = defaultdict(Counter)
    for seq in sequences:
        s = list(seq)
        for i in range(len(s)):
            for d in range(max_depth + 1):  # context length d uses tokens [i-d, i), follower = s[i]
                lo = i - d
                if lo < 0:
                    break
                ctx = tuple(s[lo:i])
                context_counts[ctx] += 1
                follow_counts[ctx][s[i]] += 1
    return context_counts, follow_counts


def fit_pst(
    sequences: list[list[int]],
    *,
    max_depth: int = 4,
    pmin: float = 0.1,
    alpha: float = 0.5,
    min_count: int = 5,
) -> dict:
    """Fit a Probabilistic Suffix Tree and return summary metrics.

    A context ``s`` is kept as a node if some token ``a`` has ``P(a|s) >= pmin`` AND
    ``P(a|s) >= (1+alpha) * P(a|suffix(s))`` (s is more predictive than its shorter suffix) AND the
    context appears at least ``min_count`` times. The number of kept nodes quantifies syntax richness.

    Returns: n_significant_nodes, n_contexts_total, branching (n_significant/n_total),
        max_depth_reached, entropy_rate (mean next-token entropy over kept nodes).
    """
    if not sequences:
        return {
            "n_significant_nodes": 0,
            "n_contexts_total": 0,
            "branching": 0.0,
            "max_depth_reached": 0,
            "entropy_rate": None,
        }
    context_counts, follow_counts = _count_contexts(sequences, max_depth)
    n_total = sum(1 for c, n in context_counts.items() if len(c) >= 1 and n >= min_count)
    significant = 0
    max_depth_reached = 0
    entropies: list[float] = []
    for ctx, n in context_counts.items():
        d = len(ctx)
        if d == 0 or n < min_count:
            continue
        dist = follow_counts[ctx]
        total = sum(dist.values())
        suffix = ctx[1:]
        suffix_total = context_counts.get(suffix, 0)
        is_sig = False
        h = 0.0
        for a, ca in dist.items():
            p = ca / total
            if p > 0:
                h -= p * np_log2(p)
            if p < pmin:
                continue
            p_suffix = (follow_counts[suffix][a] / suffix_total) if suffix_total else 0.0
            if p >= (1.0 + alpha) * p_suffix:
                is_sig = True
        if is_sig:
            significant += 1
            max_depth_reached = max(max_depth_reached, d)
            entropies.append(h)
    return {
        "n_significant_nodes": significant,
        "n_contexts_total": n_total,
        "branching": significant / n_total if n_total else 0.0,
        "max_depth_reached": max_depth_reached,
        "entropy_rate": float(sum(entropies) / len(entropies)) if entropies else None,
    }


def np_log2(x: float) -> float:
    import math  # noqa: PLC0415

    return math.log2(x)
