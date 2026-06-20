"""Motif-aware null tests via permutation (reuses null_models conventions).

Tests whether observed motif support exceeds what you'd get by shuffling token
order (destroying sequential structure while preserving unigram frequencies).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations
import numpy as np
from bioacoustic_pipeline.diagnostics.null_models import NullTestResult
from .token_motifs import mine_ngrams


def _total_motif_support(sequences, min_n, max_n, min_support):
    counts = mine_ngrams(sequences, min_n, max_n, min_support)
    return sum(counts.values())


def motif_support_null(sequences, *, min_n=2, max_n=5, min_support=3,
                       n_permutations=200, alpha=0.05, seed=0):
    """Is total motif support > chance? Shuffles within each file (preserves unigrams)."""
    from bioacoustic_pipeline.diagnostics.null_models import within_file_shuffle
    rng = np.random.RandomState(seed)
    observed = _total_motif_support(sequences, min_n, max_n, min_support)
    ge = 0
    samples = np.empty(n_permutations)
    for b in range(n_permutations):
        shuffled = within_file_shuffle(sequences, rng)
        val = _total_motif_support(shuffled, min_n, max_n, min_support)
        samples[b] = val
        if val >= observed:
            ge += 1
    p = (1 + ge) / (n_permutations + 1)
    return NullTestResult(observed, float(samples.mean()), float(samples.std()), p,
                          (observed - samples.mean()) / (samples.std() + 1e-12),
                          n_permutations, "motif_support", alpha)
