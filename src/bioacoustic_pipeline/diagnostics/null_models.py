"""Null-model permutation tests for diagnostics.

Every diagnostic produces a point estimate. Without a null model, "5 alias
groups" or "LZ = 4.5" is unfalsifiable: is that more than chance? This module
provides generic permutation tests that destroy sequential structure while
preserving chosen marginals, then asks whether the observed statistic exceeds
the null distribution.

Shuffle strategies (increasing destruction of structure):
  within_file  — permute tokens inside each sequence; preserves per-sequence
                 unigram counts + sequence lengths; destroys local bigrams.
  cross_file   — pool all tokens, reassign preserving sequence lengths;
                 preserves global unigram + lengths; destroys file structure
                 too.
  unigram      — alias of cross_file (preserves only global unigram frequency).

The statistic under test is supplied by the caller (e.g. alias density, LZ
complexity, group count), making this reusable across the whole diagnostic
suite. Pure numpy — no torch.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ShuffleFn = Callable[[list[list[int]], np.random.RandomState], list[list[int]]]
StatisticFn = Callable[[list[list[int]]], float]


@dataclass
class NullTestResult:
    """Outcome of a one-sided permutation test (observed GREATER than null)."""

    observed: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    n_permutations: int
    null_type: str
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        flag = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"observed={self.observed:.4f} null={self.null_mean:.4f}"
            f"±{self.null_std:.4f} z={self.z_score:.2f} "
            f"p={self.p_value:.4f} [{flag}, {self.null_type}]"
        )


def within_file_shuffle(
    sequences: list[list[int]], rng: np.random.RandomState
) -> list[list[int]]:
    """Permute tokens within each sequence (preserves per-seq unigrams)."""
    out = []
    for seq in sequences:
        if len(seq) > 1:
            perm = rng.permutation(len(seq))
            out.append([int(seq[i]) for i in perm])
        else:
            out.append(list(seq))
    return out


def cross_file_shuffle(
    sequences: list[list[int]], rng: np.random.RandomState
) -> list[list[int]]:
    """Pool all tokens, reassign to sequences preserving lengths.

    Preserves the global unigram distribution and every sequence length, but
    destroys file membership and all sequential structure.
    """
    lengths = [len(s) for s in sequences]
    pool = [int(t) for s in sequences for t in s]
    rng.shuffle(pool)
    out: list[list[int]] = []
    i = 0
    for n in lengths:
        out.append(pool[i : i + n])
        i += n
    return out


# unigram-preserving shuffle is the same operation as cross_file (preserves
# only the global unigram frequency + sequence lengths).
unigram_shuffle = cross_file_shuffle

SHUFFLE_STRATEGIES = {
    "within_file": within_file_shuffle,
    "cross_file": cross_file_shuffle,
    "unigram": unigram_shuffle,
}


def permutation_test(
    sequences: list[list[int]],
    statistic_fn: StatisticFn,
    shuffle_fn: ShuffleFn = within_file_shuffle,
    n_permutations: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    null_type: str | None = None,
) -> NullTestResult:
    """One-sided permutation test: is the observed statistic greater than null?

    Args:
        sequences: per-sequence token lists (e.g. per-file streams).
        statistic_fn: maps a sequence list -> a scalar statistic.
        shuffle_fn: structure-destroying shuffle preserving chosen marginals.
        n_permutations: number of null replicates.
        alpha: significance level.
        seed: RNG seed for reproducibility.
        null_type: label for the result (defaults to the shuffle name).
    """
    rng = np.random.RandomState(seed)
    observed = float(statistic_fn(sequences))
    null_samples = np.empty(n_permutations, dtype=np.float64)
    ge = 0
    for b in range(n_permutations):
        shuffled = shuffle_fn(sequences, rng)
        val = float(statistic_fn(shuffled))
        null_samples[b] = val
        if val >= observed:
            ge += 1
    null_mean = float(null_samples.mean())
    null_std = float(null_samples.std())
    # p-value with the +1/+1 correction (Phipson & Smyth 2010) so p>0 is always
    # attainable and a perfect separation gives p = 1/(B+1), not 0.
    p_value = (1 + ge) / (n_permutations + 1)
    z_score = (observed - null_mean) / (null_std + 1e-12)
    label = null_type or getattr(shuffle_fn, "__name__", "shuffle")
    return NullTestResult(
        observed=observed,
        null_mean=null_mean,
        null_std=null_std,
        p_value=p_value,
        z_score=z_score,
        n_permutations=n_permutations,
        null_type=label,
        alpha=alpha,
    )
