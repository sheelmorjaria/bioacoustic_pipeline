"""Bootstrap resampling for diagnostic confidence intervals + edge stability.

Every diagnostic point estimate should carry uncertainty. This resamples the
unit of independence (the recording/file/sequence) with replacement, recomputes
the statistic, and reports a percentile confidence interval. For the Functional
Alias Detector it additionally tracks per-edge survival: the fraction of
bootstrap replicates in which a given alias pair re-appears.

Pure numpy. The unit of resampling is the sequence (file), never the token —
tokens within a file are not independent.

WHEN THE FILE-BOOTSTRAP IS (AND ISN'T) VALID
--------------------------------------------
The nonparametric file-bootstrap gives honest CIs for statistics that DECOMPOSE
additively across files (mean per-file LZ complexity, mean per-file vocabulary,
mean segment duration). For statistics computed on POOLED co-occurrence across
the whole corpus — e.g. alias density, which counts cosine->0.80 pairs in the
pooled context matrix — the file-bootstrap is BIASED DOWNWARD when files are
short: each resample is a noisier co-occurrence draw than the full pool, so
fewer pairs cross the threshold. (On LibriSpeech word-level segments, median
file length is ~4 tokens: observed alias density 7.3% vs bootstrap mean 1.4%.)

For significance of a pooled/thresholded statistic, use the permutation test in
``diagnostics.null_models`` (same data, structure-destroying null) — it is the
valid test. Use ``bootstrap_statistic`` for per-file-decomposable statistics,
and ``alias_edge_stability`` for the ROBUSTNESS of specific alias edges (a
relative recurrence ranking, not an absolute CI).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

StatisticFn = Callable[[list[list[int]]], float]


@dataclass
class BootstrapResult:
    """Percentile CI for a scalar diagnostic statistic."""

    observed: float
    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    confidence: float
    n_bootstrap: int

    def summary(self) -> str:
        return (
            f"{self.observed:.4f} "
            f"[{self.ci_lo:.4f}, {self.ci_hi:.4f}] "
            f"{int(self.confidence * 100)}% CI (B={self.n_bootstrap})"
        )


def bootstrap_statistic(
    sequences: list[list[int]],
    statistic_fn: StatisticFn,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Resample sequences with replacement -> percentile CI on statistic_fn."""
    if len(sequences) == 0:
        raise ValueError("cannot bootstrap an empty sequence list")
    rng = np.random.RandomState(seed)
    n = len(sequences)
    observed = float(statistic_fn(sequences))
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        sample = [sequences[i] for i in idx]
        stats[b] = float(statistic_fn(sample))
    alpha = 1.0 - confidence
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        observed=observed,
        mean=float(stats.mean()),
        std=float(stats.std()),
        ci_lo=float(lo),
        ci_hi=float(hi),
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )


@dataclass
class EdgeStability:
    """Per-edge survival across bootstrap replicates of the alias detector."""

    survival: dict[tuple[int, int], float] = field(default_factory=dict)
    n_bootstrap: int = 0
    min_survival: float = 0.5

    def stable_edges(self) -> list[tuple[int, int]]:
        """Edges that survive in >= min_survival fraction of replicates."""
        return sorted(
            [e for e, f in self.survival.items() if f >= self.min_survival]
        )

    def summary(self) -> str:
        n_stable = len(self.stable_edges())
        return (
            f"{n_stable} stable edges "
            f"(survival >= {self.min_survival:.0%} over {self.n_bootstrap} bootstraps)"
        )


def alias_edge_stability(
    sequences: list[list[int]],
    detect_fn: Callable[[list[list[int]]], "object"],
    n_bootstrap: int = 200,
    min_survival: float = 0.5,
    seed: int = 0,
) -> EdgeStability:
    """Bootstrap edge stability for the Functional Alias Detector.

    Args:
        sequences: per-file token streams.
        detect_fn: callable returning a FunctionalAliasReport (or any object
            with .groups, each group having .member_tokens).
        min_survival: fraction of replicates an edge must survive to count.
    """
    rng = np.random.RandomState(seed)
    n = len(sequences)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        sample = [sequences[i] for i in idx]
        rep = detect_fn(sample)
        for g in getattr(rep, "groups", []):
            members = sorted(g.member_tokens)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    counts[(members[i], members[j])] += 1
    survival = {edge: c / n_bootstrap for edge, c in counts.items()}
    return EdgeStability(survival=survival, n_bootstrap=n_bootstrap, min_survival=min_survival)
