"""Opportunity-aware concentration — §46 Stage 3 (unit-presence endpoint).

For each codebook entry: is its BOUT-LEVEL PRESENCE concentrated among
callers beyond sampling opportunity? The null draws per-caller presence
counts from Binomial(E_i, pi) — exposure preserved exactly — so a caller
with 3 bouts cannot "concentrate" the way a caller with 300 can.
Concentration statistic: Herfindahl of per-caller presence RATES.

Saturated entries (present in nearly every bout) carry no context/caller
information by construction: they are flagged "uninformative", never read
as "shared". Failure to reject concentration is NOT positive evidence of
sharing — that is Stage 4's within-caller design.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SATURATION_THRESHOLD = 0.95


@dataclass
class ConcentrationResult:
    """Per-entry caller-concentration vs the exposure-preserving null."""

    observed: float
    null_mean: float
    null_std: float
    p_value: float
    pi: float  # pooled presence rate
    exposures: np.ndarray
    counts: np.ndarray
    saturated: bool
    interpretation: str

    def as_dict(self) -> dict:
        return {
            "observed_herfindahl": round(self.observed, 5),
            "null_mean": round(self.null_mean, 5),
            "null_std": round(self.null_std, 5),
            "p_value": self.p_value,
            "pooled_presence_rate": round(self.pi, 5),
            "saturated": self.saturated,
            "interpretation": self.interpretation,
        }


def bout_presence_matrix(labels: np.ndarray, offsets: np.ndarray, *, n_clusters: int):
    """Binary per-bout presence over codebook entries from frame labels."""
    n_bouts = len(offsets) - 1
    # int64: per-bout presence counts later sum over hundreds of bouts
    # (int8 wraps past 127 — a real overflow seen at exposure 210)
    P = np.zeros((n_bouts, n_clusters), dtype=np.int64)
    for b in range(n_bouts):
        seg = labels[offsets[b] : offsets[b + 1]]
        P[b, seg] = 1
    return P


def herfindahl_of_rates(counts: np.ndarray, exposures: np.ndarray) -> float:
    """Herfindahl of per-caller presence rates (weights = rate shares)."""
    rates = np.asarray(counts, dtype=np.float64) / np.maximum(
        np.asarray(exposures, dtype=np.float64), 1e-12
    )
    total = rates.sum()
    if total <= 0:
        return 0.0
    w = rates / total
    return float((w**2).sum())


def excess_concentration(
    counts: np.ndarray,
    exposures: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 0,
    saturation: float = SATURATION_THRESHOLD,
) -> ConcentrationResult:
    """Caller-concentration of one entry's presence vs opportunity null.

    ``counts[i]`` = bouts of caller i containing the entry;
    ``exposures[i]`` = bouts of caller i examined. Null: Binomial(E_i, pi)
    with pi = sum(counts)/sum(exposures), exposure structure preserved.
    """
    counts = np.asarray(counts, dtype=np.int64)
    exposures = np.asarray(exposures, dtype=np.int64)
    pi = counts.sum() / max(exposures.sum(), 1)
    observed = herfindahl_of_rates(counts, exposures)
    rng = np.random.RandomState(seed)
    nulls = np.empty(n_permutations, dtype=np.float64)
    for b in range(n_permutations):
        sim = rng.binomial(exposures, pi)
        nulls[b] = herfindahl_of_rates(sim, exposures)
    exceed = int((nulls >= observed).sum())
    saturated = bool(pi >= saturation)
    interpretation = (
        "saturated (near-universal presence): uninformative for caller/context "
        "specificity — not evidence of sharing"
        if saturated
        else (
            "concentration beyond opportunity"
            if exceed == 0
            else "not concentrated beyond opportunity (failure to reject is NOT evidence of sharing)"
        )
    )
    return ConcentrationResult(
        observed=observed,
        null_mean=float(nulls.mean()),
        null_std=float(nulls.std()),
        p_value=(1.0 + exceed) / (n_permutations + 1.0),
        pi=float(pi),
        exposures=exposures,
        counts=counts,
        saturated=saturated,
        interpretation=interpretation,
    )
