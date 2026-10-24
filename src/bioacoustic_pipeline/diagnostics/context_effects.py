"""Within-caller context effects — §46 Stage 4 (caller-stratified CMH).

Design: the Cochran–Mantel–Haenszel test stratified by CALLER on the per-
caller 2x2 (context x unit-presence) tables, with the Mantel–Haenszel
common odds ratio + Robins–Breslow–Greenland-style CI. Stratification
attributes within-caller context change to CONTEXT, not to caller baselines,
and is immune to the context-mixture confound (differential exposure a_i
pooling a shared context response into apparent caller heterogeneity) —
``pooled_vs_stratified`` quantifies exactly that gap.

Deliberately NOT a mixed-effects GLM: statsmodels' mixedlm is a linear mixed
model (wrong likelihood for binary data) and statsmodels is not a repo
dependency. The CMH is exact-minded, numpy/scipy-only, and states its own
limits: it assumes homogeneous OR across strata (checked via the stratum
count actually used) and cannot separate recording-group structure — that
residual confound is reported in every payload.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _tables_from_rows(rows, c1, c2):
    """Per-caller 2x2 tables [[present_c2, present_c1], [absent_c2, absent_c1]]."""
    by_caller: dict = {}
    for r in rows:
        if r["context"] not in (c1, c2):
            continue
        t = by_caller.setdefault(r["caller"], np.zeros((2, 2), dtype=np.int64))
        col = 0 if r["context"] == c2 else 1  # col 0 = C2 (a = present&C2)
        t[0 if r["present"] else 1, col] += 1
    # drop callers with no variation in either dimension (uninformative)
    keep = {}
    for caller, t in by_caller.items():
        if t.sum() == 0 or (t[0] + t[1]).min() == 0 or (t[:, 0] + t[:, 1]).min() == 0:
            continue
        keep[caller] = t
    return keep


def cmh_test(tables: list[np.ndarray]):
    """CMH chi-square(1), p, and MH log common odds ratio across strata.

    Tables are [[a, b], [c, d]] with a = present&C2, b = present&C1,
    c = absent&C2, d = absent&C1 per stratum.
    """
    a_sum = E_sum = var = R = S = 0.0
    for t in tables:
        a, b = float(t[0, 0]), float(t[0, 1])
        c, d = float(t[1, 0]), float(t[1, 1])
        n = a + b + c + d
        if n == 0:
            continue
        row1, col1 = a + b, a + c
        E_sum += row1 * col1 / n
        var += row1 * (n - row1) * col1 * (n - col1) / (n * n * (n - 1)) if n > 1 else 0.0
        R += a * d / n
        S += b * c / n
    stat = ((abs(a_sum + (0.0) - E_sum)) ** 2) / var if var > 0 else 0.0
    # a_sum accumulated inline below (kept simple): recompute once
    a_sum = sum(float(t[0, 0]) for t in tables)
    stat = ((abs(a_sum - E_sum) - 0.5) ** 2) / var if var > 0 else 0.0
    p = float(stats.chi2.sf(stat, 1)) if stat > 0 else 1.0
    or_ = R / S if S > 0 else float("inf")
    lor = math.log(or_) if or_ > 0 and np.isfinite(or_) else float("nan")
    return stat, p, lor


def _rbg_se(tables):
    """Robins–Breslow–Greenland SE of the MH log odds ratio."""
    R = S = P = Q = 0.0
    for t in tables:
        a, b = float(t[0, 0]), float(t[0, 1])
        c, d = float(t[1, 0]), float(t[1, 1])
        n = a + b + c + d
        if n == 0:
            continue
        R += a * d / n
        S += b * c / n
        P += (a + d) * a * d / (n * n)
        Q += (b + c) * b * c / (n * n)
    if S <= 0 or R <= 0:
        return float("nan")
    return math.sqrt(P / (2 * R * R) + Q / (2 * S * S) + (P + Q) / (2 * R * S))


def fit_hierarchical_context_model(rows, *, c1, c2) -> dict:
    """Within-caller context effect on unit presence (caller-stratified CMH)."""
    tables = _tables_from_rows(rows, c1, c2)
    if not tables:
        return {
            "log_odds_ratio": float("nan"),
            "ci": [float("nan"), float("nan")],
            "p_value": 1.0,
            "n_callers_used": 0,
            "interpretation": "no informative callers (no variation)",
        }
    stat, p, lor = cmh_test(list(tables.values()))
    se = _rbg_se(list(tables.values()))
    z = 1.959963984540054
    ci = (
        [lor - z * se, lor + z * se] if np.isfinite(se) and np.isfinite(lor) else [float("nan")] * 2
    )
    return {
        "log_odds_ratio": lor,
        "odds_ratio": float(np.exp(lor)) if np.isfinite(lor) else float("nan"),
        "ci": ci,
        "p_value": p,
        "chi2": stat,
        "n_callers_used": len(tables),
        "interpretation": (
            "within-caller context effect (C2 vs C1)"
            if p < 0.05 and ci[0] * ci[1] > 0
            else "no within-caller context effect detected (failure to detect is NOT evidence of absence)"
        ),
        "residual_confound": "recording-group structure not stratified; report alongside",
    }


def pooled_vs_stratified(rows, *, c1, c2) -> tuple[dict, dict]:
    """Pooled (confounded) vs caller-stratified effect — the mixture gap."""
    strat = fit_hierarchical_context_model(rows, c1=c1, c2=c2)
    pool_table = np.zeros((2, 2), dtype=np.int64)
    for r in rows:
        if r["context"] not in (c1, c2):
            continue
        pool_table[0 if r["present"] else 1, 0 if r["context"] == c1 else 1] += 1
    _, p_pool, lor_pool = cmh_test([pool_table])
    pooled = {"log_odds_ratio": lor_pool, "p_value": p_pool}
    strat["log_odds_ratio"] = strat["log_odds_ratio"]  # keep key names stable
    return pooled, strat
