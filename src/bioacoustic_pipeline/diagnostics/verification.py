"""Caller-verification measurement instruments — EER + leakage-safe pairs.

Measurement instruments in the ``diagnostics.probes`` sense: they quantify how
well a representation separates callers, with the statistics the blueprint
demands (permutation nulls + cluster bootstrap CIs). They are NOT models, are
never firewall-registered, and never feed a verdict axis: a caller-supervised
representation cannot be evidence for caller-structure claims (circularity),
so every number produced here is a diagnostic, ceiling-bound well below meaning.

Anti-leakage mechanics (the reason this module exists rather than a plain
``sklearn.metrics`` call):

- Genuine pairs are always **same caller, different file** — two windows from
  one recording share channel and session, not just the caller, and would
  inflate EER optimism.
- Permutations shuffle labels **at file granularity**, preserving the cluster
  structure a naive row-level shuffle would destroy.
- Bootstrap resamples **files** (the unit of independence), rebuilding pairs
  inside each replicate.

EER is the ROC equal-error crossing with linear interpolation between grid
points, not an ``argmin`` on the empirical grid: the grid has resolution 1/N,
so argmin is biased by up to 1/(2N) and returns an arbitrary threshold.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bioacoustic_pipeline.diagnostics.bootstrap import BootstrapResult
from bioacoustic_pipeline.diagnostics.null_models import NullTestResult


@dataclass
class EERResult:
    """Equal-error-rate point on the ROC of a verification score."""

    eer: float
    threshold: float
    n_pos: int
    n_neg: int

    def summary(self) -> str:
        return f"EER={self.eer:.4f} @thr={self.threshold:.4f} (n={self.n_pos}/{self.n_neg})"


@dataclass
class PairSample:
    """A balanced genuine/impostor pair sample with scores.

    ``labels``: 1 = genuine (same caller, different file), 0 = impostor.
    ``pair_i``/``pair_j``: row indices into the embedding matrix, kept so
    callers can audit pair construction (the unit tests do).
    """

    labels: np.ndarray
    scores: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray

    @property
    def n_genuine(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def n_impostor(self) -> int:
        return int((self.labels == 0).sum())


def calculate_eer(pos_scores, neg_scores) -> EERResult:
    """Equal-error rate from genuine (``pos_scores``) and impostor scores.

    Sorts the pooled scores descending, sweeps the threshold across the grid,
    and interpolates the segment where FNR crosses FPR. Perfect separation
    returns 0.0; a score that never crosses (or identical distributions)
    returns 0.5.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("calculate_eer needs non-empty pos and neg score sets")

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]

    fpr = np.cumsum(labels == 0) / neg.size
    tpr = np.cumsum(labels == 1) / pos.size
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    fnr = 1.0 - tpr
    # diff = fnr - fpr is non-increasing from grid point 1 (fpr only rises,
    # fnr only falls as the threshold drops), so there is one crossing segment
    # to interpolate — no argmin-on-grid bias, no ambiguity between perfect
    # separation and an inverted score.
    diff = fnr - fpr
    if diff[1] <= 0:
        # first sorted score already belongs to the impostor side: degenerate
        # (inverted) score — fall back to the grid min of max(far, frr)
        vals = np.maximum(fpr, fnr)
        k = int(np.argmin(vals))
        return EERResult(float(vals[k]), float(scores[max(k - 1, 0)]), pos.size, neg.size)
    k = int(np.argmax(diff <= 0))  # first grid point at/below the crossing
    denom = diff[k - 1] - diff[k]
    if denom <= 1e-12:
        return EERResult(float(fpr[k - 1]), float(scores[k - 1]), pos.size, neg.size)
    s = diff[k - 1] / denom
    eer = fpr[k - 1] + s * (fpr[k] - fpr[k - 1])
    return EERResult(float(eer), float(scores[k - 1]), pos.size, neg.size)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def build_genuine_impostor_pairs(
    embeddings,
    caller_ids,
    file_ids,
    *,
    n_pairs: int = 2000,
    max_per_caller: int | None = None,
    seed: int = 0,
    group_ids=None,
    impostor_scope: str = "any",
) -> PairSample:
    """LFW-style balanced genuine/impostor pair sample with cosine scores.

    Genuine = same caller, **different file**; impostor = different callers.
    ``max_per_caller`` caps genuine pairs per caller (default: ceil of an even
    spread over eligible callers) so abundant callers cannot dominate.

    §46 twin confound: with ``group_ids`` (e.g. twin family per row) and
    ``impostor_scope`` — ``"within_group"`` restricts impostors to same-group
    caller pairs (the genetically/environmentally hard condition),
    ``"cross_group"`` to different groups (the easy condition), ``"any"``
    ignores groups (historical behaviour). Comparing EER across scopes
    measures the biological limit: how much caller-ID signal survives when
    the impostor is your twin.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    if emb.ndim != 2 or len(emb) != len(callers) or len(emb) != len(files):
        raise ValueError("embeddings/caller_ids/file_ids length mismatch")
    if impostor_scope not in ("any", "within_group", "cross_group"):
        raise ValueError("impostor_scope must be any/within_group/cross_group")
    groups = None
    if group_ids is not None and impostor_scope != "any":
        groups = np.asarray(group_ids)
        if len(groups) != len(emb):
            raise ValueError("group_ids length mismatch")

    files_by_caller: dict = {}
    for caller in np.unique(callers):
        rows = np.where(callers == caller)[0]
        per_file = {f: rows[files[rows] == f] for f in np.unique(files[rows])}
        if len(per_file) >= 2:
            files_by_caller[caller] = per_file
    eligible = sorted(files_by_caller)
    if len(eligible) < 2:
        raise ValueError("need >=2 callers with >=2 files each to build pairs")

    n_half = n_pairs // 2
    cap = max_per_caller or max(1, math.ceil(n_half / len(eligible)))
    rng = np.random.RandomState(seed)

    labels, pair_i, pair_j = [], [], []

    def _row(caller, exclude_file=None):
        pool = [f for f in files_by_caller[caller] if f != exclude_file]
        f = pool[rng.randint(len(pool))]
        rows = files_by_caller[caller][f]
        return int(rows[rng.randint(len(rows))])

    used: dict = {c: 0 for c in eligible}
    attempts = 0
    while len(labels) < n_half and attempts < n_half * 50:
        attempts += 1
        c = eligible[rng.randint(len(eligible))]
        if used[c] >= cap:
            continue
        fa = list(files_by_caller[c])[rng.randint(len(files_by_caller[c]))]
        rows_a = files_by_caller[c][fa]
        i = int(rows_a[rng.randint(len(rows_a))])
        j = _row(c, exclude_file=fa)
        if j == i:
            continue
        labels.append(1)
        pair_i.append(i)
        pair_j.append(j)
        used[c] += 1
    n_genuine = len(labels)
    if n_genuine == 0:
        raise ValueError("per-caller caps admitted no genuine pairs; raise max_per_caller")

    attempts = 0
    while len(labels) < 2 * n_genuine and attempts < n_genuine * 50:
        attempts += 1
        c1, c2 = eligible[rng.randint(len(eligible))], eligible[rng.randint(len(eligible))]
        if c1 == c2:
            continue
        if groups is not None:
            g1 = groups[callers == c1][0]
            g2 = groups[callers == c2][0]
            same_group = bool(g1 == g2)
            if impostor_scope == "within_group" and not same_group:
                continue
            if impostor_scope == "cross_group" and same_group:
                continue
        i, j = _row(c1), _row(c2)
        labels.append(0)
        pair_i.append(i)
        pair_j.append(j)

    labels_a = np.asarray(labels)
    ia, ja = np.asarray(pair_i), np.asarray(pair_j)
    scores = _cosine(emb[ia], emb[ja])
    return PairSample(labels=labels_a, scores=scores, pair_i=ia, pair_j=ja)


def permute_labels_at_file_level(caller_ids, file_ids, rng: np.random.RandomState):
    """Shuffle caller labels whole-file-at-a-time (cluster-aware permutation).

    Returns a new label array where every row of a file carries the label of
    some (random) file — the file->label map is permuted, so the cluster
    structure that a row-level shuffle would quietly break stays intact.
    """
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    unique_files = np.unique(files)
    file_labels = np.array([callers[files == f][0] for f in unique_files])
    shuffled = file_labels.copy()
    rng.shuffle(shuffled)
    remap = {f: shuffled[k] for k, f in enumerate(unique_files)}
    return np.array([remap[f] for f in files])


def label_permutation_null(
    stat_fn,
    embeddings,
    caller_ids,
    file_ids,
    *,
    n_permutations: int = 200,
    seed: int = 0,
    direction: str = "lower",
    alpha: float = 0.05,
) -> NullTestResult:
    """File-level permutation null for a verification statistic.

    ``stat_fn(embeddings, caller_ids, file_ids) -> float`` is re-evaluated under
    permuted file->caller maps. ``direction`` says which tail is evidence:
    "lower" for error rates (EER), "greater" for accuracies. Phipson–Smyth
    corrected p-value, matching ``null_models`` convention.
    """
    if direction not in ("lower", "greater"):
        raise ValueError("direction must be 'lower' or 'greater'")
    emb = np.asarray(embeddings)
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    observed = float(stat_fn(emb, callers, files))
    rng = np.random.RandomState(seed)
    nulls = []
    for b in range(n_permutations):
        perm = permute_labels_at_file_level(callers, files, rng)
        try:
            nulls.append(float(stat_fn(emb, perm, files)))
        except ValueError:
            continue  # degenerate replicate (e.g. <2 pairable callers): undefined
    if len(nulls) < max(2, n_permutations // 2):
        raise ValueError(
            f"too few valid permutation replicates ({len(nulls)}/{n_permutations}) — "
            "eval set too thin for this statistic"
        )
    nulls_a = np.asarray(nulls)
    if direction == "lower":
        count = int((nulls_a <= observed).sum())
    else:
        count = int((nulls_a >= observed).sum())
    std = float(nulls_a.std()) or 1e-12
    return NullTestResult(
        observed=observed,
        null_mean=float(nulls_a.mean()),
        null_std=float(nulls_a.std()),
        p_value=(1.0 + count) / (len(nulls) + 1.0),
        z_score=(observed - float(nulls_a.mean())) / std,
        n_permutations=len(nulls),
        null_type=f"file-label-permutation ({direction})",
        alpha=alpha,
    )


def cluster_bootstrap_files(
    stat_fn,
    embeddings,
    caller_ids,
    file_ids,
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Cluster (file-level) bootstrap CI for a verification statistic.

    Resamples files with replacement, rebuilds the (embeddings, labels) blocks
    per replicate — the pair builder's same-file exclusion treats repeated
    draws of one file correctly, so replicate statistics stay honest.
    """
    emb = np.asarray(embeddings)
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    unique_files = np.unique(files)
    observed = float(stat_fn(emb, callers, files))
    rng = np.random.RandomState(seed)
    stats = []
    for b in range(n_bootstrap):
        pick = unique_files[rng.randint(0, len(unique_files), len(unique_files))]
        rows = np.concatenate([np.where(files == f)[0] for f in pick])
        try:
            stats.append(float(stat_fn(emb[rows], callers[rows], files[rows])))
        except ValueError:
            continue  # degenerate resample: statistic undefined there
    if len(stats) < max(2, n_bootstrap // 2):
        raise ValueError(
            f"too few valid bootstrap replicates ({len(stats)}/{n_bootstrap}) — "
            "eval set too thin for this statistic"
        )
    stats_a = np.asarray(stats)
    a = 1.0 - confidence
    lo, hi = np.percentile(stats_a, [100 * a / 2, 100 * (1 - a / 2)])
    return BootstrapResult(
        observed=observed,
        mean=float(stats_a.mean()),
        std=float(stats_a.std()),
        ci_lo=float(lo),
        ci_hi=float(hi),
        confidence=confidence,
        n_bootstrap=len(stats),
    )


@dataclass
class PairedDeltaResult:
    """Paired bootstrap CI on the DIFFERENCE of two arms' statistics.

    Same files, same resample indices, same pair seeds inside every
    replicate — the arms are compared on identical evidence, which is the
    honest stop-rule statistic (a CI on Δ, not two CIs to eyeball).
    """

    observed: float
    ci_lo: float
    ci_hi: float
    n_bootstrap: int
    confidence: float

    def summary(self) -> str:
        return (
            f"delta={self.observed:.4f} [{self.ci_lo:.4f}, {self.ci_hi:.4f}] "
            f"{int(self.confidence * 100)}% CI (B={self.n_bootstrap})"
        )


def paired_bootstrap_delta(
    stat_fn_a,
    stat_fn_b,
    embeddings_a,
    embeddings_b,
    caller_ids,
    file_ids,
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedDeltaResult:
    """Cluster bootstrap on ``stat_fn_b - stat_fn_a`` with SHARED resamples.

    Both arms must be evaluated on the same rows (same callers/files order);
    each replicate resamples files once and hands the identical subset (and
    pair-building seed) to both statistics, so the only variation inside a
    replicate is the representation.
    """
    ea = np.asarray(embeddings_a)
    eb = np.asarray(embeddings_b)
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    if len(ea) != len(eb) or len(ea) != len(callers):
        raise ValueError("paired arms must align row-for-row")
    observed = float(stat_fn_b(eb, callers, files)) - float(stat_fn_a(ea, callers, files))
    unique_files = np.unique(files)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_bootstrap):
        pick = unique_files[rng.randint(0, len(unique_files), len(unique_files))]
        rows = np.concatenate([np.where(files == f)[0] for f in pick])
        try:
            da = float(stat_fn_a(ea[rows], callers[rows], files[rows]))
            db = float(stat_fn_b(eb[rows], callers[rows], files[rows]))
        except ValueError:
            continue  # degenerate resample: undefined for BOTH arms alike
        deltas.append(db - da)
    if len(deltas) < max(2, n_bootstrap // 2):
        raise ValueError(f"too few valid paired replicates ({len(deltas)}/{n_bootstrap})")
    arr = np.asarray(deltas)
    a = 1.0 - confidence
    lo, hi = np.percentile(arr, [100 * a / 2, 100 * (1 - a / 2)])
    return PairedDeltaResult(
        observed=observed,
        ci_lo=float(lo),
        ci_hi=float(hi),
        n_bootstrap=len(deltas),
        confidence=confidence,
    )
