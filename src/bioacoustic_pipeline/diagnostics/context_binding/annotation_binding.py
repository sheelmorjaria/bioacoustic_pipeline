"""Annotation-grounded context binding — token distribution ↔ behavioral label.

The bat corpus ships per-bout annotations (Emitter, Addressee, Context, pre/post
actions). These let us ask, WITHOUT needing within-bout temporal order or any
semantic decoding: is the per-bout token distribution conditioned on a behavioral
label? That is "functional-CONTEXT evidence" — a token class predictive of
annotated social context. It is explicitly NOT "meaning validated": a strong
association says a token is predictive of context, not that the bat uses it to
MEAN that context (that needs Track C playback).

Tests (per label type — Context / Emitter / Addressee):
  * mutual information + normalized MI (scale-free, in [0,1])
  * chi-square + Cramer's V
  * permutation p-value (shuffle label assignment across bouts, preserving the
    per-bout token multisets and the label marginal)
  * within-group null: shuffle the label ONLY within bouts sharing a group
    (e.g. Emitter), so a positive token↔Context association is not just a
    speaker-identity confound
  * Benjamini–Hochberg FDR across the label family

Pure numpy / stdlib — no torch, no model deps. Operates on per-bout token lists
+ a parallel label vector, so it is decoupled from segmentation / tokenization.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BindingResult:
    """One token↔label association test."""

    label_name: str
    n_bouts: int
    n_tokens: int
    n_label_values: int
    n_token_values: int
    observed_mi: float
    observed_nmi: float
    observed_chi2: float
    cramers_v: float
    # Permutation null on the chosen statistic (default normalized MI).
    statistic: str
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    n_permutations: int
    null_type: str  # "global" | "within_group"
    q_value: float | None = None  # BH-FDR across the label family (set later)
    extra: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "label_name": self.label_name,
            "n_bouts": self.n_bouts,
            "n_tokens": self.n_tokens,
            "n_label_values": self.n_label_values,
            "n_token_values": self.n_token_values,
            "observed_mi": round(self.observed_mi, 6),
            "observed_nmi": round(self.observed_nmi, 6),
            "observed_chi2": round(self.observed_chi2, 3),
            "cramers_v": round(self.cramers_v, 4),
            "statistic": self.statistic,
            "null_mean": round(self.null_mean, 6),
            "null_std": round(self.null_std, 6),
            "p_value": round(self.p_value, 4),
            "z_score": round(self.z_score, 3),
            "n_permutations": self.n_permutations,
            "null_type": self.null_type,
            "q_value": round(self.q_value, 4) if self.q_value is not None else None,
            "significant": self.significant,
            **self.extra,
        }


# --------------------------------------------------------------------------- #
# Contingency + statistics
# --------------------------------------------------------------------------- #


def build_contingency(token_lists, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V×C count table: cell[t,c] = # times token t occurs in bouts labeled c."""
    seqs = [list(s) for s in token_lists]
    labs = [lab for lab in labels]
    if len(seqs) != len(labs):
        raise ValueError("token_lists and labels must be aligned (same length)")
    # Vocabulary over tokens that actually appear, and over labels that appear.
    tok_vocab = sorted({int(t) for s in seqs for t in s})
    lab_vocab = sorted({lab for lab in labs})
    if not tok_vocab or len(lab_vocab) < 2:
        return (
            np.zeros((max(len(tok_vocab), 1), max(len(lab_vocab), 1)), dtype=np.float64),
            np.array(tok_vocab),
            np.array(lab_vocab),
        )
    tok_idx = {t: i for i, t in enumerate(tok_vocab)}
    lab_idx = {c: i for i, c in enumerate(lab_vocab)}
    V, C = len(tok_vocab), len(lab_vocab)
    table = np.zeros((V, C), dtype=np.float64)
    for s, lab in zip(seqs, labs):
        c = lab_idx[lab]
        for t in s:
            table[tok_idx[int(t)], c] += 1.0
    return table, np.array(tok_vocab), np.array(lab_vocab)


def mutual_information(table: np.ndarray) -> tuple[float, float]:
    """(MI, normalized MI in [0,1]) from a count contingency table."""
    total = table.sum()
    if total <= 0:
        return 0.0, 0.0
    P = table / total
    pt = P.sum(axis=1, keepdims=True)  # token marginals
    pc = P.sum(axis=0, keepdims=True)  # label marginals
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = P / (pt @ pc + 1e-300)
        mi_terms = np.where(P > 0, P * np.log(ratio + 1e-300), 0.0)
    mi = float(mi_terms.sum())
    h_t = float(-np.sum(pt[pt > 0] * np.log(pt[pt > 0])))
    h_c = float(-np.sum(pc[pc > 0] * np.log(pc[pc > 0])))
    denom = np.sqrt(h_t * h_c)
    nmi = float(mi / denom) if denom > 0 else 0.0
    return mi, min(max(nmi, 0.0), 1.0)


def chi_square(table: np.ndarray) -> tuple[float, float, float]:
    """(chi2, df, Cramer's V) from a V×C count table."""
    total = table.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = (row @ col) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)))
    V, C = table.shape
    df = max((V - 1) * (C - 1), 1)
    k = min(V, C) - 1
    k = k if k > 0 else 1
    cramers_v = float(np.sqrt(chi2 / (total * k)))
    return chi2, float(df), min(cramers_v, 1.0)


# --------------------------------------------------------------------------- #
# Label-shuffle nulls
# --------------------------------------------------------------------------- #


def shuffle_labels_global(labels: np.ndarray, group, rng: np.random.RandomState) -> np.ndarray:
    """Permute label assignment across all bouts (preserves token multisets + label marginal)."""
    perm = rng.permutation(len(labels))
    return np.asarray(labels)[perm]


def shuffle_labels_within_group(labels: np.ndarray, group, rng: np.random.RandomState) -> np.ndarray:
    """Permute labels ONLY within bouts sharing a group value (e.g. Emitter).

    Controls the speaker-identity confound: a token↔Context association that
    survives this null is not merely mediated by who is vocalizing.
    """
    out = np.array(labels, dtype=object)
    group = np.asarray(group, dtype=object)
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        if len(idx) > 1:
            out[idx] = labels[rng.permutation(idx)]
    return out


def _table_from_bout_columns(token_rows, bout_idx, col_per_bout, V, C) -> np.ndarray:
    """Rebuild the V×C table given a per-bout column assignment (one bincount).

    ``token_rows`` are DENSE token-row indices in [0, V) (raw token ids must be
    mapped first, since V counts distinct tokens, not max id + 1).
    """
    token_cols = col_per_bout[bout_idx]
    flat_index = token_rows * C + token_cols
    counts = np.bincount(flat_index, minlength=V * C)
    return counts.reshape(V, C).astype(np.float64)


def association_test(
    token_lists,
    labels,
    *,
    statistic: str = "nmi",
    n_permutations: int = 200,
    seed: int = 0,
    group=None,
    null_type: str | None = None,
) -> BindingResult:
    """Permutation test of token↔label association.

    Args:
        token_lists: per-bout token lists (aligned with ``labels``).
        labels: per-bout label (Context / Emitter / Addressee value).
        statistic: 'nmi' | 'mi' | 'chi2' | 'cramers_v' — the null is built on this.
        n_permutations: null replicates.
        group: optional per-bout group id (e.g. Emitter) for the within-group null.
        null_type: 'global' (default) or 'within_group'. Within-group requires ``group``.
    """
    seqs = [list(int(t) for t in s) for s in token_lists]
    labels = list(labels)
    if len(seqs) != len(labels):
        raise ValueError("token_lists and labels must be aligned")
    if null_type is None:
        null_type = "within_group" if group is not None else "global"

    table, tok_vocab, lab_vocab = build_contingency(seqs, labels)
    V, C = table.shape
    n_tokens = int(table.sum())
    if V < 2 or C < 2 or n_tokens == 0:
        return BindingResult(
            label_name="", n_bouts=len(seqs), n_tokens=n_tokens,
            n_label_values=C, n_token_values=V,
            observed_mi=0.0, observed_nmi=0.0, observed_chi2=0.0, cramers_v=0.0,
            statistic=statistic, null_mean=0.0, null_std=0.0, p_value=1.0,
            z_score=0.0, n_permutations=0, null_type=null_type,
            extra={"not_evaluable": "fewer than 2 token or label classes"},
        )

    mi, nmi = mutual_information(table)
    chi2, _df, cramers_v = chi_square(table)
    obs = {"mi": mi, "nmi": nmi, "chi2": chi2, "cramers_v": cramers_v}[statistic]

    # Map label values to dense columns AND token ids to dense rows; flatten token
    # instances once so each null replicate is a single bincount over row*col.
    lab_col = {lab: i for i, lab in enumerate(lab_vocab)}
    tok_row = {int(t): i for i, t in enumerate(tok_vocab)}
    bout_idx = (
        np.concatenate([np.full(len(s), i) for i, s in enumerate(seqs)])
        if any(seqs) else np.array([], dtype=np.int64)
    )
    tokens_flat = (
        np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs if len(s)])
        if any(seqs) else np.array([], dtype=np.int64)
    )
    token_rows = np.fromiter(
        (tok_row[int(t)] for t in tokens_flat), dtype=np.int64, count=len(tokens_flat)
    )

    shuffle = shuffle_labels_within_group if null_type == "within_group" else shuffle_labels_global
    labels_obj = np.array(labels, dtype=object)
    grp = list(group) if (group is not None and null_type == "within_group") else None

    rng = np.random.RandomState(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for b in range(n_permutations):
        perm_labels = shuffle(labels_obj, grp, rng)
        perm_col = np.fromiter(
            (lab_col[lab] for lab in perm_labels), dtype=np.int64, count=len(perm_labels)
        )
        tbl = _table_from_bout_columns(token_rows, bout_idx, perm_col, V, C)
        _mi, _nmi = mutual_information(tbl)
        _chi2, _df2, _cv = chi_square(tbl)
        null[b] = {"mi": _mi, "nmi": _nmi, "chi2": _chi2, "cramers_v": _cv}[statistic]

    ge = int(np.sum(null >= obs))
    p_value = (1 + ge) / (n_permutations + 1)
    null_mean, null_std = float(null.mean()), float(null.std())
    z = (obs - null_mean) / (null_std + 1e-12)

    return BindingResult(
        label_name="", n_bouts=len(seqs), n_tokens=n_tokens,
        n_label_values=C, n_token_values=V,
        observed_mi=mi, observed_nmi=nmi, observed_chi2=chi2, cramers_v=cramers_v,
        statistic=statistic, null_mean=null_mean, null_std=null_std,
        p_value=p_value, z_score=z, n_permutations=n_permutations, null_type=null_type,
    )


# --------------------------------------------------------------------------- #
# FDR + family test
# --------------------------------------------------------------------------- #


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values (smallest reachable FDR for each test)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values, dtype=float)[order]
    q = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest rank down
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = [0.0] * m
    for rank, idx in enumerate(order):
        out[idx] = float(q[rank])
    return out


def test_token_annotations(
    token_lists,
    annotations: dict[str, list],
    *,
    n_permutations: int = 200,
    seed: int = 0,
    emitter: list | None = None,
) -> dict[str, BindingResult]:
    """Run the token↔label family (Context / Emitter / Addressee / ...) + BH-FDR.

    For each label in ``annotations`` this runs a global permutation null. If
    ``emitter`` (a parallel per-bout list) is supplied, Context additionally gets
    a within-Emitter null so the reported Context result is speaker-controlled.
    """
    results: dict[str, BindingResult] = {}
    for name, labels in annotations.items():
        labels = list(labels)
        if len(labels) != len(token_lists):
            raise ValueError(f"label {name!r} not aligned with token_lists")
        group = emitter if (name == "Context" and emitter is not None) else None
        null_type = "within_group" if group is not None else "global"
        r = association_test(
            token_lists, labels, statistic="nmi",
            n_permutations=n_permutations, seed=seed, group=group, null_type=null_type,
        )
        r.label_name = name
        results[name] = r

    qs = benjamini_hochberg([r.p_value for r in results.values()])
    for r, q in zip(results.values(), qs):
        r.q_value = q
    return results
