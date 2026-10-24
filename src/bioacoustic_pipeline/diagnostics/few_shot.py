"""Few-shot caller instruments — k-NN and prototype accuracy on frozen embeddings.

The Phase-4 question these measure: does a representation learned WITHOUT
caller supervision (or, for the caller-conditioned arm, with it) generalize to
callers it never saw in training? Support files and query files are disjoint
per caller (channel/session leakage would let a memorized recording answer),
and chance is reported alongside every accuracy — 1/n_callers, stated, never
implied. Measurement instruments in the ``diagnostics.probes`` sense: no model
registration, no verdict axis, ceiling far below meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FewShotResult:
    """Few-shot accuracy over held-out callers with disjoint file splits."""

    accuracy: float
    chance: float
    n_query: int
    n_callers: int
    supports: int
    support_files: dict = field(default_factory=dict)
    query_files: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"few-shot acc={self.accuracy:.3f} (chance={self.chance:.3f}, "
            f"n_callers={self.n_callers}, support={self.supports}, n_query={self.n_query})"
        )


def _split_support_query(callers, files, *, n_support, min_query, rng):
    """Per caller: sample n_support files, keep >=min_query files for query.

    Returns {caller: (support_rows, query_rows, support_file_names,
    query_file_names)}; callers that cannot field both sides are skipped.
    """
    out = {}
    for caller in np.unique(callers):
        rows = np.where(callers == caller)[0]
        cfiles = np.unique(files[rows])
        if len(cfiles) < n_support + min_query:
            continue
        perm = rng.permutation(len(cfiles))
        support = cfiles[perm[:n_support]]
        query = cfiles[perm[n_support:]]
        srows = rows[np.isin(files[rows], support)]
        qrows = rows[np.isin(files[rows], query)]
        out[caller] = (srows, qrows, list(support), list(query))
    return out


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = a @ b.T
    den = np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    return num / np.maximum(den, 1e-12)


def knn_few_shot(
    embeddings,
    caller_ids,
    file_ids,
    *,
    n_support: int = 3,
    k: int = 1,
    seed: int = 0,
    min_query: int = 2,
) -> FewShotResult:
    """k-nearest-neighbour caller accuracy with disjoint support/query files.

    Query windows are labelled by a majority vote over the k nearest support
    windows (cosine). Chance = 1/n_eligible_callers.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    rng = np.random.RandomState(seed)
    split = _split_support_query(
        callers, files, n_support=n_support, min_query=min_query, rng=rng
    )
    eligible = sorted(split)
    if len(eligible) == 0:
        raise ValueError("no caller has enough files to split support/query")
    support_files, query_files, correct, total = {}, {}, 0, 0
    support_rows, support_labels, query_rows, query_labels = [], [], [], []
    for caller in eligible:
        srows, qrows, sfn, qfn = split[caller]
        support_files[caller] = sfn
        query_files[caller] = qfn
        support_rows.extend(int(r) for r in srows)
        support_labels.extend([caller] * len(srows))
        query_rows.extend(int(r) for r in qrows)
        query_labels.extend([caller] * len(qrows))
    sup = emb[support_rows]
    sup_lab = np.asarray(support_labels)
    if k > len(sup_lab):
        k = len(sup_lab)
    sims = _cosine_matrix(emb[query_rows], sup)
    top = np.argsort(-sims, axis=1)[:, :k]
    for qi, neighbors in enumerate(top):
        votes, counts = np.unique(sup_lab[neighbors], return_counts=True)
        pred = votes[np.argmax(counts)]
        correct += int(pred == query_labels[qi])
        total += 1
    return FewShotResult(
        accuracy=correct / max(total, 1),
        chance=1.0 / len(eligible),
        n_query=total,
        n_callers=len(eligible),
        supports=n_support,
        support_files=support_files,
        query_files=query_files,
    )


def prototype_accuracy(
    embeddings,
    caller_ids,
    file_ids,
    *,
    n_support: int = 2,
    seed: int = 0,
    min_query: int = 2,
) -> FewShotResult:
    """Nearest-prototype caller accuracy (mean of support windows per caller)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    callers = np.asarray(caller_ids)
    files = np.asarray(file_ids)
    rng = np.random.RandomState(seed)
    split = _split_support_query(
        callers, files, n_support=n_support, min_query=min_query, rng=rng
    )
    eligible = sorted(split)
    if len(eligible) == 0:
        raise ValueError("no caller has enough files to split support/query")
    prototypes = np.stack([emb[split[c][0]].mean(axis=0) for c in eligible])
    support_files, query_files = {}, {}
    query_rows, query_labels = [], []
    for caller in eligible:
        srows, qrows, sfn, qfn = split[caller]
        support_files[caller] = sfn
        query_files[caller] = qfn
        query_rows.extend(int(r) for r in qrows)
        query_labels.extend([caller] * len(qrows))
    sims = _cosine_matrix(emb[query_rows], prototypes)
    pred = np.asarray(eligible)[np.argmax(sims, axis=1)]
    labels = np.asarray(query_labels)
    return FewShotResult(
        accuracy=float((pred == labels).mean()),
        chance=1.0 / len(eligible),
        n_query=len(labels),
        n_callers=len(eligible),
        supports=n_support,
        support_files=support_files,
        query_files=query_files,
    )
