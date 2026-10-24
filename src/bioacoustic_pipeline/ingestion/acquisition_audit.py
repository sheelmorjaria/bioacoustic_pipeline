"""Acquisition-structure audit — source confounding BEFORE any sweep (§46).

Taxon predictable from recording source alone = acquisition confound: no
feature engineering manufactures cross-source biological generalization.
The audit builds the taxon x source support matrix, yields source-grouped
splits (each source wholly inside one fold), and runs the metadata-only
baseline that quantifies the confound.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import random


def build_source_support_matrix(rows, *, taxon_key, source_key) -> dict:
    """Clip counts per (taxon, source) cell."""
    m: dict = {}
    for r in rows:
        key = (r[taxon_key], r[source_key])
        m[key] = m.get(key, 0) + 1
    return m


def get_grouped_splits(rows, *, group_key, n_splits=3, seed=0):
    """Yield (train_rows, test_rows) with each GROUP wholly in one fold.

    A group appearing in both train and test would leak recording conditions
    (channel, site, equipment) across the split — the thing this audit
    exists to prevent.
    """
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(r[group_key], []).append(i)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    folds = [set() for _ in range(n_splits)]
    for k, name in enumerate(names):
        folds[k % n_splits].update(groups[name])
    for f in folds:
        if not f or len(f) == len(rows):
            continue  # degenerate fold (all groups in one) — skip
        train = [rows[i] for i in range(len(rows)) if i not in f]
        test = [rows[i] for i in sorted(f)]
        yield train, test


def run_metadata_only_baseline(rows, *, target_key, feature_key) -> float:
    """Accuracy predicting target from the metadata feature ALONE.

    Near 1.0 = perfect acquisition confound (target travels with the
    metadata); ~chance = sources are class-balanced. Either way, report it
    before interpreting any acoustic sweep.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.model_selection import cross_val_score  # noqa: PLC0415
    from sklearn.preprocessing import OneHotEncoder  # noqa: PLC0415

    feats = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(
        [[r[feature_key]] for r in rows]
    )
    y = [r[target_key] for r in rows]
    clf = LogisticRegression(max_iter=1000)
    cv = min(3, min(sum(1 for t in y if t == c) for c in set(y)), len(y))
    if cv < 2:
        return 0.0  # cannot cross-validate a single class
    return float(cross_val_score(clf, feats, y, cv=cv).mean())
