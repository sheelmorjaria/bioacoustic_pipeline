"""Cross-taxa silhouette with permutation null (44-D sweep instrument).

Silhouette measures acoustic-geometry separation of labeled groups in the
fingerprint space. Reported WITH a label-permutation null (Phipson–Smyth
corrected p): under class imbalance a bare silhouette can look impressive
for structure that label shuffling reproduces.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def compute_silhouette_by_class(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over samples; 0.0 when fewer than two classes."""
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(embeddings, labels))


def silhouette_with_null(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """(observed silhouette, permutation p) — greater-than-tail null."""
    observed = compute_silhouette_by_class(embeddings, labels)
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(labels)
        if compute_silhouette_by_class(embeddings, perm) >= observed:
            exceed += 1
    return observed, (1.0 + exceed) / (n_permutations + 1.0)
