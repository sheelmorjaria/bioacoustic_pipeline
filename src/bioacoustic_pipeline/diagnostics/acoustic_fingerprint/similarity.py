"""Fingerprint similarity primitives (cosine, batch z-normalize, mean-to-centroid).

Batch normalization (z-score per feature across rows) is applied before cosine so that
 wildly-scaled features (duration in ms, centroid in Hz) don't dominate flatness (0-1).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from . import _dsp


def normalize_batch(matrix: np.ndarray) -> np.ndarray:
    """Per-feature z-score across rows. Constant columns -> 0 (ignored by cosine)."""
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return matrix
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < _dsp.EPS] = 1.0
    return (matrix - mean) / std


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _dsp.EPS or nb < _dsp.EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_to_centroid(rows: np.ndarray) -> float:
    """Mean cosine of each row to the centroid (cohesion); rows already normalized."""
    if rows.ndim != 2 or rows.shape[0] < 2:
        return 0.0
    centroid = rows.mean(axis=0)
    return float(np.mean([cosine(r, centroid) for r in rows]))
