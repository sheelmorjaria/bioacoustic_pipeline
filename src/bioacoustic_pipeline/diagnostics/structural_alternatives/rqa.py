"""Recurrence Quantification Analysis (RQA) — tokenization-free structural complexity.

Architecture v1.0 "alternative structure" technique (the recommended addition). Unlike DUSTED
(token-sequence alignment), RQA operates directly on the continuous frame-fingerprint trajectory:
it builds a Recurrence Plot (which acoustic states the call returns to over time) and quantifies
its structure. This THRIVES on graded/continuous vocalizations (e.g. Egyptian fruit bat bouts)
where DUSTED's DTW over-merges — RQA gives objective metrics to prove the sub-structure is
"continuous/graded" vs "discrete/sequential".

Metrics (standard RQA):
  recurrence_rate (REC) — density of recurrent points (held ~constant across calls by choosing
      the radius to hit a target rate, so REC itself is not the discriminator).
  determinism (DET) — fraction of recurrent points on DIAGONAL lines (length ≥ l_min). High DET
      ⇒ predictable sequential structure / syntax.
  laminarity (LAM) — fraction on VERTICAL lines (length ≥ v_min). High LAM ⇒ the call holds a
      steady state or stutters.
  mean_diag_length (L), trapping_time (TT) — mean diagonal / vertical line lengths.

Pure numpy (no pyrqa dependency). Firewall-clean: operates on the all-species acoustic
fingerprint trajectory (a physics-grounded descriptor), so it is species-portable and carries no
learned-token semantics. Deterministic. An ACOUSTIC structural audit, NOT validated meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def _cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix (1 - normalised cosine similarity)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def recurrence_plot(
    trajectory: np.ndarray, *, target_rec: float = 0.05, metric: str = "cosine"
) -> tuple[np.ndarray, float]:
    """Build a binary Recurrence Plot at ~``target_rec`` density (off-diagonal).

    The radius is chosen as the ``target_rec`` quantile of the upper-triangular distances, so REC
    is ~constant across calls of different lengths — DET and LAM (the structural metrics) become
    the cross-corpus discriminators. Diagonal excluded.

    Args:
        trajectory: (n_frames, D) frame-fingerprint trajectory.
        target_rec: target recurrence rate (fraction of recurrent off-diagonal points).
        metric: "cosine" (default; matches the fingerprint's geometry) or "euclidean".
    """
    X = np.asarray(trajectory, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.zeros((max(X.shape[0], 0),) * 2, dtype=np.int8), 0.0
    dist = (
        _cosine_dist_matrix(X)
        if metric == "cosine"
        else (np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)))
    )
    n = len(X)
    iu = np.triu_indices(n, k=1)
    off = dist[iu]
    if off.size == 0:
        return np.zeros((n, n), dtype=np.int8), 0.0
    radius = float(np.quantile(off, target_rec))
    rp = (dist <= radius).astype(np.int8)
    np.fill_diagonal(rp, 0)
    return rp, radius


def _line_point_fraction(rp: np.ndarray, min_len: int, axis: str) -> tuple[int, float]:
    """Count recurrent points that lie on runs (lines) of length ≥ min_len.

    axis="diagonal" → diagonal lines (i+k, j+k); "vertical" → vertical lines (i+k, j).
    Returns (points_on_long_lines, mean_line_length_over_long_lines).
    """
    n = rp.shape[0]
    total_long = 0
    lengths: list[int] = []
    if axis == "diagonal":
        for k in range(-(n - 1), n):
            diag = np.diagonal(rp, offset=k).astype(np.int8)
            for run, length in _runs(diag):
                if length >= min_len:
                    total_long += length
                    lengths.append(length)
    else:  # vertical: each column
        for j in range(n):
            for run, length in _runs(rp[:, j].astype(np.int8)):
                if length >= min_len:
                    total_long += length
                    lengths.append(length)
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    return total_long, mean_len


def _runs(x: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encode a 0/1 array → list of (value, length) for value==1 runs."""
    if x.size == 0:
        return []
    changes = np.flatnonzero(x[:-1] != x[1:]) + 1
    bounds = np.concatenate(([0], changes, [x.size]))
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if x[s] == 1:
            out.append((1, int(e - s)))
    return out


def rqa(rp: np.ndarray, *, l_min: int = 2, v_min: int = 2) -> dict:
    """RQA metrics from a binary Recurrence Plot.

    Args:
        rp: (n, n) binary recurrence plot (diagonal excluded).
        l_min / v_min: minimum diagonal / vertical line length to count.
    """
    rp = np.asarray(rp)
    n = rp.shape[0]
    recurrent = int(rp.sum())
    if recurrent == 0 or n < 2:
        return {
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "mean_diag_length": 0.0,
            "trapping_time": 0.0,
            "n_frames": n,
        }
    det_pts, mean_diag = _line_point_fraction(rp, l_min, "diagonal")
    lam_pts, mean_vert = _line_point_fraction(rp, v_min, "vertical")
    return {
        "recurrence_rate": recurrent / (n * n),
        "determinism": det_pts / recurrent,
        "laminarity": lam_pts / recurrent,
        "mean_diag_length": mean_diag,
        "trapping_time": mean_vert,
        "n_frames": int(n),
    }


def rqa_for_trajectory(
    trajectory: np.ndarray,
    *,
    target_rec: float = 0.05,
    l_min: int = 2,
    v_min: int = 2,
    metric: str = "cosine",
) -> dict:
    """Convenience: trajectory → recurrence plot → RQA metrics (+ the radius)."""
    rp, radius = recurrence_plot(trajectory, target_rec=target_rec, metric=metric)
    out = rqa(rp, l_min=l_min, v_min=v_min)
    out["radius"] = radius
    return out
