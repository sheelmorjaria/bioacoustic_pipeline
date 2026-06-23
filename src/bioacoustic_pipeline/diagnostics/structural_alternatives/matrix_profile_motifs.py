"""Matrix Profile motif discovery (STUMP) — segmentation-free contiguous recurrence.

Architecture v1.0 "alternative structure" technique. Instead of HECB boundaries + token matching,
treat the call as a continuous multidimensional time series and find which fixed-length subsequences
recurs. STUMP computes, for every window of length m, the distance to its nearest non-trivial
neighbour — the pairs with the smallest distance are MOTIFS (repeating continuous micro-structure,
e.g. a pitch sweep); the largest are DISCORDS (anomalies). Tokenization-free, deterministic, O(n log n).

Implemented on the top PCA component(s) of the frame-fingerprint trajectory (STUMP is happiest on
1–3-D signals). `stumpy` is imported lazily; the module reports ``available``/``load_error`` per the
codebase convention. Firewall-clean (operates on the all-species acoustic fingerprint). NOT meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

try:  # lazy — module stays importable without the ml/extra math dep
    import stumpy  # noqa: F401

    _AVAILABLE = True
    _ERR = ""
except Exception as exc:  # noqa: BLE001
    stumpy = None
    _AVAILABLE = False
    _ERR = repr(exc)

available = _AVAILABLE
load_error = _ERR


def _reduce(trajectory: np.ndarray, n_components: int) -> np.ndarray:
    """PCA-reduce the trajectory to ``n_components`` 1-D signals (one column per component)."""
    X = np.asarray(trajectory, dtype=np.float64)
    if X.shape[0] < 2:
        return X
    from sklearn.decomposition import PCA  # noqa: PLC0415

    n_components = min(n_components, X.shape[1])
    return PCA(n_components=n_components, random_state=0).fit_transform(X)


def matrix_profile(trajectory: np.ndarray, *, m: int = 8, n_components: int = 1) -> dict:
    """STUMP matrix profile on the PCA-reduced trajectory.

    Args:
        trajectory: (n_frames, D) frame-fingerprint trajectory.
        m: subsequence (window) length in frames.
        n_components: PCA components to feed STUMP (1 = univariate).
    """
    if not _AVAILABLE:
        return {"available": False, "load_error": _ERR}
    X = _reduce(trajectory, n_components)
    n = X.shape[0]
    if n < 2 * m:
        return {
            "available": True,
            "n_frames": n,
            "m": m,
            "motif_distance": None,
            "discord_distance": None,
            "n_motifs": 0,
        }
    # STUMP univariate on PC1. stumpy.stump returns an `mparray`: a 2D OBJECT array
    # (col 0 = nearest-neighbour distance, col 1 = index, ...). Take col 0 and coerce
    # object→float64 (raw [:, 0] is object dtype, which np.isfinite rejects).
    mp = np.asarray(stumpy.stump(X[:, 0], m=m)[:, 0], dtype=np.float64)
    mp = mp[np.isfinite(mp)]
    if mp.size == 0:
        return {
            "available": True,
            "n_frames": n,
            "m": m,
            "motif_distance": None,
            "discord_distance": None,
            "n_motifs": 0,
        }
    motif_dist = float(np.min(mp))
    discord_dist = float(np.max(mp))
    # recurrence: fraction of windows whose nearest-neighbour distance is "small" (< motif_dist*3)
    n_motifs = int(np.sum(mp < max(motif_dist * 3.0, 1e-9)))
    return {
        "available": True,
        "n_frames": n,
        "m": m,
        "motif_distance": motif_dist,
        "discord_distance": discord_dist,
        "n_motifs": n_motifs,
        "recurrence_fraction": n_motifs / mp.size,
    }
