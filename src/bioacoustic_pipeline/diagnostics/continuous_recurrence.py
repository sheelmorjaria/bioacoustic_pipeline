"""Continuous recurrence discovery — frame-level motifs without a codebook.

Corrected design after the translation-invariance identity
(Z(x - a) = Z(x) under z-normalised matching): caller-mean subtraction is a
NO-OP by construction for z-normalised matching, so caller-centering is an
explicit ablation, never inferred from CPC loss contrasts. Windows are
boundary-safe (never cross bouts), the eligible-neighbour restriction is
applied BEFORE the minimum, one global PCA is fit on training data and
frozen, duplicate episodes are suppressed, constant segments rejected, and
attribution is exposure-normalised (Stage 3 caveats carried forward).

Interpretation ceiling: recurrent matches are representation-dependent
acoustic patterns — shared production mechanisms and receiver relevance
require additional evidence.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def _z_normalise(window: np.ndarray) -> np.ndarray:
    """Z-normalise per coordinate; constant coordinates guarded to zero."""
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    return (window - mean) / std


class FixedGlobalTransform:
    """PCA fit on training data, frozen for eval (never refit per bout)."""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self._pca = None

    def fit(self, train_data: np.ndarray):
        from sklearn.decomposition import PCA  # noqa: PLC0415

        self._pca = PCA(n_components=self.n_components, random_state=42).fit(train_data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Transform not fitted")
        return self._pca.transform(data)


def frame_is_padding(frame: np.ndarray) -> bool:
    """A 44-D frame is padding when every feature is exactly zero.

    Window-level constant checks miss partial padding (a few real frames plus
    a padded remainder); validity must be identified at the FRAME level and
    windows built only from valid runs (see ``valid_frame_mask``).
    """
    return bool(np.all(np.asarray(frame) == 0.0))


def valid_frame_mask(trajectory: np.ndarray) -> np.ndarray:
    """Per-frame padding mask for a (T, F) trajectory."""
    return ~np.all(np.asarray(trajectory) == 0.0, axis=1)


def boundary_safe_windows(
    total_frames: int, window_len: int, bout_boundaries: list[int]
) -> list[tuple[int, int]]:
    """Windows that never cross bout boundaries."""
    bounds = sorted({0, *bout_boundaries, total_frames})
    windows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for s in range(lo, hi - window_len + 1):
            windows.append((s, s + window_len))
    return windows


def closest_eligible_neighbour(
    query_idx: int,
    windows: np.ndarray,
    caller_ids: list,
    exclude_same_caller: bool = True,
) -> dict:
    """Closest eligible neighbour — restriction applied BEFORE the minimum."""
    query_caller = caller_ids[query_idx]
    zq = _z_normalise(windows[query_idx])
    best_dist, best_idx = float("inf"), -1
    for j in range(len(windows)):
        if j == query_idx:
            continue
        if exclude_same_caller and caller_ids[j] == query_caller:
            continue
        d = float(np.linalg.norm(zq - _z_normalise(windows[j])))
        if d < best_dist:
            best_dist, best_idx = d, j
    return {"neighbour_idx": best_idx, "distance": best_dist}


def suppress_duplicate_episodes(candidates: list[dict], min_gap: int) -> list[dict]:
    """Collapse overlapping windows into distinct episodes (best distance kept)."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c["start"])
    episodes = [ordered[0]]
    for c in ordered[1:]:
        if c["start"] - episodes[-1]["start"] < min_gap:
            if c.get("distance", float("inf")) < episodes[-1].get("distance", float("inf")):
                episodes[-1] = c
        else:
            episodes.append(c)
    return episodes


def discover_cross_caller_motifs(
    bouts: list[np.ndarray],
    caller_ids: list,
    window_len: int,
    n_families: int = 5,
    exclude_same_caller: bool = True,
    hop: int = 1,
    valid_masks: list | None = None,
    min_valid_frames: int | None = None,
) -> list[dict]:
    """Recurrent cross-caller motifs with boundary-safe windows.

    Window->bout->caller mapping goes through ``window_sources`` — the
    eligible-caller identity of every window is its BOUT's caller, never a
    tiled list. ``hop`` strides the window starts (subsampling for scale);
    matches retain original bout index and frame offset.

    Representation support (2026-10-18 adjudication): ``valid_masks`` (one
    boolean frame mask per bout, same grid as the bout rows) with
    ``min_valid_frames`` excludes windows below the minimum valid-frame count
    from BOTH pools before the neighbour argmin — a window-level
    representation-support criterion applied before selection, so gated/
    single-spike windows can no longer win on z-normalised gate structure
    alone. Both default to None (behaviour unchanged for existing callers).
    """
    windows, sources = [], []
    for b_idx, bout in enumerate(bouts):
        bout = np.asarray(bout)
        for s in range(0, max(len(bout) - window_len + 1, 0), hop):
            windows.append(bout[s : s + window_len])
            sources.append((b_idx, s))
    if not windows:
        return []
    W = np.asarray(windows, dtype=np.float64)
    win_callers = np.array([caller_ids[b] for b, _ in sources])
    # two-sided validity: padding/constant windows leave BOTH pools before
    # any distance is taken (one-sided filtering lets invalid windows win
    # the neighbour argmin and contaminate the saved results)
    # vectorized z-normalisation (per window, per coordinate) + blocked
    # eligible-neighbour argmin — same semantics as the scalar path, ~100x
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    Z = ((W - mu) / sd).reshape(len(W), -1)
    is_const = np.all(np.isclose(W, W[:, :1, :]), axis=(1, 2))
    invalid = is_const | np.all(W == 0.0, axis=(1, 2))
    if valid_masks is not None and min_valid_frames is not None:
        counts = np.array(
            [
                int(np.asarray(valid_masks[b], dtype=bool)[s : s + window_len].sum())
                for b, s in sources
            ]
        )
        invalid = invalid | (counts < min_valid_frames)
    valid = ~invalid
    norms = (Z * Z).sum(axis=1)
    best_idx = np.full(len(W), -1, dtype=np.int64)
    best_dist = np.full(len(W), np.inf)
    B = 1024
    for s in range(0, len(W), B):
        e = min(s + B, len(W))
        d2 = norms[s:e, None] + norms[None, :] - 2.0 * (Z[s:e] @ Z.T)
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        d[:, invalid] = np.inf  # neighbours: invalid pool excluded BEFORE min
        rows = np.arange(s, e)
        d[rows - s, rows] = np.inf  # no self-match
        if exclude_same_caller:
            d[win_callers[s:e, None] == win_callers[None, :]] = np.inf
        j = d.argmin(axis=1)
        best_idx[s:e] = j
        best_dist[s:e] = d[np.arange(e - s), j]
    n_unavailable = 0
    matches = []
    for i in range(len(W)):
        if not valid[i]:
            continue  # invalid queries are not candidates at all
        if best_idx[i] < 0 or not np.isfinite(best_dist[i]):
            n_unavailable += 1  # no eligible neighbour -> unavailable, NOT 0
            continue
        b1, p1 = sources[i]
        b2, p2 = sources[int(best_idx[i])]
        matches.append(
            {
                "bouts": [b1, b2],
                "positions": [p1, p2],
                "caller_ids": [caller_ids[b1], caller_ids[b2]],
                "distance": float(best_dist[i]),
                "is_constant": bool(is_const[i]),
            }
        )
    matches = [m for m in matches if not m["is_constant"]]
    # episode suppression runs on the query-side stream (global frame index)
    cands = [{"start": m["positions"][0], "distance": m["distance"], "match": m} for m in matches]
    kept = suppress_duplicate_episodes(cands, min_gap=max(window_len // 2, 1))
    matches = [c["match"] for c in kept]
    matches.sort(key=lambda m: m["distance"])
    return matches[:n_families]


def compute_exposure_normalized_concentration(
    motif_occurrences: list[dict], exposure: dict
) -> dict:
    """Exposure-normalised Herfindahl over per-caller occurrence RATES."""
    counts: dict = {}
    for occ in motif_occurrences:
        counts[occ["caller"]] = counts.get(occ["caller"], 0) + 1
    rates = {c: counts[c] / exposure[c] for c in counts if exposure.get(c, 0) > 0}
    total = sum(rates.values())
    if total == 0:
        return {"herfindahl_H": 0.0, "caller_counts": counts, "rates": rates}
    props = {c: r / total for c, r in rates.items()}
    return {
        "herfindahl_H": float(sum(p * p for p in props.values())),
        "caller_counts": counts,
        "rates": rates,
    }


def valid_frame_sequences(trajectory: np.ndarray, *, min_run: int = 3):
    """Timestamps-preserving valid runs: never invent adjacency.

    Returns [(start_idx, end_idx, frames)] for each maximal run of valid
    (non-zero) frames. Analyses over valid frames consume runs — transitions
    never cross excluded regions, and genuine recorded gaps keep their
    original temporal relationship (start indices are original positions).
    """
    mask = valid_frame_mask(trajectory)
    runs = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_run:
                runs.append((start, i, trajectory[start:i]))
            start = None
    if start is not None and len(mask) - start >= min_run:
        runs.append((start, len(mask), trajectory[start:]))
    return runs
