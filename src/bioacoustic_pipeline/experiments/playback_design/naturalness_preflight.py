"""Naturalness preflight — is a synthesized stimulus acoustically close to natural? (Phase 1)

A *preflight* check, not behavioral validation. Before playing a stimulus to animals, screen
it against natural exemplars along three cheap, species-agnostic axes: per-token fingerprint
cosine to the natural sequence, a lean DTW over the per-token fingerprint sequence (order-
robust), and a mean-spectrum distance between the rendered audio and the natural audio.

This is offline acoustic screening only — it can never certify that an animal accepts or
understands the playback (only live response does). Reimplemented lean; the prototype's
``ethological_validation/prosodic_dtw.py`` is the conceptual reference.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dtw_distance(a, b, window: int | None = None) -> float:
    """Lean Sakoe-Chiba-banded DTW over (T, D) feature sequences → cumulative cost.

    ``a``, ``b`` are 2-D arrays (T × D). Returns the path cost normalized by path length.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("inf")
    w = max(abs(na - nb), window or max(na, nb))
    INF = float("inf")
    prev = np.full(nb + 1, INF)
    prev[0] = 0.0
    for i in range(1, na + 1):
        cur = np.full(nb + 1, INF)
        lo = max(1, i - w)
        hi = min(nb, i + w)
        for j in range(lo, hi + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[nb]) / (na + nb)


def _mean_spectrum(audio, sr: int, n_bins: int = 32, frame_ms: float = 25.0) -> np.ndarray:
    """Duration-stable spectral signature: mean |STFT| resampled to ``n_bins``."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return np.zeros(n_bins)
    n = max(int(sr * frame_ms / 1000.0), 64)
    hop = n // 2
    win = np.hanning(n)
    mags = []
    for start in range(0, max(1, len(x) - n + 1), hop):
        seg = x[start : start + n] * win
        m = np.abs(np.fft.rfft(seg))
        mags.append(m)
    if not mags:
        m = np.abs(np.fft.rfft(x[:n] * win if len(x) >= n else x * win[: len(x)]))
        mags = [m]
    spec = np.mean(np.vstack([_resample_row(m, n_bins) for m in mags]), axis=0)
    peak = spec.max()
    return spec / peak if peak > 0 else spec


def _resample_row(row, n_bins: int) -> np.ndarray:
    if len(row) == n_bins:
        return row
    idx = np.linspace(0, len(row) - 1, n_bins)
    return np.interp(idx, np.arange(len(row)), row)


def fingerprint_sequence_distance(stim_fp_seq, natural_fp_seq) -> dict:
    """Order-aware + order-robust distances between two per-token fingerprint sequences.

    Returns ``{mean_cosine, cosine_distance, dtw}``. Shorter/longer sequences are handled
    (cosine over the min prefix; DTW over full sequences).
    """
    stim = [np.asarray(f, dtype=np.float64) for f in stim_fp_seq if np.any(np.asarray(f))]
    nat = [np.asarray(f, dtype=np.float64) for f in natural_fp_seq if np.any(np.asarray(f))]
    if not stim or not nat:
        return {"mean_cosine": 0.0, "cosine_distance": 1.0, "dtw": float("inf")}
    m = min(len(stim), len(nat))
    cos = float(np.mean([_cosine(stim[i], nat[i]) for i in range(m)]))
    return {
        "mean_cosine": round(cos, 4),
        "cosine_distance": round(1.0 - cos, 4),
        "dtw": round(
            dtw_distance(np.vstack(stim), np.vstack(nat), window=max(len(stim), len(nat))), 4
        ),
    }


def spectrogram_distance(stim_audio, natural_audio, sr: int) -> float:
    """L1 distance between the two audios' mean spectral signatures (duration-stable)."""
    a = _mean_spectrum(stim_audio, sr)
    b = _mean_spectrum(natural_audio, sr)
    return float(round(np.abs(a - b).sum(), 4))


def preflight_stimulus(
    stim_audio,
    natural_audio,
    sr,
    stim_fp_seq=None,
    natural_fp_seq=None,
    *,
    fp_cosine_pass: float = 0.7,
) -> dict:
    """Preflight one stimulus against its natural-condition counterpart.

    Returns ``{fingerprint, spectrogram_distance, pass}``. ``pass`` is True when the
    fingerprint cosine to natural is at least ``fp_cosine_pass`` (the stimulus lands inside
    the natural acoustic distribution). Stimuli that fail should NOT be played.
    """
    fp = (
        fingerprint_sequence_distance(stim_fp_seq or [], natural_fp_seq or [])
        if stim_fp_seq is not None and natural_fp_seq is not None
        else {"mean_cosine": None, "cosine_distance": None, "dtw": None}
    )
    spec_d = spectrogram_distance(stim_audio, natural_audio, sr)
    cos = fp["mean_cosine"]
    ok = cos is not None and cos >= fp_cosine_pass
    return {
        "fingerprint": fp,
        "spectrogram_distance": spec_d,
        "pass": bool(ok),
    }
