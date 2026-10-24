"""Machine-assisted annotation proposals — spectral excess-energy family.

DELIBERATELY DIFFERENT from the frozen v1 RMS/hysteresis detector: this
module works in the time-frequency domain (per-bin robust background
subtraction, MAD-normalized excess energy, connected temporal runs), so the
proposals it generates cannot be circular with the method under evaluation.
The v1 detector is amplitude-envelope; this is spectral-shape.

These are PROPOSALS, not annotations. The human annotator accepts, adjusts,
splits, merges, or rejects them without seeing v1 output. A proposal-free
subset (~25%) serves as the bias audit.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def spectrogram_excess(
    audio: np.ndarray,
    sr: int,
    *,
    nfft: int = 512,
    hop: int = 128,
    fmin: float = 2e3,
    fmax: float = 125e3,
):
    """Background-subtracted, MAD-normalized spectrogram excess.

    Returns (Z, freqs, times) where Z(f,t) = (S - median_t S) / (MAD_t S + eps).
    """
    from scipy import signal as sps  # noqa: PLC0415

    x = np.asarray(audio, dtype=np.float64)
    freqs, times, S = sps.spectrogram(x, sr, nperseg=nfft, noverlap=nfft - hop)
    band = (freqs >= fmin) & (freqs <= fmax)
    S = S[band]
    freqs = freqs[band]
    # robust per-bin background over time
    med = np.median(S, axis=1, keepdims=True)
    mad = np.median(np.abs(S - med), axis=1, keepdims=True)
    Z = (S - med) / (mad + 1e-12)
    return Z, freqs, times


def temporal_activity(Z: np.ndarray, *, tau: float = 4.0, min_bins: int = 3):
    """Binary column activity: at least ``min_bins`` frequency bins above tau."""
    return (Z > tau).sum(axis=0) >= min_bins


def activity_runs(active: np.ndarray, *, min_frames: int = 3, min_gap: int = 5):
    """Convert the binary activity column to half-open [start, stop) runs.

    Runs shorter than ``min_frames`` are dropped. Runs separated by fewer
    than ``min_gap`` inactive columns are merged. Parameters deliberately
    different from the v1 RMS detector (frames are spectrogram hops, not
    10 ms RMS frames).
    """
    runs = []
    start = None
    gap = 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                if i - gap - start >= min_frames:
                    runs.append((start, i - gap + 1))
                start = None
                gap = 0
    if start is not None and len(active) - start >= min_frames:
        runs.append((start, len(active)))
    return runs


def spectral_proposals(
    audio: np.ndarray,
    sr: int,
    *,
    tau: float = 4.0,
    min_bins: int = 3,
    min_frames: int = 3,
    min_gap: int = 5,
    nfft: int = 512,
    hop: int = 128,
    fmin: float = 2e3,
    fmax: float = 125e3,
) -> list[dict]:
    """Candidate vocal intervals from spectral excess energy (proposal layer).

    Returns [{"start_sample", "end_sample", "peak_z", "n_active_bins"}].
    """
    Z, freqs, times = spectrogram_excess(audio, sr, nfft=nfft, hop=hop, fmin=fmin, fmax=fmax)
    active = temporal_activity(Z, tau=tau, min_bins=min_bins)
    runs = activity_runs(active, min_frames=min_frames, min_gap=min_gap)
    proposals = []
    for s, e in runs:
        seg = Z[:, s:e]
        proposals.append(
            {
                "start_sample": int(times[s] * sr),
                "end_sample": int(times[min(e, len(times) - 1)] * sr),
                "start_ms": round(float(times[s]) * 1000, 1),
                "end_ms": round(float(times[min(e, len(times) - 1)]) * 1000, 1),
                "peak_z": round(float(seg.max()), 2),
                "n_active_bins": int((seg > tau).sum()),
            }
        )
    return proposals
