"""Log-mel spectrogram for learned-model inputs.

Mel-spaced triangular filterbank applied to the STFT magnitude, then log-compressed
and per-segment normalized. The standard input for BioMAE / CNN / Perch-style encoders.

Pure numpy (mel filterbank computed from first principles, no external deps).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import numpy as np

from .stft import compute_stft


def _hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float) -> float:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank: [n_freq, n_mels]."""
    fmax = min(fmax, sr / 2.0)
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = np.array([_mel_to_hz(m) for m in mels])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((len(freqs), n_mels), dtype=np.float64)
    for m in range(n_mels):
        lo, ctr, hi = hz[m], hz[m + 1], hz[m + 2]
        for i, f in enumerate(freqs):
            if lo <= f <= ctr:
                fb[i, m] = (f - lo) / max(ctr - lo, 1e-10)
            elif ctr < f <= hi:
                fb[i, m] = (hi - f) / max(hi - ctr, 1e-10)
    return fb


def compute_log_mel(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    n_fft: int = 512,
    hop_length: int = 128,
    log_eps: float = 1e-10,
) -> np.ndarray:
    """Log-mel spectrogram: [n_mels, n_frames]. Normalized to [0, 1] per segment."""
    fmax = fmax if fmax is not None else sr / 2.0
    mag, _freqs, _times = compute_stft(waveform, sr, n_fft, hop_length)
    fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = fb.T @ mag  # [n_mels, n_frames]
    mel = np.log(mel + log_eps)
    # per-segment peak normalization -> [0, 1]
    mx = mel.max()
    mn = mel.min()
    return (mel - mn) / max(mx - mn, 1e-10) if mx > mn else np.zeros_like(mel)
