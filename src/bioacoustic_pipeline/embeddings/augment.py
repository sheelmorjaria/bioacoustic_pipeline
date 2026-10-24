"""Training-time audio augmentation — noise injection and gain jitter.

Wired through ``AudioSequenceDataset``'s previously-unused ``transform=``
hook so augmentation applies to the raw frame block BEFORE the context/
future cut (noise stays consistent across the prediction boundary — the
point of CPC is to predict through it, not to predict the noise seam).

Numpy-only: usable by torch datasets and offline pipelines alike.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def add_white_noise(audio: np.ndarray, *, snr_db: float, rng: np.random.RandomState):
    """Add Gaussian white noise at a target signal-to-noise ratio (dB).

    A silent input is returned unchanged: SNR is undefined at zero signal
    power, and inventing a scale there would silently corrupt training.
    """
    x = np.asarray(audio)
    dtype = x.dtype
    xd = x.astype(np.float64)
    p_sig = float((xd**2).mean())
    if p_sig <= 0.0:
        return x.copy()
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape) * np.sqrt(p_noise)
    return (xd + noise).astype(dtype)


def random_gain(
    audio: np.ndarray, *, lo: float, hi: float, rng: np.random.RandomState
):
    """Uniform random gain in [lo, hi] — recording-level loudness jitter."""
    if lo > hi:
        raise ValueError(f"lo={lo} must not exceed hi={hi}")
    g = rng.uniform(lo, hi)
    return (np.asarray(audio).astype(np.float64) * g).astype(audio.dtype)
