"""Macrotexture layer of the fingerprint (Layer 2, species-neutral).

Overall *shape* of the segment: envelope curvature/asymmetry, spectral-centroid and
bandwidth trajectories (slope/range), amplitude-modulation rate/depth, and GLCM
spectrogram texture (Haralick). All universal — describe any enveloped signal, any
species. Harmonic-shape descriptors (slope, H1/H2 ratios) are tonal-only and live in
``physics.voiced``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from . import _dsp


def _modulation(env: np.ndarray, sr_hop_hz: float) -> tuple[float, float]:
    """Dominant envelope modulation rate (Hz, 2-16 Hz band) + depth (peak/mean)."""
    if env.size < 4:
        return 0.0, 0.0
    env_c = env - env.mean()
    spec = np.abs(np.fft.rfft(env_c))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / max(sr_hop_hz, 1.0))
    band = (freqs >= 2.0) & (freqs <= 16.0)
    if not band.any():
        return 0.0, 0.0
    rate = float(freqs[band][np.argmax(spec[band])])
    depth = float(env.max() / (env.mean() + _dsp.EPS))
    return rate, depth


def _slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def macrotexture(audio, sr):
    """Universal macrotexture descriptors (~13-D)."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < 4:
        return np.zeros(13)
    spec = _dsp.spectrum(audio, sr)
    env = _dsp.envelope(audio, sr)
    hop_hz = 1.0 / (max(int(sr * 10 / 1000), 1) / sr)
    # envelope shape
    env_n = env / (env.max() + _dsp.EPS)
    curvature = float(np.mean(np.diff(env_n, n=2))) if env_n.size > 2 else 0.0
    asymmetry = float((env_n - env_n.mean()).sum() / (env_n.std() * len(env_n) + _dsp.EPS))
    mod_rate, mod_depth = _modulation(env, hop_hz)
    _, decay_ms, sustain, release_ms = _dsp.adsr(env, hop_hz)
    # spectral-curve trajectories
    c_slope = _slope(spec.centroid) if spec.centroid.size else 0.0
    c_range = float(spec.centroid.max() - spec.centroid.min()) if spec.centroid.size else 0.0
    b_slope = _slope(spec.spread) if spec.spread.size else 0.0
    # GLCM spectrogram texture (time-adjacent co-occurrence)
    g = _dsp.glcm_features(spec.mag)
    return np.array([
        curvature, asymmetry, mod_rate, mod_depth,
        decay_ms, sustain, release_ms,
        c_slope, c_range, b_slope,
        g[0], g[1], g[2], g[3],  # GLCM contrast, correlation, energy, homogeneity
    ])
