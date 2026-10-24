"""Microtexture layer of the fingerprint (Layer 3, species-neutral).

Fine-grained *grain* of the segment: spectral-derivative statistics (frame-to-frame
magnitude change), envelope dynamics (rise/fall rate, range, CV, skew), inter-onset
interval (ICI) statistics, and short-window variances (flux / entropy / flatness).
All universal — describe internal roughness of any signal, any species. No F0-dependent
jitter/shimmer here (those live in ``physics.voiced``, tonal-only).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from . import _dsp


def _moment_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    """mean, std, skew, kurtosis of a 1-D array (robust to size 0/constant)."""
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = x.mean()
    s = x.std()
    if s < _dsp.EPS:
        return float(m), 0.0, 0.0, 0.0
    z = (x - m) / s
    return float(m), float(s), float(np.mean(z**3)), float(np.mean(z**4))


def microtexture(audio, sr):
    """Universal microtexture descriptors (~14-D)."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < 4:
        return np.zeros(14)
    spec = _dsp.spectrum(audio, sr)
    env = _dsp.envelope(audio, sr)
    # spectral-derivative statistics (positive frame-to-frame magnitude change)
    if spec.mag.shape[0] > 1:
        deriv = np.maximum(0.0, spec.mag[1:] - spec.mag[:-1]).sum(axis=1)
        d_mean, d_std, d_skew, d_kurt = _moment_stats(deriv)
        d_max, d_range = (
            float(deriv.max() if deriv.size else 0.0),
            float(np.ptp(deriv) if deriv.size else 0.0),  # ndarray.ptp removed in numpy 2
        )
    else:
        d_mean = d_std = d_skew = d_kurt = d_max = d_range = 0.0
    # envelope dynamics (rise/fall rate, range dB, CV, skew)
    if env.size > 1:
        diff = np.diff(env)
        rise = float(np.maximum(0.0, diff).mean())
        fall = float(np.maximum(0.0, -diff).mean())
        env_db = 20 * np.log10(env + _dsp.EPS)
        dyn_range = float(env_db.max() - env_db.min())
        _, _, env_skew, _ = _moment_stats(env)
        dyn_cv = float(env.std() / (env.mean() + _dsp.EPS))
    else:
        rise = fall = dyn_range = env_skew = dyn_cv = 0.0
    # inter-onset intervals
    onsets = _dsp.detect_onsets(audio, sr)
    if onsets.size > 1:
        ici = np.diff(onsets).astype(float) / sr * 1000.0
        ici_mean = float(ici.mean())
        ici_cv = float(ici.std() / (ici.mean() + _dsp.EPS))
    else:
        ici_mean = ici_cv = 0.0
    # short-window variances
    flux_var = float(np.var(spec.flux)) if spec.flux.size else 0.0
    ent_var = float(np.var(spec.entropy)) if spec.entropy.size else 0.0
    return np.array([
        d_mean, d_std, d_skew, d_kurt, d_max, d_range,
        rise, fall, dyn_range, dyn_cv, env_skew,
        ici_mean, ici_cv,
        flux_var, ent_var,
    ])
