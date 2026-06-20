"""Acoustic-physics layer of the fingerprint (Layer 1, species-neutral).

Split by applicability to avoid anthropomorphising non-tonal taxa (the pipeline's
ethological stance: acoustics audit, never a meaning claim):

* ``physics()`` — **universal** descriptors valid for ANY signal: duration, energy,
  ZCR, *linear* spectral shape (centroid/spread/skew/kurtosis), rolloff, flatness,
  entropy, flux, dominant frequency, ADSR envelope. These form the default
  cross-species fingerprint.
* ``voiced()`` — F0, harmonicity, HNR, harmonic ratios, jitter, shimmer: only
  meaningful for **tonal/harmonic** calls. Returns zeros + ``quality`` flags when the
  segment is non-harmonic / ultrasonic, so a 0 is never misread as "poor voice quality."

Mel-MFCC (human-perceptual) lives separately in ``perceptual.py`` and is opt-in.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from . import _dsp

# Heuristic: a segment is treated as tonal/harmonic only if it has a clear spectral
# peak above the flat floor (low flatness) and non-trivial energy. Voiced features are
# populated only then; otherwise they're zeroed with a quality flag, not silently used.
TONAL_FLATNESS_MAX = 0.55
TONAL_PEAK_PROMINENCE = 6.0  # peak/median spectral magnitude ratio


def _is_tonal(spec: _dsp.Spectrum, avg_mag: np.ndarray) -> bool:
    if avg_mag.size == 0:
        return False
    flat = float(np.mean(spec.flatness))
    med = float(np.median(avg_mag)) + _dsp.EPS
    peak = float(avg_mag.max())
    return flat < TONAL_FLATNESS_MAX and (peak / med) > TONAL_PEAK_PROMINENCE


def physics(audio, sr):
    """Universal physics descriptors (~15-D). Valid for any signal, any species."""
    audio = np.asarray(audio, dtype=np.float64)
    n = len(audio)
    dur_ms = n / sr * 1000.0
    if n == 0:
        return np.zeros(15), {"too_short": True, "low_energy": True, "harmonic": False}
    rms = float(np.sqrt(np.mean(audio**2)) + _dsp.EPS)
    peak = float(np.max(np.abs(audio)))
    crest = peak / (rms + _dsp.EPS)
    sign = (audio[:-1] >= 0) != (audio[1:] >= 0)
    zcr = float(sign.sum()) / max(n - 1, 1)
    spec = _dsp.spectrum(audio, sr)
    avg_mag = _dsp.average_spectrum(spec.mag)
    dom_freq = float(spec.freqs[int(np.argmax(avg_mag))]) if avg_mag.size else 0.0
    env = _dsp.envelope(audio, sr)
    hop_hz = 1.0 / (max(int(sr * 10 / 1000), 1) / sr)
    atk, dec, sus, rel = _dsp.adsr(env, hop_hz)
    vec = np.array([
        dur_ms, rms, peak, crest, zcr,
        float(np.mean(spec.centroid)) if spec.centroid.size else 0.0,
        float(np.mean(spec.spread)) if spec.spread.size else 0.0,
        float(np.mean(spec.skewness)) if spec.skewness.size else 0.0,
        float(np.mean(spec.kurtosis)) if spec.kurtosis.size else 0.0,
        float(np.mean(spec.rolloff)) if spec.rolloff.size else 0.0,
        float(np.mean(spec.flatness)) if spec.flatness.size else 0.0,
        float(np.mean(spec.entropy)) if spec.entropy.size else 0.0,
        float(np.mean(spec.flux)) if spec.flux.size else 0.0,
        dom_freq, atk,
    ])
    quality = {
        "too_short": n < int(sr * 0.02),
        "low_energy": rms < 1e-3,
        "harmonic": _is_tonal(spec, avg_mag),
    }
    return vec, quality


def voiced(audio, sr, quality):
    """Voiced/harmonic descriptors (~7-D): F0, f0_range, harmonicity, HNR, jitter,
    shimmer, H1/H2 ratio. Zeroed (not noise) when the segment is non-harmonic."""
    audio = np.asarray(audio, dtype=np.float64)
    if not quality.get("harmonic", False) or len(audio) < 4:
        return np.zeros(7)
    f0, f0_range = _dsp.f0_estimate(audio, sr)
    spec = _dsp.spectrum(audio, sr)
    avg_mag = _dsp.average_spectrum(spec.mag)
    total = avg_mag.sum() + _dsp.EPS
    threshold = avg_mag.max() * 0.1
    harmonicity = float((avg_mag[avg_mag > threshold]).sum() / total)
    signal_energy = float(np.sum(audio**2))
    high = np.abs(np.diff(audio))
    noise_energy = float(np.sum(high**2)) + _dsp.EPS
    hnr = float(min(signal_energy / noise_energy, 100.0))
    env = _dsp.envelope(audio, sr)
    jitter, shimmer = _dsp.perturbation(audio, sr, env)
    harms = _dsp.harmonic_amplitudes(avg_mag, spec.freqs, f0, sr)
    h1_h2 = float(harms[0] / (harms[1] + _dsp.EPS)) if harms[1] > 0 else 0.0
    return np.array([f0, f0_range, harmonicity, hnr, jitter, shimmer, h1_h2])
