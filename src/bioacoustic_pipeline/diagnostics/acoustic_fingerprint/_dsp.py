"""Shared DSP helpers for the acoustic fingerprint (pure numpy).

Species-agnostic, no model deps. Ports the proven primitives from the prototype's
Rust ``micro_dynamics_extractor.rs`` (112-D Rosetta feature vector): a short-time
magnitude spectrogram, RMS envelope, per-frame spectral curves, MFCC (Mel+DCT),
F0 (autocorrelation; YIN is the v0.2 upgrade), onset detection, harmonic-series
extraction, GLCM Haralick texture, envelope ADSR points, and jitter/shimmer.

Mirrors the framing approach in ``segmentation.nbd_proposal.compute_signals``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    return 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _dct_ii(x: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II (matches scipy norm='ortho' / librosa)."""
    n = len(x)
    if n == 0:
        return x
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    out = m @ x
    out[0] *= np.sqrt(1.0 / n)
    out[1:] *= np.sqrt(2.0 / n)
    return out


@dataclass
class Spectrum:
    """Per-frame magnitude spectrogram + derived spectral curves."""

    mag: np.ndarray            # (n_frames, n_freq)
    freqs: np.ndarray          # (n_freq,)
    centroid: np.ndarray       # (n_frames,)
    spread: np.ndarray         # (n_frames,) bandwidth / spectral spread
    skewness: np.ndarray       # (n_frames,)
    kurtosis: np.ndarray       # (n_frames,)
    flatness: np.ndarray       # (n_frames,)
    entropy: np.ndarray        # (n_frames,)
    rolloff: np.ndarray        # (n_frames,) Hz
    flux: np.ndarray           # (n_frames,) positive spectral change vs prev frame


def stft_mag(audio, sr, *, frame_ms=25.0, hop_ms=10.0):
    """Short-time magnitude spectrogram -> (mag [n_frames, n_freq], freqs [n_freq])."""
    audio = np.asarray(audio, dtype=np.float64)
    hop = max(int(sr * hop_ms / 1000), 1)
    n_fft = max(2 * hop, 1)
    n_frames = max(len(audio) // hop, 1)
    win = _hann(n_fft)
    rows = []
    for i in range(n_frames):
        s = i * hop
        e = s + n_fft
        if e > len(audio):
            break
        rows.append(np.abs(np.fft.rfft(audio[s:e] * win)))
    if not rows:  # segment shorter than one frame -> zero-pad a single frame
        frame = np.zeros(n_fft)
        frame[: len(audio)] = audio[:n_fft]
        rows.append(np.abs(np.fft.rfft(frame * win)))
    mag = np.vstack(rows)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return mag, freqs


def envelope(audio, sr, *, hop_ms=10.0, win_ms=10.0):
    """Per-frame RMS envelope (n_frames,)."""
    audio = np.asarray(audio, dtype=np.float64)
    hop = max(int(sr * hop_ms / 1000), 1)
    win = max(int(sr * win_ms / 1000), 1)
    n = max(len(audio) // hop, 1)
    env = np.zeros(n)
    for i in range(n):
        s = i * hop
        env[i] = np.sqrt(np.mean(audio[s : s + win] ** 2)) + EPS
    return env


def spectrum(audio, sr) -> Spectrum:
    """Magnitude spectrogram + per-frame spectral-shape curves (Rosetta-style)."""
    audio = np.asarray(audio, dtype=np.float64)
    mag, freqs = stft_mag(audio, sr)
    psd = mag**2
    total = psd.sum(axis=1, keepdims=True)
    total[total < EPS] = 1.0
    p = psd / total
    centroid = (p * freqs).sum(axis=1)
    dev = freqs[None, :] - centroid[:, None]
    spread = np.sqrt((dev**2 * p).sum(axis=1))
    spd_safe = np.where(spread < EPS, 1.0, spread)
    skewness = (dev**3 * p).sum(axis=1) / (spd_safe**3)
    kurtosis = (dev**4 * p).sum(axis=1) / (spd_safe**4)
    with np.errstate(divide="ignore", invalid="ignore"):
        logpsd = np.log(psd + EPS)
    geom = np.exp(logpsd.mean(axis=1))
    arith = psd.mean(axis=1)
    flatness = geom / (arith + EPS)
    ent = -(p * np.log(p + EPS)).sum(axis=1)
    ent = np.where(np.isfinite(ent), ent, 0.0)
    cum = np.cumsum(psd, axis=1)
    rolloff_idx = (cum < 0.85 * cum[:, -1:]).sum(axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    flux = np.zeros(mag.shape[0])
    if mag.shape[0] > 1:
        flux[1:] = np.maximum(0.0, mag[1:] - mag[:-1]).sum(axis=1)
    return Spectrum(mag, freqs, centroid, spread, skewness, kurtosis,
                    flatness, ent, rolloff, flux)


def average_spectrum(mag: np.ndarray) -> np.ndarray:
    """Time-averaged magnitude spectrum (one vector)."""
    return mag.mean(axis=0) if mag.size else np.zeros(0)


def mel_filterbank(n_fft: int, n_mels: int, sr: int, fmin=0.0, fmax=None) -> np.ndarray:
    """Slaney-style Mel filterbank -> (n_mels, n_freq)."""
    fmax = (fmax or sr / 2.0)
    fmax = min(fmax, sr / 2.0)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for m in range(n_mels):
        left, cen, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        for j, f in enumerate(fft_freqs):
            if left <= f < cen and cen > left:
                fb[m, j] = (f - left) / (cen - left)
            elif cen <= f <= right and right > cen:
                fb[m, j] = (right - f) / (right - cen)
        width = max(mel_to_hz(cen + (mel_pts[m + 2] - mel_pts[m]) / 2) -
                    mel_to_hz(cen - (mel_pts[m + 2] - mel_pts[m]) / 2), 1.0)
        fb[m] *= 2.0 / width
    return fb


def mfcc(audio, sr, *, n_mfcc=13, n_mels=26, frame_ms=25.0, hop_ms=10.0) -> np.ndarray:
    """Mean MFCC over frames (n_mfcc,). librosa-compatible (power dB + orthonormal DCT)."""
    audio = np.asarray(audio, dtype=np.float64)
    mag, _ = stft_mag(audio, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if mag.shape[0] == 0:
        return np.zeros(n_mfcc)
    n_fft = 2 * (mag.shape[1] - 1)
    psd = mag**2
    fb = mel_filterbank(n_fft, n_mels, sr)
    mel = psd @ fb.T  # (n_frames, n_mels)
    log_mel = 10 * np.log10(mel.mean(axis=0) + EPS)
    cc = _dct_ii(log_mel)
    return cc[:n_mfcc]


def f0_estimate(audio, sr, *, fmin=50.0, fmax=None) -> tuple[float, float]:
    """Coarse F0 + range via the autocorrelation of the average spectrum peak.

    A simple, robust estimator for v0.1; the prototype's YIN estimator (``pitch.rs``)
    is the more accurate v0.2 upgrade, especially for ultrasonic taxa.
    Returns (mean_f0_hz, f0_range_hz) — (0, 0) if no clear pitch.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < 4:
        return 0.0, 0.0
    mag, freqs = stft_mag(audio, sr)
    avg = average_spectrum(mag)
    if avg.size == 0:
        return 0.0, 0.0
    fmax = (fmax or sr / 2.0)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return 0.0, 0.0
    # strongest harmonic-structure F0: peak in the low-band spectrum
    cand = freqs[band][np.argmax(avg[band])]
    return float(cand), 0.0


def detect_onsets(audio, sr, *, frame_ms=25.0, hop_ms=10.0) -> np.ndarray:
    """Onset sample indices via half-wave-rectified energy-flux peaks (Rosetta-style)."""
    audio = np.asarray(audio, dtype=np.float64)
    hop = max(int(sr * hop_ms / 1000), 1)
    frame = max(int(sr * frame_ms / 1000), 1)
    n = max(len(audio) // hop, 1)
    env = np.zeros(n)
    for i in range(n):
        s = i * hop
        env[i] = np.sqrt(np.sum(audio[s : s + frame] ** 2))
    if len(env) < 3:
        return np.zeros(0, dtype=int)
    flux = np.maximum(0.0, np.diff(env))
    if flux.size == 0:
        return np.zeros(0, dtype=int)
    thr = flux.mean() + flux.std()
    min_dist = max(int(0.05 * sr / hop), 1)
    onsets = []
    for i in range(1, len(flux) - 1):
        if flux[i] > thr and flux[i] > flux[i - 1] and flux[i] >= flux[i + 1]:
            if not onsets or (i - onsets[-1]) >= min_dist:
                onsets.append(i)
            elif flux[i] > flux[onsets[-1]]:
                onsets[-1] = i
    return np.array(onsets, dtype=int) * hop


def harmonic_amplitudes(avg_spectrum_mag: np.ndarray, freqs: np.ndarray, f0: float, sr: float, n_h=10) -> np.ndarray:
    """Amplitudes of the first ``n_h`` harmonics of ``f0`` (peak-search ±2 bins)."""
    if f0 <= 0 or avg_spectrum_mag.size == 0:
        return np.zeros(n_h)
    bin_hz = sr / (2 * (len(freqs) - 1))
    amps = np.zeros(n_h)
    for h in range(1, n_h + 1):
        hbin = int(round((f0 * h) / bin_hz))
        lo, hi = max(0, hbin - 2), min(len(freqs), hbin + 3)
        if hi > lo:
            amps[h - 1] = avg_spectrum_mag[lo:hi].max()
    return amps


def glcm_features(mag: np.ndarray, n_levels: int = 16) -> np.ndarray:
    """Gray-Level Co-occurrence (horizontal, time-adjacent) Haralick features from the
    spectrogram. Returns [contrast, correlation, energy(ASM), homogeneity] (4-D)."""
    if mag.shape[0] < 2 or mag.size == 0:
        return np.zeros(4)
    lo, hi = mag.min(), mag.max()
    rng = hi - lo
    if rng < EPS:
        return np.zeros(4)
    q = np.clip(((mag - lo) / rng * (n_levels - 1)).round().astype(int), 0, n_levels - 1)
    glcm = np.zeros((n_levels, n_levels))
    a = q[:-1].reshape(-1)
    b = q[1:].reshape(-1)
    np.add.at(glcm, (a, b), 1.0)
    glcm += glcm.T
    glcm /= glcm.sum() + EPS
    levels = np.arange(n_levels)
    contrast = (glcm * (levels[:, None] - levels[None, :]) ** 2).sum()
    mu = (glcm * levels[:, None]).sum()
    sigma = np.sqrt((glcm * (levels[:, None] - mu) ** 2).sum()) + EPS
    correlation = (glcm * (levels[:, None] - mu) * (levels[None, :] - mu)).sum() / (sigma**2)
    energy = (glcm**2).sum()
    homogeneity = (glcm / (1 + abs(levels[:, None] - levels[None, :]))).sum()
    return np.array([contrast, correlation, energy, homogeneity])


def adsr(envelope_arr: np.ndarray, sr_hop_hz: float) -> tuple[float, float, float, float]:
    """Attack/decay/sustain/release (ms) from an RMS envelope (Rosetta-style)."""
    if envelope_arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mx = envelope_arr.max()
    if mx < EPS:
        return 0.0, 0.0, 0.0, 0.0
    peak = int(np.argmax(envelope_arr))
    # attack: time to 90% of peak
    atk = next((i for i, v in enumerate(envelope_arr) if v >= 0.9 * mx), peak)
    attack_ms = atk / max(sr_hop_hz, 1.0) * 1000.0
    # decay: peak -> 10% of peak
    after = envelope_arr[peak:]
    dec = next((i for i, v in enumerate(after) if v <= 0.1 * mx), len(after))
    decay_ms = dec / max(sr_hop_hz, 1.0) * 1000.0
    # sustain: median of the middle two quarters, normalized
    n = len(envelope_arr)
    lo = n // 4
    hi = max(3 * n // 4, lo + 1)
    sustain = float(np.median(envelope_arr[lo:hi]) / mx)
    release_ms = decay_ms * 0.6
    return attack_ms, decay_ms, sustain, release_ms


def perturbation(audio, sr, envelope_arr: np.ndarray) -> tuple[float, float]:
    """Jitter (zero-crossing-period CV) + shimmer (envelope-peak-amplitude CV)."""
    audio = np.asarray(audio, dtype=np.float64)
    sign = (audio[:-1] >= 0) != (audio[1:] >= 0)
    zc = np.where(sign)[0]
    jitter = 0.0
    if len(zc) > 1:
        intervals = np.diff(zc).astype(float)
        m = intervals.mean()
        jitter = float(intervals.std() / m) if m > EPS else 0.0
    # shimmer: CV of envelope peaks
    shimmer = 0.0
    if len(envelope_arr) > 2:
        peaks = []
        for i in range(1, len(envelope_arr) - 1):
            if envelope_arr[i] > envelope_arr[i - 1] and envelope_arr[i] >= envelope_arr[i + 1]:
                peaks.append(envelope_arr[i])
        if len(peaks) > 1:
            pk = np.array(peaks)
            m = pk.mean()
            shimmer = float(pk.std() / m) if m > EPS else 0.0
    return jitter, shimmer
