"""NBD boundary proposal generator (#1 of the refined pipeline).

The fixed-threshold NBD either fires or doesn't. This module turns it into a
CALIBRATION-READY proposal system: compute the raw acoustic signals, then emit
candidate boundaries (local peaks above a low floor) each carrying a feature
vector. A downstream calibrator (``nbd_calibrator``) maps features ->
P(boundary); uncertain candidates become ABSTAIN rather than hard segments.

Signals (per hop frame, default 10 ms):
  energy_change   |dRMS|            (word/pause onsets)
  spectral_flux   positive FFT mag difference (phoneme/syllable change)
  band_flux       low-band spectral flux (formant / energy-band change)
  rms_slope       |gradient of RMS|
  rms             local RMS level (SNR proxy)

Pure numpy. No torch. Reusable by both the calibrator and the live pipeline.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NBDSignals:
    """Per-frame acoustic signal arrays (all length n_frames)."""

    rms: np.ndarray
    energy_change: np.ndarray
    spectral_flux: np.ndarray
    band_flux: np.ndarray
    rms_slope: np.ndarray
    hop_ms: float
    sr: int

    @property
    def n_frames(self) -> int:
        return len(self.rms)


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def compute_signals(audio: np.ndarray, sr: int, hop_ms: float = 10.0) -> NBDSignals:
    """Compute per-frame NBD signals from raw audio.

    Audio is expected normalized (peak ~1). RMS is self-normalized to its max so
    energy_change / rms_slope live in [0, 1].
    """
    hop = max(int(sr * hop_ms / 1000), 1)
    fft_size = hop * 2
    n_frames = max(len(audio) // hop, 1)
    win = max(hop // 2, 1)

    rms = np.array(
        [np.sqrt(np.mean(audio[i * hop : i * hop + win] ** 2)) + 1e-10 for i in range(n_frames)],
        dtype=np.float64,
    )
    rms_n = rms / (rms.max() + 1e-10)
    energy_change = np.abs(np.diff(rms_n, prepend=rms_n[0]))
    rms_slope = np.abs(np.gradient(rms_n))

    hann = _hann(fft_size)
    low_bins = max(fft_size // 12, 1)  # bottom ~8% of the spectrum = low band
    flux = np.zeros(n_frames)
    band = np.zeros(n_frames)
    prev_mag = None
    prev_low = None
    for i in range(n_frames):
        s = i * hop
        e = s + fft_size
        if e > len(audio):
            break
        mag = np.abs(np.fft.rfft(audio[s:e] * hann))
        low = mag[:low_bins]
        if prev_mag is not None:
            flux[i] = np.sum(np.maximum(0.0, mag - prev_mag))
            band[i] = np.sum(np.maximum(0.0, low - prev_low))
        prev_mag = mag
        prev_low = low

    if flux.max() > 1e-10:
        flux /= flux.max()
    if band.max() > 1e-10:
        band /= band.max()

    return NBDSignals(
        rms=rms_n,
        energy_change=energy_change,
        spectral_flux=flux,
        band_flux=band,
        rms_slope=rms_slope,
        hop_ms=hop_ms,
        sr=sr,
    )


def _peaks_above_floor(signal: np.ndarray, floor: float) -> list[int]:
    """Frame index of the argmax peak inside each run above `floor`."""
    peaks: list[int] = []
    above = signal > floor
    i = 0
    n = len(signal)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        peaks.append(i + int(np.argmax(signal[i:j])))
        i = j
    return peaks


@dataclass
class Proposal:
    """A candidate boundary with its raw NBD features."""

    time_ms: float
    frame: int
    energy_change: float
    spectral_flux: float
    band_flux: float
    local_rms: float
    rms_slope: float
    duration_since_prev_ms: float
    detector: str  # "energy" | "spectral" | "both"


def generate_proposals(
    signals: NBDSignals,
    *,
    floor_energy: float = 0.05,
    floor_flux: float = 0.03,
    debounce_ms: float = 30.0,
) -> list[Proposal]:
    """Emit candidate boundaries: local peaks of energy_change / spectral_flux.

    Low floors (not decision thresholds) — the calibrator decides which
    proposals are real. Peaks within debounce_ms are merged (stronger wins).
    """
    e_peaks = _peaks_above_floor(signals.energy_change, floor_energy)
    s_peaks = _peaks_above_floor(signals.spectral_flux, floor_flux)

    # Tag each peak frame with which detector(s) fired.
    tagged: dict[int, set[str]] = {}
    for f in e_peaks:
        tagged.setdefault(f, set()).add("energy")
    for f in s_peaks:
        tagged.setdefault(f, set()).add("spectral")

    # Merge peaks closer than debounce_ms: keep the stronger combined signal.
    frame_ms = signals.hop_ms
    debounce_frames = max(int(debounce_ms / frame_ms), 1)
    frames = sorted(tagged)
    merged: list[int] = []
    for f in frames:
        if merged and f - merged[-1] < debounce_frames:
            prev = merged[-1]
            prev_strength = signals.energy_change[prev] + signals.spectral_flux[prev]
            cur_strength = signals.energy_change[f] + signals.spectral_flux[f]
            if cur_strength > prev_strength:
                merged[-1] = f
                tagged[merged[-1]] = tagged[f] | tagged.get(merged[-1], set())
        else:
            merged.append(f)

    proposals: list[Proposal] = []
    prev_ms = 0.0
    for f in merged:
        detectors = tagged[f]
        detector = "both" if {"energy", "spectral"} <= detectors else next(iter(detectors))
        t = f * frame_ms
        proposals.append(
            Proposal(
                time_ms=t,
                frame=f,
                energy_change=float(signals.energy_change[f]),
                spectral_flux=float(signals.spectral_flux[f]),
                band_flux=float(signals.band_flux[f]),
                local_rms=float(signals.rms[f]),
                rms_slope=float(signals.rms_slope[f]),
                duration_since_prev_ms=t - prev_ms,
                detector=detector,
            )
        )
        prev_ms = t
    return proposals
