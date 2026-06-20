"""Linear-frequency STFT for physics/ultrasonic analysis.

Short-Time Fourier Transform returning a magnitude spectrogram (n_freq x n_frames).
Uses species-appropriate window/hop from SpectrogramConfig — NOT human-speech defaults
when applied to ultrasonic taxa (short windows for bat FM sweeps).

Pure numpy.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import numpy as np


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def compute_stft(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-frequency magnitude spectrogram.

    Returns:
        (magnitude [n_freq, n_frames], freqs [n_freq] Hz, times [n_frames] s)
    """
    audio = np.asarray(waveform, dtype=np.float64)
    n = len(audio)
    if n < n_fft:
        audio = np.pad(audio, (0, n_fft - n))
        n = len(audio)
    win = _hann(n_fft) if window == "hann" else np.ones(n_fft)
    n_frames = 1 + (n - n_fft) // hop_length
    mag = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        s = i * hop_length
        frame = audio[s : s + n_fft] * win
        mag[:, i] = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(n_frames) * hop_length / sr
    return mag, freqs, times
