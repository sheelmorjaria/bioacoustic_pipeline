"""Spectrogram factory — builds multi-view spectrogram representations for NBD segments.

The spectrogram is the BRIDGE between raw audio and every other representation
(fingerprint, CPC, BioMAE, AVES). Each NBD segment gets:
  * linear STFT magnitude (physics / ultrasonic analysis)
  * log-mel (learned-model input)
  * quality flags
  * duration metadata

Variable-duration spectrograms are preserved for fingerprinting / DTW / motif mining;
use ``resize_to_fixed`` to produce a fixed-size image for CNN/BioMAE-style encoders.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SpectrogramConfig
from .log_mel import compute_log_mel
from .quality import spectrogram_quality
from .stft import compute_stft


@dataclass
class SegmentSpectrogramRepresentation:
    """Multi-view spectrogram representation of one NBD segment."""

    segment_id: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0
    species: str = ""
    boundary_type: str = ""
    sample_rate: int = 0
    linear_stft: np.ndarray = field(default_factory=lambda: np.zeros(0))   # [n_freq, n_frames]
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))         # Hz bins
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))         # frame centers (s)
    log_mel: np.ndarray = field(default_factory=lambda: np.zeros(0))       # [n_mels, n_frames]
    duration_ms: float = 0.0
    quality: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.linear_stft.shape


def build_segment_spectrograms(
    audio: np.ndarray,
    sr: int,
    config: SpectrogramConfig,
    *,
    segment_id: str = "",
    start_ms: float = 0.0,
    end_ms: float | None = None,
    species: str = "",
    boundary_type: str = "",
) -> SegmentSpectrogramRepresentation:
    """Build the multi-view spectrogram for one segment's audio."""
    audio = np.asarray(audio, dtype=np.float64)
    dur_ms = end_ms if end_ms is not None else len(audio) / sr * 1000.0
    fmax = config.fmax if config.fmax is not None else sr / 2.0

    mag, freqs, times = compute_stft(audio, sr, config.n_fft, config.hop_length, config.window)
    mel = compute_log_mel(audio, sr, config.n_mels, config.fmin, fmax,
                          config.n_fft, config.hop_length)
    quality = spectrogram_quality(mag, audio)

    return SegmentSpectrogramRepresentation(
        segment_id=segment_id, start_ms=start_ms, end_ms=end_ms if end_ms else dur_ms,
        species=species, boundary_type=boundary_type, sample_rate=sr,
        linear_stft=mag, freqs=freqs, times=times, log_mel=mel,
        duration_ms=dur_ms, quality=quality,
    )


def resize_to_fixed(spec: np.ndarray, target: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize a 2D spectrogram to a fixed (freq, time) shape for CNN-style encoders.

    Uses bilinear interpolation. Distorts duration — only for learned encoders that need
    fixed-size input; use variable-duration spectrograms for fingerprinting/motif mining.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.fromarray(spec.astype(np.float32), mode="F")
    # PIL resize takes (width, height) = (time, freq)
    resized = img.resize((target[1], target[0]), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)
