"""Spectrogram configuration + species-aware presets.

Species-specific settings matter: bat ultrasonic calls need short windows (1-3 ms) and
ultrasonic frequency bands; marmoset/bird tonal calls benefit from longer windows and
mel/CQT scales; human speech uses standard 25 ms / 10 ms hop. Never default human-speech
STFT settings onto ultrasonic taxa — the FM sweeps and ultrasonic texture would be smeared.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrogramConfig:
    """STFT + log-mel parameters for one species/recording context."""

    sample_rate: int
    n_fft: int
    hop_length: int
    window: str = "hann"
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None  # None -> Nyquist
    log_compress: bool = True
    normalize: str = "peak"  # "peak" | "minmax" | "none"

    @property
    def win_ms(self) -> float:
        return self.n_fft / self.sample_rate * 1000.0

    @property
    def hop_ms(self) -> float:
        return self.hop_length / self.sample_rate * 1000.0


SPECIES_PRESETS: dict[str, SpectrogramConfig] = {
    # Bat ultrasonic FM sweeps: short window (2 ms) + small hop (0.5 ms) + ultrasonic band.
    "bat": SpectrogramConfig(
        sample_rate=250000, n_fft=512, hop_length=128, window="hann",
        n_mels=64, fmin=10000.0, fmax=120000.0,
    ),
    # Marmoset social calls (96 kHz): 10 ms window, 2.5 ms hop, audible-to-ultrasonic band.
    "marmoset": SpectrogramConfig(
        sample_rate=96000, n_fft=1024, hop_length=256, window="hann",
        n_mels=128, fmin=200.0, fmax=48000.0,
    ),
    # Human speech (16 kHz): standard 25 ms window, 10 ms hop.
    "human": SpectrogramConfig(
        sample_rate=16000, n_fft=512, hop_length=160, window="hann",
        n_mels=80, fmin=50.0, fmax=8000.0,
    ),
}


def get_config(species: str) -> SpectrogramConfig:
    """Return the spectrogram preset for a species (or raise KeyError)."""
    if species not in SPECIES_PRESETS:
        raise KeyError(f"no spectrogram preset for species {species!r}; known: {sorted(SPECIES_PRESETS)}")
    return SPECIES_PRESETS[species]
