#!/usr/bin/env python3
"""
Ultrasonic Log-Linear Spectrogram Computation

Replaces Mel-scale filterbanks with linear frequency axis to preserve
ultrasonic formant geometry for bioacoustic analysis.

Key differences from Mel-spectrograms:
- Linear frequency bins (constant Hz spacing, not warped)
- Preserves absolute frequency information critical for ultrasonic taxa
- No anthropocentric perceptual bias

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
# See LICENSE file for licensing information.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchaudio


@dataclass
class SpectrogramConfig:
    """Configuration for ultrasonic spectrogram computation."""

    sample_rate: int = 96000  # Support ultrasonic sampling
    n_fft: int = 1024  # FFT window size
    hop_length: int = 240  # ~2.5ms at 96kHz
    win_length: Optional[int] = None  # Defaults to n_fft
    power: float = 2.0  # Power spectrogram (2.0) or magnitude (1.0)
    top_db: float = 80.0  # Dynamic range in dB
    normalize: bool = False  # Normalize waveform before computing

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins."""
        return self.n_fft // 2 + 1

    @property
    def freq_resolution_hz(self) -> float:
        """Frequency resolution in Hz (linear spacing)."""
        return self.sample_rate / self.n_fft

    @property
    def time_resolution_ms(self) -> float:
        """Time resolution in milliseconds."""
        return (self.hop_length / self.sample_rate) * 1000.0


def create_spectrogram(config: Optional[SpectrogramConfig] = None) -> UltrasonicSpectrogram:
    """Factory function to create spectrogram transformer."""
    if config is None:
        config = SpectrogramConfig()
    return UltrasonicSpectrogram(config)


class UltrasonicSpectrogram(nn.Module):
    """
    Computes log-linear spectrograms tailored for ultrasonic taxa.

    Unlike Mel-spectrograms which apply perceptual warping to the frequency axis,
    this module preserves linear frequency spacing. This is critical for:

    1. **Ultrasonic formants**: Bat echolocation (20-100kHz) requires linear
       frequency bins to preserve harmonic structure.

    2. **Species identification**: Absolute frequency cues (e.g., 60kHz vs 65kHz)
       are compressed by Mel-scaling but preserved here.

    3. **Physics-based analysis**: Frequency modulation rates, chirp slopes,
       and spectral entropy are computed accurately on linear scale.

    Args:
        config: SpectrogramConfig with parameters

    Example:
        >>> config = SpectrogramConfig(sample_rate=96000)
        >>> spec = UltrasonicSpectrogram(config)
        >>> waveform = torch.randn(1, 48000)  # 0.5s at 96kHz
        >>> log_spec = spec(waveform)
        >>> print(log_spec.shape)  # (1, 513, 200) - (B, Freq, Time)
    """

    def __init__(self, config: SpectrogramConfig):
        super().__init__()
        self.config = config

        # Spectrogram computation (power spectrum)
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            power=config.power,
            normalized=False,
        )

        # Amplitude to dB conversion
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=config.top_db
        )

    def to(self, device):
        """Move module to device and ensure spectrogram window is also moved."""
        result = super().to(device)
        # Explicitly move the spectrorogram's window to the device
        if hasattr(self.spectrogram, "spectrogram"):
            if hasattr(self.spectrogram.spectrogram, "window"):
                self.spectrogram.spectrogram.window = self.spectrogram.spectrogram.window.to(device)
        return result

    def forward(self, waveform: torch.Tensor, return_db: bool = True) -> torch.Tensor:
        """
        Compute spectrogram from audio waveform.

        Args:
            waveform: Input audio (Batch, Time) or (Time,)
            return_db: If True, return log-power spectrogram; else linear power

        Returns:
            Spectrogram of shape (Batch, Freq_bins, Time_frames)
            - Freq_bins = n_fft // 2 + 1 (e.g., 513 for n_fft=1024)
            - Time_frames depends on input length
        """
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Optional normalization
        if self.config.normalize:
            waveform = waveform / (waveform.abs().max(dim=1, keepdim=True)[0] + 1e-8)

        # Compute power spectrogram
        spec = self.spectrogram(waveform)

        if return_db:
            spec = self.amplitude_to_db(spec)

        return spec

    def frequency_axis(self) -> torch.Tensor:
        """
        Get the frequency axis values in Hz.

        Returns:
            1D tensor of frequency values (linearly spaced from 0 to Nyquist)
        """
        nyquist = self.config.sample_rate / 2
        return torch.linspace(0, nyquist, self.config.freq_bins)

    def time_axis_for_length(self, num_samples: int) -> torch.Tensor:
        """
        Get the time axis values in seconds for a given input length.

        Args:
            num_samples: Number of samples in the input waveform

        Returns:
            1D tensor of time values in seconds
        """
        num_frames = 1 + (num_samples - self.config.n_fft) // self.config.hop_length
        return torch.linspace(0, num_samples / self.config.sample_rate, num_frames)

    def expected_output_length(self, num_samples: int) -> int:
        """
        Calculate the expected number of time frames for an input length.

        Args:
            num_samples: Number of samples in the input waveform

        Returns:
            Number of time frames in the spectrogram
        """
        return 1 + (num_samples - self.config.n_fft) // self.config.hop_length


class FrequencyCompensator(nn.Module):
    """
    Optional frequency compensation for ultrasonic recordings.

    Many ultrasonic detectors exhibit roll-off at high frequencies.
    This module applies pre-emphasis to boost high frequencies.

    Note: This is optional and should be calibrated per recording device.
    """

    def __init__(self, sample_rate: int, pre_emphasis: float = 0.97):
        super().__init__()
        self.pre_emphasis = pre_emphasis
        self.register_buffer("filter", torch.tensor([1.0, -pre_emphasis]))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter."""
        if self.pre_emphasis == 0.0:
            return waveform
        return torch.nn.functional.conv1d(
            waveform.unsqueeze(1), self.filter.view(1, 1, -1), padding=1
        ).squeeze(1)


# Preset configurations for common ultrasonic taxa

BAT_CONFIG = SpectrogramConfig(
    sample_rate=96000,  # Standard bat detector sampling
    n_fft=1024,  # ~10.7ms window
    hop_length=240,  # ~2.5ms hop
    top_db=80.0,
)

CETACEAN_CONFIG = SpectrogramConfig(
    sample_rate=192000,  # High-end for dolphin clicks
    n_fft=2048,  # ~10.7ms window
    hop_length=480,  # ~2.5ms hop
    top_db=80.0,
)

BIRD_CONFIG = SpectrogramConfig(
    sample_rate=48000,  # Standard for birdsong
    n_fft=1024,  # ~21.3ms window
    hop_length=256,  # ~5.3ms hop
    top_db=80.0,
)


def create_spectrogram_for_taxa(taxa: str) -> UltrasonicSpectrogram:
    """
    Create a pre-configured spectrogram for a specific taxonomic group.

    Args:
        taxa: One of 'bat', 'cetacean', 'bird', or 'default'

    Returns:
        UltrasonicSpectrogram with appropriate configuration
    """
    taxa = taxa.lower()
    if taxa == "bat":
        return UltrasonicSpectrogram(BAT_CONFIG)
    elif taxa in ("cetacean", "dolphin", "whale"):
        return UltrasonicSpectrogram(CETACEAN_CONFIG)
    elif taxa == "bird":
        return UltrasonicSpectrogram(BIRD_CONFIG)
    else:
        return UltrasonicSpectrogram(SpectrogramConfig())
