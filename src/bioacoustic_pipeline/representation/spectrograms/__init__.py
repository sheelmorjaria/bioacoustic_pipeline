"""Spectrogram representation factory — multi-view acoustic images for NBD segments.

The spectrogram is the bridge between raw audio and every other representation layer
(fingerprint, CPC, BioMAE, AVES). Each segment gets a linear STFT (physics/ultrasonic)
and a log-mel (learned-model input) view, with species-aware settings and quality flags.

Usage:
    from bioacoustic_pipeline.representation.spectrograms import (
        build_segment_spectrograms, get_config,
    )
    cfg = get_config("marmoset")
    spec = build_segment_spectrograms(audio, sr, cfg, species="marmoset")
    spec.linear_stft  # [n_freq, n_frames]
    spec.log_mel      # [n_mels, n_frames]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.representation.spectrograms.config import (
    SPECIES_PRESETS,
    SpectrogramConfig,
    get_config,
)
from bioacoustic_pipeline.representation.spectrograms.factory import (
    SegmentSpectrogramRepresentation,
    build_segment_spectrograms,
    resize_to_fixed,
)
from bioacoustic_pipeline.representation.spectrograms.log_mel import compute_log_mel
from bioacoustic_pipeline.representation.spectrograms.quality import spectrogram_quality
from bioacoustic_pipeline.representation.spectrograms.stft import compute_stft

__all__ = [
    "SPECIES_PRESETS",
    "SpectrogramConfig",
    "get_config",
    "SegmentSpectrogramRepresentation",
    "build_segment_spectrograms",
    "resize_to_fixed",
    "compute_stft",
    "compute_log_mel",
    "spectrogram_quality",
]
