"""Spectrogram backbone — conform to the Backbone ABC so the EmbeddingBank treats
spectrograms as a first-class interpretable view.

The embedding is the time-mean log-mel frequency profile (n_mels-D): a compact, fixed-length
summary of the segment's spectral energy distribution over frequency, averaged across time.
This is the bridge representation — richer than the fingerprint's scalar summaries (it
preserves the frequency-band profile) and deterministic/portable (no training, all-species).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import numpy as np

from bioacoustic_pipeline.embeddings.backbones import Backbone
from bioacoustic_pipeline.representation.spectrograms import compute_log_mel, get_config


class SpectrogramBackbone(Backbone):
    """Log-mel spectrogram → time-mean freq-profile embedding. Always available."""

    view_class = "interpretable"

    def __init__(self, species: str = "human"):
        self.name = f"spectrogram_{species}"
        self.provenance_name = "spectrogram"
        try:
            self._config = get_config(species)
        except KeyError:
            self._config = get_config("human")
        self.embed_dim = self._config.n_mels

    @property
    def available(self) -> bool:
        return True

    @property
    def load_error(self) -> str | None:
        return None

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        cfg = self._config
        fmax = cfg.fmax if cfg.fmax is not None else sr / 2.0
        mel = compute_log_mel(audio, sr, cfg.n_mels, cfg.fmin, fmax,
                              cfg.n_fft, cfg.hop_length)
        return mel.mean(axis=1).astype(np.float32)  # time-mean → n_mels-D freq profile
