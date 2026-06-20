"""Acoustic-fingerprint backbone — the species-portable interpretable bank view.

A deterministic, handcrafted descriptor (no training data → no leakage), so the firewall
registers it as all-species (``applies_to_species: []``). It is the one representation
the firewall lets cross species → the Track B view where no species CPC exists. Tagged
``view_class="interpretable"`` so it is never conflated with learned latents.

``embed`` returns the species-neutral ``combined`` vector (physics + macrotexture +
microtexture); the voiced/perceptual sub-vectors stay with the extractor for diagnostics.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from bioacoustic_pipeline.diagnostics.acoustic_fingerprint import extract

from .backbones import Backbone


class AcousticFingerprintBackbone(Backbone):
    """Physics-grounded acoustic descriptor as an embeddings-bank view."""

    name = "acoustic_fingerprint"
    provenance_name = "acoustic_fingerprint"
    view_class = "interpretable"
    # embed_dim set lazily from the first extraction (universal core length).
    embed_dim = 42

    def __init__(self, *, include_perceptual: bool = False):
        self.include_perceptual = include_perceptual
        self._err = None

    @property
    def available(self) -> bool:
        return True

    @property
    def load_error(self) -> str | None:
        return self._err

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        fp = extract(audio, sr, include_perceptual=self.include_perceptual)
        if self.embed_dim != fp.dim:
            self.embed_dim = fp.dim
        return np.asarray(fp.combined, dtype=np.float32)
