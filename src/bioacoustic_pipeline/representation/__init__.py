"""Representation layer — spectrogram factory + backbone adapters.

The spectrogram is the bridge between raw audio and every other representation
(fingerprint, CPC, BioMAE, AVES). ``SpectrogramBackbone`` conforms to the Backbone
ABC so the ``EmbeddingBank`` treats it as a first-class interpretable view.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.representation.spectrogram_backbone import SpectrogramBackbone

__all__ = ["SpectrogramBackbone"]
