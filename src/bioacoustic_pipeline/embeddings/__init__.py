"""Embeddings: JSON-driven firewall, multi-view bank, cache, backbone adapters."""

from bioacoustic_pipeline.embeddings.backbones import (
    AVESBackbone,
    BioMAEBackbone,
    Backbone,
    CPCBackbone,
)
from bioacoustic_pipeline.embeddings.bank import BackboneView, EmbeddingBank
from bioacoustic_pipeline.embeddings.cache import EmbeddingCache, audio_hash
from bioacoustic_pipeline.embeddings.firewall import (
    Firewall,
    ModelProvenance,
    ProvenanceViolation,
    TRACK_C_ONLY,
    Use,
    default_firewall,
)

__all__ = [
    "Backbone",
    "CPCBackbone",
    "BioMAEBackbone",
    "AVESBackbone",
    "EmbeddingBank",
    "BackboneView",
    "EmbeddingCache",
    "audio_hash",
    "Firewall",
    "ModelProvenance",
    "ProvenanceViolation",
    "TRACK_C_ONLY",
    "Use",
    "default_firewall",
]
