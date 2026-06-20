"""Acoustic-fingerprint package — physics-grounded, species-portable segment descriptor.

v0.1: universal core (physics + macrotexture + microtexture, all linear) + tonal-only
voiced sub-vector + opt-in human-perceptual (Mel-MFCC). NOT a semantic representation;
it audits learned tokens and separates acoustic similarity from functional equivalence.

Layered to avoid anthropomorphism: the cross-species ``combined`` view is universal-only;
voiced (F0/harmonic) and perceptual (Mel) features are flagged sub-vectors populated only
where valid, mirroring the ethological firewall's stance.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from .coherence import (
    alias_acoustic_divergence,
    classify_alias,
    token_fingerprint_coherence,
)
from .extractor import SegmentFingerprint, extract
from . import macrotexture, microtexture, perceptual, physics
from .nulls import alias_divergence_null, species_distance_null, token_coherence_null
from .report import format_fingerprint_report
from .similarity import cosine, mean_to_centroid, normalize_batch

__all__ = [
    "SegmentFingerprint",
    "extract",
    "physics",
    "macrotexture",
    "microtexture",
    "perceptual",
    "cosine",
    "mean_to_centroid",
    "normalize_batch",
    "token_fingerprint_coherence",
    "alias_acoustic_divergence",
    "classify_alias",
    "token_coherence_null",
    "alias_divergence_null",
    "species_distance_null",
    "format_fingerprint_report",
]
