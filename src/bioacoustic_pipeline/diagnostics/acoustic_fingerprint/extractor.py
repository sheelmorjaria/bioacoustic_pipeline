"""Acoustic-fingerprint extractor — assembles the per-segment descriptor.

The default ``combined`` view is the **species-neutral core** (physics + macrotexture +
microtexture), all *linear* descriptors valid for any taxon. Two opt-in sub-vectors are
kept separate so they never silently colour the cross-species view, mirroring the
ethological firewall:

* ``voiced`` — F0/harmonicity/HNR/harmonic-ratios/jitter/shimmer: populated only when
  the segment is tonal (else zeros + a ``quality['harmonic']=False`` flag).
* ``perceptual`` — Mel-MFCC (human-auditory); opt-in via ``include_perceptual``.

``combined`` is returned raw (per-feature scales differ wildly); batch normalization for
cosine similarity is done in :mod:`acoustic_fingerprint.similarity`.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import macrotexture, microtexture, perceptual, physics


@dataclass
class SegmentFingerprint:
    """One segment's physics-grounded acoustic descriptor (a view, not a meaning)."""

    segment_id: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0
    species: str = ""
    boundary_type: str = ""
    boundary_confidence: float = 1.0
    physics: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voiced: np.ndarray = field(default_factory=lambda: np.zeros(0))      # tonal-only
    macrotexture: np.ndarray = field(default_factory=lambda: np.zeros(0))
    microtexture: np.ndarray = field(default_factory=lambda: np.zeros(0))
    perceptual: np.ndarray | None = None                                  # Mel-MFCC, opt-in
    combined: np.ndarray = field(default_factory=lambda: np.zeros(0))     # universal concat
    quality: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.combined.shape[0]) if self.combined.size else 0


def extract(
    audio,
    sr,
    *,
    segment_id: str = "",
    start_ms: float = 0.0,
    end_ms: float | None = None,
    species: str = "",
    boundary_type: str = "",
    boundary_confidence: float = 1.0,
    include_perceptual: bool = False,
) -> SegmentFingerprint:
    """Compute the SegmentFingerprint for one audio slice.

    ``combined`` is the species-neutral core (physics + macro + micro). Voiced features
    are filled only for tonal segments; perceptual (Mel-MFCC) only when requested.
    """
    audio = np.asarray(audio, dtype=np.float64)
    phys_vec, quality = physics.physics(audio, sr)
    voiced_vec = physics.voiced(audio, sr, quality)
    macro_vec = macrotexture.macrotexture(audio, sr)
    micro_vec = microtexture.microtexture(audio, sr)
    combined = np.concatenate([phys_vec, macro_vec, micro_vec])
    if quality.get("low_energy") or quality.get("too_short"):
        # a near-silent / tiny segment has no meaningful physics: zero the core so it
        # cannot masquerade as a real cluster member
        combined = np.zeros_like(combined)
    perc = perceptual.perceptual(audio, sr) if include_perceptual else None
    return SegmentFingerprint(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=(end_ms if end_ms is not None else len(audio) / sr * 1000.0),
        species=species,
        boundary_type=boundary_type,
        boundary_confidence=boundary_confidence,
        physics=phys_vec,
        voiced=voiced_vec,
        macrotexture=macro_vec,
        microtexture=micro_vec,
        perceptual=perc,
        combined=combined.astype(np.float32),
        quality=quality,
    )
