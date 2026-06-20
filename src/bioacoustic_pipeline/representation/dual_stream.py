"""Dual-stream segment representation — typed event sequence for diagnostics.

The segment is the unit of analysis. Each segment carries:
  * **shared acoustic state** (CPC/BioMAE/AVES/spectrogram embedding)
  * **syntactic stream** (VQ-VAE discrete token — categorical acoustic unit)
  * **affective stream** (β-VAE continuous vector — graded state/arousal/prosody)
  * **interpretable view** (acoustic fingerprint — physics-grounded)
  * **social metadata** (emitter/addressee/context — for null models)

The dual-stream design is deliberately NOT fused: VQ-VAE = "what categorical
acoustic unit occurred?" β-VAE = "how did graded state vary?" Late fusion only —
the streams are kept separate so cross-stream diagnostics (affect-within-token
variance, token-conditioned affect, alias-affect equivalence) can test their
relationship without losing interpretability.

Usage:
    segments = build_dual_stream(nbd_segments, tokenizer, affect_backbone,
                                 fingerprint_fn, metadata_fn)
    # → list[DualStreamSegment]
    # for seq in group_by_session(segments): ...

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DualStreamSegment:
    """One segment's multi-view, dual-stream typed event."""

    segment_id: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0
    boundary_type: str = ""          # hard | soft | transitional
    boundary_confidence: float = 1.0

    # --- shared representation ---
    z_shared: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # --- syntactic stream (VQ-VAE) ---
    vq_token: int = -1               # discrete acoustic token (-1 = too short)
    vq_code: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vq_distance: float = 0.0         # distance to assigned code (confidence proxy)
    token_confidence: float = 0.0

    # --- affective stream (β-VAE) ---
    affect_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    affect_logvar: np.ndarray = field(default_factory=lambda: np.zeros(0))
    affect_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    affect_intensity: float = 0.0    # ||affect_mu||

    # --- interpretable views ---
    acoustic_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fingerprint_cluster: int | None = None

    # --- social metadata (for null models) ---
    emitter: str | None = None
    addressee: str | None = None
    context: str | None = None

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def as_event(self) -> dict:
        """Compact typed event: the minimal dict for sequence diagnostics."""
        return {
            "segment_id": self.segment_id,
            "t0": self.start_ms,
            "t1": self.end_ms,
            "token": self.vq_token,
            "token_conf": self.token_confidence,
            "affect": self.affect_mu,
            "affect_norm": self.affect_intensity,
            "fingerprint": self.acoustic_fingerprint,
            "fp_cluster": self.fingerprint_cluster,
            "emitter": self.emitter,
            "addressee": self.addressee,
            "context": self.context,
        }


def build_dual_stream(
    nbd_segments: list,
    tokenizer=None,
    affect_backbone=None,
    fingerprint_fn=None,
    metadata_fn=None,
) -> list[DualStreamSegment]:
    """Assemble DualStreamSegments from NBD segments + representation backbones.

    Args:
        nbd_segments: list of orchestration.Segment (with .audio, .sr, .start_ms, etc.)
        tokenizer: object with .tokenize(audio, sr) -> int (VQ-VAE or SpeciesAcousticTokenizer)
        affect_backbone: object with .embed(audio, sr) -> np.ndarray (β-VAE backbone)
        fingerprint_fn: callable(audio, sr) -> np.ndarray (acoustic fingerprint .combined)
        metadata_fn: callable(segment) -> dict with emitter/addressee/context (optional)

    Returns:
        list of DualStreamSegment — the typed event sequence.
    """
    out: list[DualStreamSegment] = []
    for i, seg in enumerate(nbd_segments):
        audio, sr = seg.audio, seg.sr
        ds = DualStreamSegment(
            segment_id=f"seg_{i:05d}",
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            boundary_type=seg.boundary_type,
            boundary_confidence=seg.confidence,
        )

        # syntactic stream
        if tokenizer is not None:
            tok = tokenizer.tokenize(audio, sr)
            ds.vq_token = tok
            ds.token_confidence = max(0.0, 1.0 - ds.vq_distance) if tok != -1 else 0.0

        # affective stream
        if affect_backbone is not None and ds.vq_token != -1:
            try:
                mu = affect_backbone.embed(audio, sr)
                ds.affect_mu = mu
                ds.affect_z = mu  # deterministic (use mu for analysis, not sampled z)
                ds.affect_intensity = float(np.linalg.norm(mu))
            except Exception:  # noqa: BLE001
                pass

        # interpretable view
        if fingerprint_fn is not None:
            ds.acoustic_fingerprint = fingerprint_fn(audio, sr)

        # social metadata
        if metadata_fn is not None:
            meta = metadata_fn(seg) or {}
            ds.emitter = meta.get("emitter")
            ds.addressee = meta.get("addressee")
            ds.context = meta.get("context")

        out.append(ds)
    return out


def group_by_field(
    segments: list[DualStreamSegment], field_name: str
) -> dict[str, list[DualStreamSegment]]:
    """Group segments by a metadata field (e.g., 'emitter' or 'context')."""
    groups: dict[str, list[DualStreamSegment]] = {}
    for seg in segments:
        key = getattr(seg, field_name, None) or "unknown"
        groups.setdefault(key, []).append(seg)
    return groups
