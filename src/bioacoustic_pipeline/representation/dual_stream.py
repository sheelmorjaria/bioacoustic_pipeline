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
    boundary_type: str = ""  # hard | soft | transitional
    boundary_confidence: float = 1.0

    # --- shared representation ---
    z_shared: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # --- syntactic stream (VQ-VAE) ---
    vq_token: int = -1  # discrete acoustic token (-1 = too short)
    vq_code: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vq_distance: float = 0.0  # distance to assigned code (confidence proxy)
    token_confidence: float = 0.0

    # --- affective stream (β-VAE) ---
    affect_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    affect_logvar: np.ndarray = field(default_factory=lambda: np.zeros(0))
    affect_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    affect_intensity: float = 0.0  # ||affect_mu||

    # --- interpretable views ---
    acoustic_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fingerprint_cluster: int | None = None

    # --- social metadata (for null models) ---
    emitter: str | None = None
    addressee: str | None = None
    context: str | None = None

    # --- hierarchical sub-call structure (HECB internal boundaries) ---
    # A whole-call segment may carry sub-call DualStreamSegments (one per internal
    # boundary). Whole-call representation stays PRIMARY; sub-calls are an augmentation
    # layer (see representation.subcall). None/empty for flat NBD segmentation.
    parent_call_id: str | None = None
    sub_segments: list = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def n_sub_segments(self) -> int:
        return len(self.sub_segments)

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
            "parent_call_id": self.parent_call_id,
            "n_sub": self.n_sub_segments,
            "emitter": self.emitter,
            "addressee": self.addressee,
            "context": self.context,
        }


def _fill_ds(
    ds: DualStreamSegment,
    audio: np.ndarray,
    sr: int,
    *,
    tokenizer=None,
    affect_backbone=None,
    fingerprint_fn=None,
    metadata_src=None,
    metadata_fn=None,
) -> None:
    """Populate one DualStreamSegment's streams from its audio slice (shared helper)."""
    if tokenizer is not None:
        tok = tokenizer.tokenize(audio, sr)
        ds.vq_token = tok
        ds.token_confidence = max(0.0, 1.0 - ds.vq_distance) if tok != -1 else 0.0
    if affect_backbone is not None and ds.vq_token != -1:
        try:
            mu = affect_backbone.embed(audio, sr)
            ds.affect_mu = mu
            ds.affect_z = mu  # deterministic (use mu for analysis, not sampled z)
            ds.affect_intensity = float(np.linalg.norm(mu))
        except Exception:  # noqa: BLE001
            pass
    if fingerprint_fn is not None:
        ds.acoustic_fingerprint = fingerprint_fn(audio, sr)
    if metadata_fn is not None and metadata_src is not None:
        meta = metadata_fn(metadata_src) or {}
        ds.emitter = meta.get("emitter")
        ds.addressee = meta.get("addressee")
        ds.context = meta.get("context")


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
        list of DualStreamSegment — the typed event sequence (flat; no sub-call structure).
    """
    out: list[DualStreamSegment] = []
    for i, seg in enumerate(nbd_segments):
        ds = DualStreamSegment(
            segment_id=f"seg_{i:05d}",
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            boundary_type=seg.boundary_type,
            boundary_confidence=seg.confidence,
        )
        _fill_ds(
            ds,
            seg.audio,
            seg.sr,
            tokenizer=tokenizer,
            affect_backbone=affect_backbone,
            fingerprint_fn=fingerprint_fn,
            metadata_src=seg,
            metadata_fn=metadata_fn,
        )
        out.append(ds)
    return out


def build_dual_stream_from_calls(
    call_events: list,
    audio: np.ndarray,
    sr: int,
    *,
    fingerprint_fn=None,
    tokenizer=None,
    affect_backbone=None,
    metadata_fn=None,
) -> list[DualStreamSegment]:
    """Hierarchical DualStreamSegments from HECB CallEvents (whole call + sub-calls).

    Each CallEvent → one PARENT DualStreamSegment (the whole call, primary
    representation) carrying a ``sub_segments`` list of child DualStreamSegments (one
    per internal boundary, the augmentation layer). Children link back via
    ``parent_call_id``. ``Segmentation.hierarchical.hecb_decode`` /
    ``two_stage_segment`` produce the CallEvents (with ``internal_segments``).

    Args:
        call_events: list of segmentation.hierarchical.CallEvent (with .onset_s,
            .offset_s, .internal_segments, .call_id, .crnn_confidence).
        audio: full-clip audio array.
        sr: sample rate.
        fingerprint_fn / tokenizer / affect_backbone / metadata_fn: as in build_dual_stream.

    Returns:
        list of parent DualStreamSegments (each with its sub_segments populated).
    """
    out: list[DualStreamSegment] = []
    for ci, call in enumerate(call_events):
        s = int(call.onset_s * sr)
        e = max(int(call.offset_s * sr), s + int(0.03 * sr))
        parent = DualStreamSegment(
            segment_id=getattr(call, "call_id", None) or f"call_{ci:05d}",
            start_ms=call.onset_s * 1000.0,
            end_ms=call.offset_s * 1000.0,
            boundary_type="hecb_call",
            boundary_confidence=float(getattr(call, "crnn_confidence", 1.0)),
        )
        _fill_ds(
            parent,
            audio[s:e],
            sr,
            tokenizer=tokenizer,
            affect_backbone=affect_backbone,
            fingerprint_fn=fingerprint_fn,
            metadata_src=call,
            metadata_fn=metadata_fn,
        )
        # sub-call children (one per internal boundary)
        for si, seg in enumerate(getattr(call, "internal_segments", []) or []):
            ss = int(seg.start_s * sr)
            se_ = max(int(seg.end_s * sr), ss + int(0.012 * sr))
            child = DualStreamSegment(
                segment_id=f"{parent.segment_id}_sub{si:02d}",
                start_ms=seg.start_s * 1000.0,
                end_ms=seg.end_s * 1000.0,
                boundary_type="hecb_internal",
                boundary_confidence=float(getattr(seg, "nbd_confidence", 1.0)),
                parent_call_id=parent.segment_id,
            )
            _fill_ds(
                child,
                audio[ss:se_],
                sr,
                tokenizer=tokenizer,
                affect_backbone=affect_backbone,
                fingerprint_fn=fingerprint_fn,
                metadata_src=call,
                metadata_fn=metadata_fn,
            )
            parent.sub_segments.append(child)
        out.append(parent)
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
