"""Hierarchical two-stage segmentation: CRNN call isolation + NBD internal segmentation.

Architecture (per the validated experimental results):
  Stage 1 — Call isolation:
    PCEN + CRNN frame detector → P(call) per frame → hysteresis → call events
    (F1=0.703 one-to-one on MeerKAT; excels at discrete calls in field noise)

  Stage 2 — Internal segmentation (optional, gated by call duration):
    NBD proposals INSIDE each detected call event → sub-call boundaries
    Only runs on calls longer than min_internal_segmentation_ms
    Prevents the over-segmentation that defeated NBD in continuous noise

The NBD calibrator on continuous audio (F1~0.26) is weak because 88% of its false
positives are non-call events. But INSIDE a confirmed call, NBD's energy/flux signals
are meaningful — they find real internal acoustic transitions. The CRNN isolates the
call from noise; NBD finds structure within it.

Hierarchical event model:
  CallEvent — one detected vocalization (CRNN onset/offset)
    ↓ optional
  InternalSegment — sub-call unit (NBD-derived, for multi-part calls)

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class InternalSegment:
    """A sub-call unit derived by NBD inside a CallEvent."""

    parent_call_id: str
    start_s: float
    end_s: float
    boundary_type: str = ""  # hard | soft | transitional
    nbd_confidence: float = 1.0

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000.0

    def as_dict(self) -> dict:
        return {
            "parent_call_id": self.parent_call_id,
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "duration_ms": round(self.duration_ms, 1),
            "boundary_type": self.boundary_type,
            "nbd_confidence": round(self.nbd_confidence, 4),
        }


@dataclass
class CallEvent:
    """One detected vocalization from the CRNN call detector."""

    call_id: str
    file_id: str = ""
    onset_s: float = 0.0
    offset_s: float = 0.0
    crnn_confidence: float = 0.0  # mean P(call) over the event
    call_type_prediction: str | None = None  # optional, from downstream probe
    internal_segments: list[InternalSegment] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.offset_s - self.onset_s) * 1000.0

    @property
    def has_internal_segments(self) -> bool:
        return len(self.internal_segments) > 0

    def as_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "file_id": self.file_id,
            "onset_s": round(self.onset_s, 4),
            "offset_s": round(self.offset_s, 4),
            "duration_ms": round(self.duration_ms, 1),
            "crnn_confidence": round(self.crnn_confidence, 4),
            "call_type_prediction": self.call_type_prediction,
            "n_internal_segments": len(self.internal_segments),
            "internal_segments": [s.as_dict() for s in self.internal_segments],
        }


def proba_to_call_events(
    proba: np.ndarray,
    frame_ms: float,
    enter_thr: float = 0.7,
    exit_thr: float = 0.3,
    min_duration_ms: float = 40.0,
    file_id: str = "",
    start_index: int = 0,
) -> list[CallEvent]:
    """Convert a P(call) timeline to CallEvents via hysteresis + connected components.

    Args:
        proba: P(call) per frame (1-D array).
        frame_ms: duration of each frame in ms.
        enter_thr: enter call state when P(call) >= this.
        exit_thr: exit call state when P(call) < this.
        min_duration_ms: discard events shorter than this.
        file_id: source file identifier.
        start_index: frame index offset (for sub-window detection).

    Returns: list of CallEvent (without internal segments).
    """
    in_call = False
    events: list[CallEvent] = []
    call_counter = 0
    confidence_sum = 0.0

    for i, p in enumerate(proba):
        if not in_call and p >= enter_thr:
            in_call = True
            start_frame = i
            confidence_sum = p
        elif in_call:
            if p < exit_thr:
                in_call = False
                duration_ms = (i - start_frame) * frame_ms
                if duration_ms >= min_duration_ms:
                    onset_s = (start_frame + start_index) * frame_ms / 1000.0
                    offset_s = (i + start_index) * frame_ms / 1000.0
                    events.append(CallEvent(
                        call_id=f"{file_id}_call{call_counter}",
                        file_id=file_id,
                        onset_s=onset_s,
                        offset_s=offset_s,
                        crnn_confidence=confidence_sum / (i - start_frame),
                    ))
                    call_counter += 1
            else:
                confidence_sum += p

    # Handle event extending to the end
    if in_call:
        duration_ms = (len(proba) - start_frame) * frame_ms
        if duration_ms >= min_duration_ms:
            onset_s = (start_frame + start_index) * frame_ms / 1000.0
            offset_s = (len(proba) + start_index) * frame_ms / 1000.0
            events.append(CallEvent(
                call_id=f"{file_id}_call{call_counter}",
                file_id=file_id,
                onset_s=onset_s,
                offset_s=offset_s,
                crnn_confidence=confidence_sum / (len(proba) - start_frame),
            ))

    return events


def segment_call_internally(
    call: CallEvent,
    audio: np.ndarray,
    sr: int,
    profile: dict,
    *,
    min_call_duration_ms: float = 250.0,
    debounce_ms: float = 40.0,
    max_segments: int = 5,
    calibrator=None,
) -> list[InternalSegment]:
    """Run NBD inside a detected call event (Stage 2).

    Only fires if the call is long enough (>= min_call_duration_ms). Prevents
    over-segmentation of short calls. Returns sub-call InternalSegments or empty list.

    Args:
        call: the CallEvent to segment internally.
        audio: full-clip audio array.
        sr: sample rate.
        profile: species NBD profile (energy/spectral thresholds).
        min_call_duration_ms: don't internally segment calls shorter than this.
        debounce_ms: NBD debounce for internal boundaries.
        max_segments: cap on internal segments per call.
        calibrator: optional BoundaryCalibrator to gate internal proposals.
    """
    if call.duration_ms < min_call_duration_ms:
        return []  # short call — use whole call as one unit

    from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: PLC0415

    # Crop audio to the call window
    start_sample = int(call.onset_s * sr)
    end_sample = int(call.offset_s * sr)
    call_audio = audio[start_sample:end_sample]
    if len(call_audio) < int(sr * 0.02):
        return []

    hop_ms = profile.get("hop_ms", 10.0)
    signals = compute_signals(call_audio, sr, hop_ms=hop_ms)

    if calibrator is not None and getattr(calibrator, "fitted", False):
        from bioacoustic_pipeline.orchestration import CALIBRATOR_PROPOSAL  # noqa: PLC0415
        proposals = generate_proposals(signals, **CALIBRATOR_PROPOSAL)
        kept, _abstained = calibrator.apply(proposals)
        boundaries = [(b.time_ms / 1000.0, b.boundary_type, b.confidence) for b in kept]
    else:
        proposals = generate_proposals(
            signals,
            floor_energy=profile.get("energy_threshold", 0.08),
            floor_flux=profile.get("spectral_threshold", 0.04),
            debounce_ms=debounce_ms,
        )
        boundaries = [(p.time_ms / 1000.0, "hard", 1.0) for p in proposals]

    if not boundaries:
        return []

    # Convert boundary times to internal segments within the call
    segments: list[InternalSegment] = []
    all_points = [0.0] + [bt[0] for bt in boundaries] + [call.duration_ms / 1000.0]
    # Deduplicate + sort
    all_points = sorted(set(round(p, 4) for p in all_points))

    for i in range(len(all_points) - 1):
        if len(segments) >= max_segments:
            break
        seg_start = all_points[i]
        seg_end = all_points[i + 1]
        if seg_end - seg_start < debounce_ms / 1000.0:
            continue
        bt = boundaries[i][1] if i < len(boundaries) else "transitional"
        conf = boundaries[i][2] if i < len(boundaries) else 1.0
        segments.append(InternalSegment(
            parent_call_id=call.call_id,
            start_s=call.onset_s + seg_start,
            end_s=call.onset_s + seg_end,
            boundary_type=bt,
            nbd_confidence=conf,
        ))

    return segments


def two_stage_segment(
    audio: np.ndarray,
    sr: int,
    crnn_proba: np.ndarray,
    frame_ms: float,
    profile: dict,
    *,
    enter_thr: float = 0.7,
    exit_thr: float = 0.3,
    min_call_duration_ms: float = 40.0,
    internal_segmentation: bool = True,
    min_internal_duration_ms: float = 250.0,
    file_id: str = "",
    calibrator=None,
) -> list[CallEvent]:
    """Full two-stage segmentation: CRNN call isolation + optional NBD internal.

    Args:
        audio: full-clip audio.
        sr: sample rate.
        crnn_proba: P(call) per frame from the CRNN detector.
        frame_ms: frame duration in ms.
        profile: species NBD profile for internal segmentation.
        enter_thr / exit_thr: hysteresis thresholds for call detection.
        min_call_duration_ms: discard calls shorter than this.
        internal_segmentation: whether to run NBD inside detected calls.
        min_internal_duration_ms: only internally segment calls longer than this.
        file_id: source file identifier.
        calibrator: optional NBD calibrator for internal proposals.

    Returns: list of CallEvent, each optionally containing InternalSegments.
    """
    # Stage 1: call isolation
    events = proba_to_call_events(
        crnn_proba, frame_ms, enter_thr, exit_thr, min_call_duration_ms, file_id
    )

    # Stage 2: internal segmentation (optional, gated)
    if internal_segmentation:
        for event in events:
            event.internal_segments = segment_call_internally(
                event, audio, sr, profile,
                min_call_duration_ms=min_internal_duration_ms,
                calibrator=calibrator,
            )

    return events
