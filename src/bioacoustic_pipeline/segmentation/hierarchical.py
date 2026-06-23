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
                    events.append(
                        CallEvent(
                            call_id=f"{file_id}_call{call_counter}",
                            file_id=file_id,
                            onset_s=onset_s,
                            offset_s=offset_s,
                            crnn_confidence=confidence_sum / (i - start_frame),
                        )
                    )
                    call_counter += 1
            else:
                confidence_sum += p

    # Handle event extending to the end
    if in_call:
        duration_ms = (len(proba) - start_frame) * frame_ms
        if duration_ms >= min_duration_ms:
            onset_s = (start_frame + start_index) * frame_ms / 1000.0
            offset_s = (len(proba) + start_index) * frame_ms / 1000.0
            events.append(
                CallEvent(
                    call_id=f"{file_id}_call{call_counter}",
                    file_id=file_id,
                    onset_s=onset_s,
                    offset_s=offset_s,
                    crnn_confidence=confidence_sum / (len(proba) - start_frame),
                )
            )

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
        segments.append(
            InternalSegment(
                parent_call_id=call.call_id,
                start_s=call.onset_s + seg_start,
                end_s=call.onset_s + seg_end,
                boundary_type=bt,
                nbd_confidence=conf,
            )
        )

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
                event,
                audio,
                sr,
                profile,
                min_call_duration_ms=min_internal_duration_ms,
                calibrator=calibrator,
            )

    return events


# ─── HECB-Seg typed internal-boundary decoder (Stages 3-5) ────────────────────
#
# HECB-Seg = Hierarchical Event-Conditioned Boundary Segmentation. The PCEN+CRNN
# detector (Stage 1, `proba_to_call_events`) is the AUTHORITATIVE call-event
# detector — its threshold-crossing onset/offset (~21 ms MAE on MeerKAT) is already
# more precise than any 20 ms-quantized NBD proposal. So this layer does NOT refine
# CRNN call-event boundaries (snapping is opt-in and off by default; it empirically
# coarsens them). Its job is to ANNOTATE typed internal sub-call structure inside
# long/complex calls: each NBD proposal inside a CRNN event is classified
# (onset/offset/internal/noise) by a trained typed classifier, and the internal-typed
# ones become InternalSegments. This gives downstream representation/motif analysis a
# sub-call structure the CRNN alone cannot provide.
#
# Inputs are in MILLISECONDS (matching NBD proposal times); outputs are CallEvents in
# seconds (matching the rest of this module). Pure-stdlib (numpy only) — the
# torch-dependent CRNN inference + typed-classifier training live in
# `scripts/train_hecb_seg.py`; this is the decoder the script and pipeline share.

HECB_ONSET = 1
HECB_OFFSET = 2
HECB_INTERNAL = 3
HECB_NOISE = 0

# Conservative-snap gate (opt-in; OFF by default because snapping coarsens CRNN timing).
HECB_SNAP_CONF = 0.5
HECB_SNAP_DIST_MS = 60.0


@dataclass
class HECBDurationPriors:
    """Call-duration / gap priors learned from ground-truth spans (Stage 5 constraints)."""

    min_call_ms: float = 40.0
    max_call_ms: float = 2000.0
    min_gap_ms: float = 30.0
    max_internal_per_call: int = 5
    median_call_ms: float = 100.0


def fit_hecb_priors(gt_spans_sec: list[list[tuple[float, float]]]) -> HECBDurationPriors:
    """Learn call-duration + inter-call-gap priors from GT spans (seconds)."""
    durs = [(e - s) * 1000.0 for spans in gt_spans_sec for s, e in spans]
    gaps: list[float] = []
    for spans in gt_spans_sec:
        ss = sorted(spans)
        for i in range(1, len(ss)):
            gaps.append((ss[i][0] - ss[i - 1][1]) * 1000.0)
    return HECBDurationPriors(
        min_call_ms=max(30.0, float(np.percentile(durs, 5)) if durs else 40.0),
        max_call_ms=min(5000.0, float(np.percentile(durs, 99)) if durs else 2000.0),
        min_gap_ms=max(20.0, float(np.percentile(gaps, 5)) if gaps else 30.0),
        median_call_ms=float(np.median(durs)) if durs else 100.0,
    )


def _safe_snap(
    boundary_ms: float,
    ev_props: list[tuple[float, np.ndarray]],
    label_idx: int,
    conf: float,
    dist_ms: float,
) -> float:
    """Snap a CRNN boundary to a typed proposal only if one is high-confidence (``conf``),
    close (``dist_ms``), and role-matched (``label_idx``). Otherwise keep the CRNN
    boundary verbatim. Pass conf=inf to disable snapping entirely."""
    cands = [
        (t, p[label_idx])
        for t, p in ev_props
        if p[label_idx] > conf and abs(t - boundary_ms) < dist_ms
    ]
    if not cands:
        return boundary_ms
    return max(cands, key=lambda x: x[1])[0]


def hecb_decode(
    crnn_events: list[tuple[float, float, float]],
    typed_proposals: list[tuple[float, np.ndarray]],
    priors: HECBDurationPriors,
    *,
    pad_ms: float = 50.0,
    snap: bool = False,
    snap_conf: float = HECB_SNAP_CONF,
    snap_dist_ms: float = HECB_SNAP_DIST_MS,
    min_internal_call_ms: float = 250.0,
    gate_durations: bool = False,
    min_parent_confidence: float = 0.0,
    file_id: str = "",
) -> list[CallEvent]:
    """Stage 5: decode CRNN events + typed proposals into CallEvents with internal segments.

    The CRNN detector is AUTHORITATIVE for call-event onset/offset. By default
    (``snap=False``, ``gate_durations=False``) every CRNN event is preserved verbatim and
    HECB only *annotates* typed internal sub-call boundaries — onset/offset are never
    altered. ``snap=True`` opt-in refines edges to typed proposals; ``gate_durations=True``
    opt-in rejects sub-``priors.min_call_ms`` events and clamps over-``max_call_ms`` ones
    (spurious-event hygiene). For each event, internal-typed proposals (confidence > 0.3,
    inside the call body, respecting min-gap + max-count) become InternalSegments — but only
    for calls longer than ``min_internal_call_ms`` (short calls are atomic units).

    ``min_parent_confidence``: HECB **abstain mechanism** — if a CRNN event's confidence
    is below this threshold, HECB skips internal annotation for that event (produces a
    CallEvent with no InternalSegments). Prevents noisy internal boundaries from polluting
    the pooled representation when the parent event is unreliable. Default 0.0 (no abstain).

    Args:
        crnn_events: [(onset_ms, offset_ms, confidence), ...].
        typed_proposals: [(time_ms, proba_4vec), ...] where proba_4vec is indexed by
            [HECB_NOISE, HECB_ONSET, HECB_OFFSET, HECB_INTERNAL].
        priors: learned HECBDurationPriors.
        snap: enable conservative onset/offset snapping (default False — preserve CRNN).
        snap_conf / snap_dist_ms: snap gate (only used when snap=True).
        min_internal_call_ms: only segment calls longer than this internally.
        gate_durations: opt-in min/max duration gating (default False — strict CRNN
            preservation: no event dropped, no offset clamped).
        file_id: source file id for generated call_ids.

    Returns: list[CallEvent] (seconds), each with typed InternalSegments where applicable.
    """
    conf = snap_conf if snap else float("inf")
    dist = snap_dist_ms if snap else float("inf")
    decoded: list[CallEvent] = []
    for ev_idx, (onset_ms, offset_ms, ev_conf) in enumerate(crnn_events):
        ev_props = [
            (t, p) for t, p in typed_proposals if onset_ms - pad_ms <= t <= offset_ms + pad_ms
        ]

        refined_onset = _safe_snap(onset_ms, ev_props, HECB_ONSET, conf, dist)
        refined_offset = _safe_snap(offset_ms, ev_props, HECB_OFFSET, conf, dist)
        if refined_offset <= refined_onset:
            refined_offset = offset_ms

        call_dur = refined_offset - refined_onset
        if gate_durations:
            if call_dur < priors.min_call_ms:
                continue  # opt-in: reject spurious short event
            if call_dur > priors.max_call_ms:
                refined_offset = refined_onset + priors.max_call_ms
                call_dur = priors.max_call_ms

        call = CallEvent(
            call_id=f"{file_id}_hecb{ev_idx}" if file_id else f"hecb{ev_idx}",
            file_id=file_id,
            onset_s=refined_onset / 1000.0,
            offset_s=refined_offset / 1000.0,
            crnn_confidence=ev_conf,
        )

        # Typed internal boundaries (only for long, confident calls)
        if call_dur > min_internal_call_ms and ev_conf >= min_parent_confidence:
            internal_props = [
                (t, p[HECB_INTERNAL])
                for t, p in ev_props
                if p[HECB_INTERNAL] > 0.3 and refined_onset + 50 < t < refined_offset - 50
            ]
            internal_props.sort(key=lambda x: -x[1])  # confidence desc
            placed: list[float] = []
            for t, _prob in internal_props:
                if len(placed) >= priors.max_internal_per_call:
                    break
                if all(abs(t - pt) > priors.min_gap_ms for pt in placed):
                    placed.append(t)
            # Build InternalSegments from sorted boundary points
            cuts = sorted(placed)
            edges = (
                [refined_onset / 1000.0] + [c / 1000.0 for c in cuts] + [refined_offset / 1000.0]
            )
            for k in range(len(edges) - 1):
                call.internal_segments.append(
                    InternalSegment(
                        parent_call_id=call.call_id,
                        start_s=edges[k],
                        end_s=edges[k + 1],
                        boundary_type="internal",
                        nbd_confidence=1.0,
                    )
                )

        decoded.append(call)

    return decoded
