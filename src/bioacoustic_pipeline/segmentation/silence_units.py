"""Silence-aware acoustic-activity segmentation pilot (Phase 1).

Auditable ACTIVITY units with preserved gaps. The detector identifies
candidate intervals above a calibrated activity rule; it does NOT assert
biological atoms, call types, or vocalization status. Caller attribution
comes from annotations, never from the energy envelope. Gaps, invalid
intervals and rejections are records, not erasures.

Processing order (invariant): source validity -> activity evidence ->
candidate intervals -> (deferred) vocal/source validation. Half-open sample
intervals [s, e) are authoritative; milliseconds are derived display values.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class IntervalStatus(str, Enum):
    BELOW_ACTIVITY = "below_activity_threshold"
    INVALID_OR_PADDED = "invalid_or_padded"
    AMBIGUOUS_ATTRIBUTION = "ambiguous_attribution"
    UNSCORED_EDGE = "unscored_edge"


@dataclass(frozen=True)
class HysteresisConfig:
    enter_db: float = -30.0
    exit_db: float = -36.0
    min_on_frames: int = 2
    min_off_frames: int = 3
    min_unit_frames: int = 2
    frame_ms: float = 10.0

    def __post_init__(self):
        if self.enter_db <= self.exit_db:
            raise ValueError("enter_db must exceed exit_db (hysteresis)")
        if self.min_on_frames < 1 or self.min_off_frames < 1 or self.min_unit_frames < 1:
            raise ValueError("frame counts must be >= 1")
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be positive")

    def as_dict(self) -> dict:
        return {
            "enter_db": self.enter_db,
            "exit_db": self.exit_db,
            "min_on_frames": self.min_on_frames,
            "min_off_frames": self.min_off_frames,
            "min_unit_frames": self.min_unit_frames,
            "frame_ms": self.frame_ms,
        }


@dataclass
class ActivitySpan:
    """A candidate activity unit — half-open [start_frame, stop_frame)."""

    start_frame: int
    stop_frame: int
    sr: int = 0
    start_sample: int = 0
    end_sample: int = 0
    censored_at_end: bool = False
    caller_candidates: list = field(default_factory=list)

    @property
    def attribution_status(self) -> str:
        if len(self.caller_candidates) > 1:
            return "ambiguous"
        if len(self.caller_candidates) == 1:
            return "annotated"
        return "unattributed"

    @property
    def resolved_caller(self):
        return self.caller_candidates[0] if len(self.caller_candidates) == 1 else None


# back-compat alias used by tests
AcousticActivityUnit = ActivitySpan


@dataclass
class TimelineInterval:
    """A non-unit interval — gap, invalid span, or unscored edge."""

    start_frame: int
    stop_frame: int
    status: IntervalStatus


@dataclass
class ActivityResult:
    spans: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    unscored_tail_samples: int = 0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "spans": [
                {
                    "start_frame": s.start_frame,
                    "stop_frame": s.stop_frame,
                    "start_sample": s.start_sample,
                    "end_sample": s.end_sample,
                    "censored_at_end": s.censored_at_end,
                }
                for s in self.spans
            ],
            "gaps": [
                {"start_frame": g.start_frame, "stop_frame": g.stop_frame, "status": g.status.value}
                for g in self.gaps
            ],
            "rejections": self.rejections,
            "unscored_tail_samples": self.unscored_tail_samples,
            "config": self.config,
        }


def frame_log_rms(
    audio: np.ndarray, *, sr: int, frame_ms: float, ref_dbfs: float = 0.0
) -> np.ndarray:
    """Per-frame log-RMS relative to a documented reference.

    ``ref_dbfs``: the reference amplitude in dB re digital full scale; 0.0
    means full scale. Computed in float64 regardless of input dtype.
    """
    x = np.asarray(audio, dtype=np.float64)
    frame = max(int(sr * frame_ms / 1000.0), 1)
    n = len(x) // frame
    if n == 0:
        return np.zeros(0)
    blocks = x[: n * frame].reshape(n, frame)
    rms = np.sqrt((blocks**2).mean(axis=1))
    ref = 10.0 ** (ref_dbfs / 20.0)
    return 20.0 * np.log10(np.maximum(rms, 1e-12) / ref)


def decode_activity(
    levels: np.ndarray, *, frame_valid: np.ndarray, config: HysteresisConfig
) -> ActivityResult:
    """State machine on log-RMS levels. Entries use >=, exits use <.

    Invalid frames are hard breaks (state reset) regardless of min_off.
    Onsets are backdated to the first qualifying frame of the confirming run.
    Offsets close at the first frame of the qualifying low run.
    """
    levels = np.asarray(levels, dtype=np.float64)
    valid = np.asarray(frame_valid, dtype=bool)
    n = len(levels)
    spans: list[ActivitySpan] = []
    gaps: list[TimelineInterval] = []
    rejections: list[dict] = []

    state = "inactive"
    onset = None
    dip_start = None
    on_run = 0
    off_run = 0
    last_active_end = None
    i = 0
    while i < n:
        lv = levels[i]
        ok = valid[i]
        if not ok:
            # hard break
            if state == "active":
                _close(spans, rejections, onset, last_active_end, config, censored=False)
                if dip_start is not None and last_active_end is not None:
                    gaps.append(TimelineInterval(last_active_end, i, IntervalStatus.BELOW_ACTIVITY))
                gaps.append(TimelineInterval(i, i + 1, IntervalStatus.INVALID_OR_PADDED))
                state, onset, dip_start, on_run, off_run = "inactive", None, None, 0, 0
                last_active_end = None
            else:
                if (
                    gaps
                    and gaps[-1].status == IntervalStatus.INVALID_OR_PADDED
                    and gaps[-1].stop_frame == i
                ):
                    gaps[-1].stop_frame = i + 1
                else:
                    gaps.append(TimelineInterval(i, i + 1, IntervalStatus.INVALID_OR_PADDED))
            i += 1
            continue

        above_enter = lv >= config.enter_db
        above_exit = lv >= config.exit_db

        if state == "inactive":
            if above_enter:
                on_run += 1
                if on_run >= config.min_on_frames:
                    state = "active"
                    onset = i - config.min_on_frames + 1
                    last_active_end = i + 1
                    dip_start = None
            else:
                on_run = 0
        else:  # active
            if above_exit:
                last_active_end = i + 1
                dip_start = None
                off_run = 0
            else:
                if dip_start is None:
                    dip_start = i
                off_run += 1
                if off_run >= config.min_off_frames:
                    _close(spans, rejections, onset, dip_start, config, censored=False)
                    state, onset, dip_start, on_run, off_run = "inactive", None, None, 0, 0
                    last_active_end = None
        i += 1

    if state == "active":
        _close(spans, rejections, onset, last_active_end, config, censored=True)
        # trailing gap from last active to end
        if last_active_end is not None and last_active_end < n:
            gaps.append(TimelineInterval(last_active_end, n, IntervalStatus.UNSCORED_EDGE))
    elif last_active_end is not None and last_active_end < n:
        pass  # trailing below-threshold handled below

    # fill observed below-activity gaps between spans (coarse)
    cursor = 0
    fill: list[TimelineInterval] = []
    for s in spans:
        if s.start_frame > cursor:
            fill.append(TimelineInterval(cursor, s.start_frame, IntervalStatus.BELOW_ACTIVITY))
        cursor = max(cursor, s.stop_frame)
    if cursor < n:
        fill.append(TimelineInterval(cursor, n, IntervalStatus.BELOW_ACTIVITY))
    gaps.extend(fill)

    return ActivityResult(spans=spans, gaps=gaps, rejections=rejections, config=config.as_dict())


def _close(spans, rejections, onset, end, config, *, censored):
    if onset is None or end is None or end <= onset:
        return
    if end - onset < config.min_unit_frames:
        rejections.append({"start_frame": onset, "stop_frame": end, "reason": "below_min_duration"})
        return
    spans.append(ActivitySpan(start_frame=onset, stop_frame=end, censored_at_end=censored))


def extract_activity_units(
    audio: np.ndarray,
    *,
    sr: int,
    config: HysteresisConfig,
    ref_amplitude: float = 1.0,
    frame_valid_override: np.ndarray | None = None,
    allow_nonfinite: bool = False,
) -> ActivityResult:
    """Waveform -> activity units (convenience wrapper over decode_activity)."""
    x = np.asarray(audio)
    if not allow_nonfinite and not np.all(np.isfinite(x)):
        raise ValueError(
            "non-finite samples present (pass allow_nonfinite=True or supply frame_valid_override)"
        )
    frame = max(int(sr * config.frame_ms / 1000.0), 1)
    n_frames = len(x) // frame
    tail = len(x) - n_frames * frame
    if frame_valid_override is not None:
        valid = np.asarray(frame_valid_override, dtype=bool)
        # reduce to frame granularity: a frame is valid only if fully valid
        v = valid[: n_frames * frame].reshape(n_frames, frame)
        frame_valid = v.all(axis=1)
        safe = np.where(valid[: n_frames * frame], x[: n_frames * frame], 0.0)
        levels = frame_log_rms(
            safe,
            sr=sr,
            frame_ms=config.frame_ms,
            ref_dbfs=20.0 * np.log10(max(ref_amplitude, 1e-12)),
        )
    else:
        frame_valid = np.ones(n_frames, dtype=bool)
        levels = frame_log_rms(
            x, sr=sr, frame_ms=config.frame_ms, ref_dbfs=20.0 * np.log10(max(ref_amplitude, 1e-12))
        )
    result = decode_activity(levels, frame_valid=frame_valid, config=config)
    result.unscored_tail_samples = tail
    # sample coordinates
    for s in result.spans:
        s.sr = sr
        s.start_sample = s.start_frame * frame
        s.end_sample = s.stop_frame * frame
    return result
