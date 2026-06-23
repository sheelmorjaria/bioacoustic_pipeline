"""Temporal ecology — stateful signal context (Step 10).

Extracts per-call temporal features from an ordered call timeline: inter-call interval,
bout position, bout length, turn-taking latency, overlap flag, daily phase. These capture
the DYNAMIC context a call occurs in — not just what it sounds like, but WHEN and in what
sequence. Connects to the turn_structure axis + the H0-CMC temporal dimension.

Lightweight dataclass + extractor. NOT meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemporalContext:
    """Per-call temporal ecology features."""

    time_since_last_call: float = 0.0  # seconds (0 for first call)
    bout_position: int = 0  # call index within its bout
    bout_length: int = 1  # total calls in this bout
    turn_taking_latency: float | None = None  # seconds since last conspecific call
    overlap_flag: bool = False  # did this call temporally overlap another?
    daily_phase: float | None = None  # 0–24h circadian
    pre_behavior: str | None = None  # what the emitter was doing before
    post_behavior: str | None = None  # what the emitter did after

    def as_dict(self) -> dict:
        return {
            "time_since_last_call": round(self.time_since_last_call, 3),
            "bout_position": self.bout_position,
            "bout_length": self.bout_length,
            "turn_taking_latency": round(self.turn_taking_latency, 3)
            if self.turn_taking_latency
            else None,
            "overlap_flag": self.overlap_flag,
            "daily_phase": self.daily_phase,
        }


def extract_temporal_context(
    call_idx: int,
    timeline: list[dict],
    *,
    bout_gap_threshold: float = 2.0,
) -> TemporalContext:
    """Extract temporal context for the call at ``call_idx`` from an ordered timeline.

    Args:
        call_idx: index into the timeline.
        timeline: ordered list of dicts, each with at least ``onset_s``, ``offset_s``,
            and optionally ``emitter``, ``daily_phase``.
        bout_gap_threshold: inter-call gap (seconds) above which a new bout starts.
    """
    if call_idx < 0 or call_idx >= len(timeline):
        return TemporalContext()

    cur = timeline[call_idx]
    ctx = TemporalContext(daily_phase=cur.get("daily_phase"))

    # Time since last call
    if call_idx > 0:
        prev = timeline[call_idx - 1]
        ctx.time_since_last_call = max(0.0, cur.get("onset_s", 0) - prev.get("offset_s", 0))
        # Turn-taking: last call from a DIFFERENT emitter
        cur_emitter = cur.get("emitter")
        for j in range(call_idx - 1, -1, -1):
            if timeline[j].get("emitter") != cur_emitter:
                ctx.turn_taking_latency = max(
                    0.0, cur.get("onset_s", 0) - timeline[j].get("offset_s", 0)
                )
                break
        # Overlap: did the previous call extend into this one?
        ctx.overlap_flag = cur.get("onset_s", 0) < prev.get("offset_s", 0)

    # Bout position + length: scan forward/backward for the bout boundary
    bout_start = call_idx
    while bout_start > 0:
        gap = timeline[bout_start].get("onset_s", 0) - timeline[bout_start - 1].get("offset_s", 0)
        if gap > bout_gap_threshold:
            break
        bout_start -= 1
    bout_end = call_idx
    while bout_end < len(timeline) - 1:
        gap = timeline[bout_end + 1].get("onset_s", 0) - timeline[bout_end].get("offset_s", 0)
        if gap > bout_gap_threshold:
            break
        bout_end += 1
    ctx.bout_position = call_idx - bout_start
    ctx.bout_length = bout_end - bout_start + 1

    return ctx
