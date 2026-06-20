"""Declared behavioral-response schema for the live (Track C) validation side.

The pipeline does NOT collect these measurements — this is the schema a playback experiment
fills in to test the manifest's ``predicted_behavior``. Pure declaration; keeps the
computational side and the wet-lab side agreed on what "response" means.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

# Canonical response measures a playback experiment records per stimulus presentation.
RESPONSE_MEASURES = (
    "orientation_to_speaker",
    "reply_latency_s",
    "reply_probability",
    "movement_change",
    "approach_avoidance",
    "callback_token_distribution",
    "social_spacing_change",
)


def empty_response_record() -> dict:
    """A blank response record (all measures None) for a single stimulus presentation."""
    return {m: None for m in RESPONSE_MEASURES}
