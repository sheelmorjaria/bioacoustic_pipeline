"""Spatial-routing metadata for playback (unicast/broadcast, directionality).

Phase 1 is pure geometry + manifest labels: VBAP-style gains for a circular speaker array
and a ``spatial_condition`` tag. It does NOT render a multi-speaker signal — it tags the
manifest with the intended spatial treatment so the spatial-mismatch hypothesis is explicit
and reproducible. Pure stdlib.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

# Spatial conditions used in the playback manifest.
CORRECT_DIRECTION = "correct_direction"
WRONG_DIRECTION = "wrong_direction"
BROADCAST = "broadcast"
DIRECTIONAL_UNICAST = "directional_unicast"
UNICAST_FROM_WRONG = "unicast_from_wrong_direction"
BROADCAST_FROM_UNICAST_SIGNAL = "broadcast_of_unicast_signal"
SPATIAL_CONDITIONS = (
    CORRECT_DIRECTION,
    WRONG_DIRECTION,
    BROADCAST,
    DIRECTIONAL_UNICAST,
    UNICAST_FROM_WRONG,
    BROADCAST_FROM_UNICAST_SIGNAL,
)


def vbap_gains(source_azimuth_deg: float, speaker_azimuths_deg) -> dict[int, float]:
    """Vector-Base Amplitude Panning gains for a source angle on a circular speaker array.

    Returns ``{speaker_index: gain}`` with energy spread over the (at most) two speakers
    bracketing the source angle. Gains are non-negative and normalized so the active pair
    sums to ~1 (constant-power-ish panning).
    """
    spk = np.asarray(sorted(speaker_azimuths_deg), dtype=np.float64) % 360.0
    src = float(source_azimuth_deg) % 360.0
    if spk.size == 0:
        return {}
    if spk.size == 1:
        return {0: 1.0}
    # angular distances source->each speaker; the two smallest bracket the source
    d = (spk - src + 360.0) % 360.0
    order = np.argsort(np.minimum(d, 360.0 - d))
    i0, i1 = int(order[0]), int(order[1])
    # weight by inverse angular distance
    w0 = 1.0 / (min(d[i0], 360.0 - d[i0]) + 1e-6)
    w1 = 1.0 / (min(d[i1], 360.0 - d[i1]) + 1e-6)
    s = w0 + w1
    return {int(i0): float(w0 / s), int(i1): float(w1 / s)}


def spatial_condition(direction_match: bool, signal_type: str) -> str:
    """Classify a presentation cell into a spatial_condition label.

    Args:
        direction_match: did playback come from the (apparent) emitter's direction?
        signal_type: "unicast" (directed) or "broadcast" (omnidirectional) signal content.
    """
    if signal_type == "unicast":
        return CORRECT_DIRECTION if direction_match else UNICAST_FROM_WRONG
    if signal_type == "broadcast":
        return BROADCAST if direction_match else BROADCAST_FROM_UNICAST_SIGNAL
    return WRONG_DIRECTION
