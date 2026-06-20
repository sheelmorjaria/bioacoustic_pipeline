"""Affect-trajectory DESCRIPTOR for playback (manifest-level, NOT an audio renderer).

Records the natural per-token affect trajectory from ``DualStreamSegment.affect_mu`` where a
species affect model exists, and emits ``natural / flattened / exaggerated / reversed`` as
**target trajectories** — hypotheses the playback experiment would test (e.g. "does the
animal respond differently when the escalation contour is flattened?").

This is deliberately NOT audio rendering. Rendering affect-shifted audio requires a trained
vocoder / DDSP decoder (Phase 2/3), which does not exist yet. Where no species affect model
exists, the descriptor reports ``not_available_no_species_affect_model`` rather than
fabricating a trajectory. The affect stream is a **graded acoustic-state candidate**, not
validated affect (Track C only) — mirroring the Semantic Ceiling invariant.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

NOT_AVAILABLE = "not_available_no_species_affect_model"


def _has_affect(segments) -> bool:
    """True iff at least one segment carries a non-empty affect_mu vector."""
    for seg in segments:
        mu = getattr(seg, "affect_mu", None)
        if mu is not None and np.asarray(mu).size > 0 and np.any(np.asarray(mu) != 0):
            return True
    return False


def affect_trajectory(segments) -> list[float]:
    """Natural per-token affect intensity trajectory (||affect_mu|| per segment).

    Returns one float per segment in order; empty list if no affect model is available.
    """
    if not _has_affect(segments):
        return []
    out = []
    for seg in segments:
        mu = getattr(seg, "affect_mu", None)
        out.append(
            float(np.linalg.norm(np.asarray(mu, dtype=np.float64))) if mu is not None else 0.0
        )
    return out


def morph_descriptors(segments) -> dict:
    """Manifest-level affect-trajectory hypotheses (target curves, not rendered audio).

    Returns ``{available, natural, flattened, exaggerated, reversed}`` where each trajectory
    is a list of target per-token intensities. When no species affect model exists,
    ``available`` is False and the trajectories are replaced by the ``NOT_AVAILABLE`` marker.
    """
    natural = affect_trajectory(segments)
    if not natural:
        return {
            "available": False,
            "note": NOT_AVAILABLE,
            "natural": NOT_AVAILABLE,
            "flattened": NOT_AVAILABLE,
            "exaggerated": NOT_AVAILABLE,
            "reversed": NOT_AVAILABLE,
        }
    arr = np.asarray(natural, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    return {
        "available": True,
        "natural": [round(float(x), 4) for x in natural],
        # flattened: collapse the contour to its mean (tests if the escalation matters)
        "flattened": [round(mean, 4)] * len(natural),
        # exaggerated: amplify deviations from the mean (tests graded-state sensitivity)
        "exaggerated": [round(float(mean + 1.5 * (x - mean)), 4) for x in natural],
        # reversed: invert the contour direction (tests trajectory directionality)
        "reversed": [round(float(x), 4) for x in reversed(natural)],
    }
