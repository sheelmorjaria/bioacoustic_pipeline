"""Spectrogram quality flags — detect degenerate segments before they contaminate tokens.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import numpy as np


def spectrogram_quality(magnitude: np.ndarray, waveform: np.ndarray | None = None) -> dict:
    """Quality flags from a magnitude spectrogram (+ optional raw waveform for clipping).

    Returns: {low_energy, clipping, silence, dynamic_range_db}.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    total_energy = float((mag ** 2).sum())
    max_bin = float(mag.max())
    q = {
        "low_energy": total_energy < 1e-6,
        "clipping": False,
        "silence": max_bin < 1e-4,
        "dynamic_range_db": 0.0,
    }
    if max_bin > 1e-10:
        min_active = float(mag[mag > max_bin * 1e-4].min()) if np.any(mag > max_bin * 1e-4) else max_bin * 1e-4
        q["dynamic_range_db"] = float(20 * np.log10(max_bin / max(min_active, 1e-10)))
    if waveform is not None:
        w = np.asarray(waveform)
        q["clipping"] = bool(np.mean(np.abs(w) > 0.99) > 0.01)
    return q
