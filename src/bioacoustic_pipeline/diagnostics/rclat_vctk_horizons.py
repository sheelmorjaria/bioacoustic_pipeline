"""Prospective retained-horizon support and loss contracts; no model invocation."""

import numpy as np
from ..representation.rclat_causal_resample import frame_source_support, target_separation


def eligible_origins(source_samples):
    """Timing-only upper bound; actual waveform/feature validity still required."""
    if type(source_samples) is not int or source_samples < 0:
        raise ValueError("Nonnegative integer source length required")
    outputs = (source_samples + 2) // 3
    frames = max(0, (outputs - 640) // 160 + 1)
    origins = np.arange(32, max(32, frames - 16), dtype=np.int64)
    for origin in origins:
        # 32-frame history starts at origin-31. No implicit filter padding.
        assert frame_source_support(int(origin) - 31)[0] >= 0
        assert all(target_separation(int(origin), h)["disjoint"] for h in [8, 16])
    return origins


def retained_losses(prediction, target, observed):
    p = np.asarray(prediction)
    y = np.asarray(target)
    m = np.asarray(observed)
    if (
        p.ndim != 3
        or p.shape != y.shape
        or m.shape != y.shape
        or p.shape[1:] != (3, 64)
        or m.dtype != bool
        or len(p) == 0
    ):
        raise ValueError("Matching nonempty B,3,64 forecasts/targets and boolean support required")
    result = {}
    for h, column in [(8, 1), (16, 2)]:
        support = m[:, column]
        counts = support.sum(-1)
        if (
            not (counts > 0).all()
            or not np.isfinite(np.where(support, p[:, column], 0)).all()
            or not np.isfinite(np.where(support, y[:, column], 0)).all()
        ):
            raise ValueError(
                "Retained horizon requires finite observed cells and support per origin"
            )
        delta = np.zeros_like(y[:, column], dtype=np.float64)
        np.subtract(p[:, column], y[:, column], out=delta, where=support)
        result[h] = (delta**2).sum(-1) / counts
    return result
