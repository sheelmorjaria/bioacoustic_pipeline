"""Acoustic-feature trajectory derivatives — gap-safe, unit-correct.

Named features with physical units, per-feature support masks, contiguous
span IDs, first derivatives, optional smoothed derivatives. Unsupported
values remain unavailable (NaN), never zero-valued acoustic evidence.

Measurement contract: the 44-D fingerprint's `combined` vector contains
heterogeneous summaries — spectral, temporal, and texture statistics.
Not all coordinates are suitable instantaneous state variables. This module
operates on a PRESPECIFIED SUBSET of well-supported local measurements,
each with declared physical units.

A derivative d(feature)/dt is a RATE OF CHANGE in an acoustic descriptor,
not a physical velocity of any production mechanism. Units are stated
per-feature (e.g., centroid → Hz/s, flatness → 1/s). Heterogeneous
derivative magnitudes are NOT combined into a single "motor speed"
without an explicit metric.

Caller-mean subtraction before ordinary differentiation is mathematically
redundant (the constant cancels in the difference) — an exact identity,
not disentanglement.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

# Version 2 = the layout verified against the live extractor (2026-10-18);
# version 1 mapped env_cv to index 18 (macro's mod_depth — identically 0 on
# 40 ms windows). Consumers must check this stamp; derivative outputs made
# under version 1 are invalid for name-indexed use.
FEATURE_SCHEMA_VERSION = 2

# Named features with physical units, from the 44-D fingerprint's combined
# vector. Only well-supported LOCAL measurements whose derivatives have
# interpretable acoustic units. Duration, envelope attack/decay (constant
# for equal-length windows), modulation statistics (require longer support
# than 40 ms at 100 Hz envelope sampling), and trajectory/texture summaries
# (GLCM, moments of the derivative distribution) are EXCLUDED.
FEATURE_UNITS = {
    "centroid": "Hz/s",
    "spread": "Hz/s",
    "rolloff": "Hz/s",
    "rms": "amplitude/s",
    "peak": "amplitude/s",
    "crest": "1/s",
    "zcr": "Hz/s",
    "entropy": "1/s",
    "flatness": "1/s",
    "flux": "1/s",
    "dom_freq": "Hz/s",
    "dyn_cv": "1/s",
}

# TRUE combined-vector layout, verified against the live extractor
# (physics 15 + macrotexture 14 + microtexture 15 = 44). Pinned by
# test_layout_pinned_to_extractor in tests/test_acoustic_derivatives.py —
# an earlier revision assumed a stale layout whose env_cv@18 actually read
# macro's mod_depth (constant 0 on 40 ms windows); the envelope CV is
# micro's dyn_cv@38.
_FEATURE_ORDER = (
    # physics (0-14)
    "dur_ms",
    "rms",
    "peak",
    "crest",
    "zcr",
    "centroid",
    "spread",
    "skewness",
    "kurtosis",
    "rolloff",
    "flatness",
    "entropy",
    "flux",
    "dom_freq",
    "atk",
    # macrotexture (15-28)
    "curvature",
    "asymmetry",
    "mod_rate",
    "mod_depth",
    "decay",
    "sustain",
    "release",
    "c_slope",
    "c_range",
    "b_slope",
    "glcm_contrast",
    "glcm_correlation",
    "glcm_energy",
    "glcm_homogeneity",
    # microtexture (29-43)
    "d_mean",
    "d_std",
    "d_skew",
    "d_kurt",
    "d_max",
    "d_range",
    "rise",
    "fall",
    "dyn_range",
    "dyn_cv",
    "env_skew",
    "ici_mean",
    "ici_cv",
    "flux_var",
    "ent_var",
)

# map named features to indices in the combined vector
_NAME_TO_IDX = {name: i for i, name in enumerate(_FEATURE_ORDER)}

# the supported subset (local measurements with interpretable derivatives)
_SUPPORTED = tuple(name for name in FEATURE_UNITS if name in _NAME_TO_IDX)


def supported_features() -> tuple:
    """Feature names whose derivatives have declared acoustic units."""
    return _SUPPORTED


def first_derivative(x: np.ndarray, valid: np.ndarray, dt: float) -> np.ndarray:
    """Backward-difference derivative, gap-safe (NaN at invalid/first).

    Both x[i] and x[i-1] must be valid for d[i] to exist. The derivative
    NEVER crosses an invalid gap (NaN, not a value). Caller-mean subtraction
    before this is redundant: the constant cancels exactly.
    """
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    d = np.full_like(x, np.nan)
    for i in range(1, len(x)):
        if valid[i] and valid[i - 1]:
            d[i] = (x[i] - x[i - 1]) / dt
    return d


def contiguous_spans(valid: np.ndarray) -> np.ndarray:
    """Span IDs for valid runs (-1 for invalid positions).

    Two adjacent valid positions get the same span ID; any invalid gap
    starts a new span. Derivatives are computed within spans only.
    """
    valid = np.asarray(valid, dtype=bool)
    spans = np.full(len(valid), -1, dtype=np.int32)
    current = -1
    for i, ok in enumerate(valid):
        if ok:
            if i == 0 or not valid[i - 1]:
                current += 1
            spans[i] = current
    return spans


def trajectory_derivatives(
    trajectory: np.ndarray,
    valid: np.ndarray,
    dt: float,
    feature_names: tuple | None = None,
) -> dict:
    """Derivatives for a named-feature subset of a fingerprint trajectory.

    Returns {"derivatives": (T, n_supported), "valid": (T,), "span_ids":
    (T,), "feature_names": tuple, "units": dict}. Unsupported features are
    excluded by the schema. Padding inertness: appended invalid positions
    never affect valid derivatives.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if feature_names is None:
        names = _SUPPORTED
    else:
        names = tuple(n for n in feature_names if n in FEATURE_UNITS and n in _NAME_TO_IDX)
    idx = [_NAME_TO_IDX[n] for n in names]
    subset = traj[:, idx]

    derivs = np.full_like(subset, np.nan)
    for j in range(len(names)):
        derivs[:, j] = first_derivative(subset[:, j], valid, dt)

    span_ids = contiguous_spans(valid)
    units = {n: FEATURE_UNITS[n] for n in names}
    return {
        "derivatives": derivs,
        "valid": valid,
        "span_ids": span_ids,
        "feature_names": names,
        "units": units,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
