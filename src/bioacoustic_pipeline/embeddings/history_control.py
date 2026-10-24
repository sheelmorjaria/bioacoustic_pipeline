"""Suffix-preserving history controls — §46 bat temporal track.

The causal test the ΔL landscape cannot make: does the caller head gain MORE
from the true early history than the population head does? That requires
holding the immediate suffix (which covers the AR receptive field) and the
prediction target FIXED while replacing only the early history — then
recomputing the AR context on the control sequence. Modes:

- ``reverse``: time-reverse the early segment (same marginal spectrum,
  destroyed temporal order)
- ``donor``: a matched segment from another bout (same caller+context;
  caller selects by fingerprint+duration match)
- ``shuffle``: frame-block shuffle within the early segment (destroys
  long-range order, preserves local texture)

A short crossfade blends the junction so the AR never sees an artificial
discontinuity — a naive splice would hand the model a fabricated "event"
and fake a history effect.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def crossfade_splice(early: np.ndarray, suffix: np.ndarray, *, blend: int = 12):
    """Concatenate with a linear crossfade across the junction.

    The blend region covers the last ``blend`` samples of ``early`` and the
    first ``blend`` of ``suffix``: early fades out (1->0) while suffix fades
    in (0->1). Beyond the blend, outputs are byte-identical to the inputs;
    inside it they mix linearly, so the junction never carries a step.
    """
    early = np.asarray(early)
    suffix = np.asarray(suffix)
    blend = int(min(max(blend, 0), len(early), len(suffix)))
    out = np.concatenate([early, suffix]).astype(np.float64)
    if blend == 0:
        return out.astype(early.dtype)
    i0 = len(early)
    w = (np.arange(1, blend + 1) / (blend + 1)).astype(np.float64)
    # broadcast over trailing dims (frame vectors are (n, feat), samples (n,))
    wb = w.reshape(-1, *([1] * (early.ndim - 1)))
    # taper: early's last `blend` entries fade out while cross-fading toward
    # the aligned suffix entries; suffix beyond the blend stays byte-identical
    out[i0 - blend : i0] = (1.0 - wb) * early[-blend:].astype(np.float64) + wb * suffix[
        :blend
    ].astype(np.float64)
    return out.astype(early.dtype)


def build_control_history(
    seq: np.ndarray,
    *,
    n_early: int,
    mode: str = "reverse",
    donor: np.ndarray | None = None,
    blend: int = 12,
    seed: int = 0,
) -> np.ndarray:
    """[H'_early, H_suffix] with the suffix byte-identical beyond the blend.

    ``n_early`` samples of history are replaced per ``mode``; the remainder
    (the suffix the AR receptive field reads) is preserved exactly. The
    returned array has the SAME length as ``seq``.
    """
    if mode not in ("reverse", "donor", "shuffle"):
        raise ValueError(f"unknown mode {mode!r}")
    seq = np.asarray(seq)
    if n_early < 8 or len(seq) - n_early < 8:
        raise ValueError("need >=8 samples of both history and suffix")
    early, suffix = seq[:n_early], seq[n_early:]
    if mode == "reverse":
        replacement = early[::-1].copy()
    elif mode == "donor":
        if donor is None:
            raise ValueError("donor mode requires a donor segment")
        donor = np.asarray(donor)
        if len(donor) < n_early:
            raise ValueError("donor shorter than the history it replaces")
        replacement = donor[:n_early].copy()
    else:  # shuffle: permute fixed blocks, preserving local texture
        rng = np.random.RandomState(seed)
        block = max(1, n_early // 16)
        n_blocks = n_early // block
        order = rng.permutation(n_blocks)
        replacement = np.concatenate([early[b * block : (b + 1) * block] for b in order])
    if blend > 0:
        return crossfade_splice(replacement, suffix, blend=blend)
    return np.concatenate([replacement, suffix])
