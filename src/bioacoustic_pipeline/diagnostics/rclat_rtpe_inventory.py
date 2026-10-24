"""Value-independent bounded RTPE origins and source-local mel verification."""

from collections import Counter
import hashlib

import numpy as np


def eligible_origins(valid, *, history, horizons):
    horizons = tuple(horizons)
    if (
        valid.ndim != 2
        or valid.dtype != np.dtype(bool)
        or valid.shape[1] == 0
        or type(history) is not int
        or history <= 0
        or not horizons
        or any(type(h) is not int or h < 4 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError(
            "Boolean frame/feature validity, positive history and nonoverlapping horizons required"
        )
    width = history + max(horizons)
    if len(valid) < width:
        return np.array([], dtype=np.int64)
    gaps = np.r_[0, np.cumsum(~valid.any(1))]
    starts = np.flatnonzero(gaps[width:] - gaps[:-width] == 0)
    return (starts + history - 1).astype(np.int64)


def select_origins(uid, candidates, *, count, salt):
    a = np.asarray(candidates)
    if (
        a.ndim != 1
        or a.dtype.kind not in "iu"
        or (a < 0).any()
        or len(np.unique(a)) != len(a)
        or type(count) is not int
        or count <= 0
        or len(a) < count
        or not isinstance(uid, str)
        or not uid
        or not isinstance(salt, str)
        or not salt
    ):
        raise ValueError("Unique eligible origins and a supported fixed sample count required")
    ranked = sorted(
        a.tolist(), key=lambda t: (hashlib.sha256(f"{salt}:{uid}:{t}".encode()).digest(), t)
    )
    return sorted(ranked[:count])


def trailing_mel(audio):
    """Recalculate only the 64-D mel branch; never normalise a whole recording.

    Same fixed 16kHz/640sample/160hop schema as the frozen acoustic frontend.
    Physical features are deliberately not recalculated or used.
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1 or len(audio) < 640 or not np.isfinite(audio).all():
        raise ValueError("Finite native mono waveform required")
    frames = np.lib.stride_tricks.sliding_window_view(audio, 640)[::160]
    power = np.abs(np.fft.rfft(frames * np.hanning(640), axis=1)) ** 2
    edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 66) / 2595) - 1)
    hz = np.fft.rfftfreq(640, 1 / 16000)
    filters = []
    for a, b, c in zip(edges[:-2], edges[1:-1], edges[2:]):
        f = np.maximum(0, np.minimum((hz - a) / (b - a), (c - hz) / (c - b)))
        filters.append(f / max(f.sum(), 1e-20))
    return np.log(power @ np.array(filters).T + 1e-10).astype(np.float32)


def recording_weights(rows):
    if not rows or len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Unique nonempty recording inventory required")
    n = Counter(r["speaker"] for r in rows)
    return {r["uid"]: 1 / (len(n) * n[r["speaker"]]) for r in rows}
