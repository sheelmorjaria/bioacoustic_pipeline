"""Synthetic-audited 48-to-16 kHz candidate with explicit source support.

No delay compensation or centred filtering. Initial zero-padded filter states
are marked unsupported, never silently counted as observed acoustic samples.
This is not integrated into the frozen RTPE frontend.
"""

import numpy as np
from scipy.signal import firwin, lfilter


def causal_decimate(waveform, sample_rate):
    x = np.asarray(waveform)
    if sample_rate != 48000 or x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty mono 48 kHz waveform required")
    taps = firwin(97, 7200, window=("kaiser", 8.6), fs=48000)
    values = lfilter(taps, [1.0], x.astype(np.float64))[::3]
    ends = np.arange(0, len(x), 3, dtype=np.int64)
    return dict(
        values=values,
        valid=ends >= 96,
        earliest_source_sample=ends - 96,
        latest_source_sample=ends,
        taps=taps,
        output_rate=16000,
        delay_compensated=False,
    )


def frame_source_support(frame):
    if type(frame) is not int or frame < 0:
        raise ValueError("Nonnegative integer frame index required")
    return 3 * (frame * 160) - 96, 3 * (frame * 160 + 639)


def target_separation(origin, horizon):
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Positive integer forecast offset required")
    _, last = frame_source_support(origin)
    first, end = frame_source_support(origin + horizon)
    gap = first - (last + 1)
    return dict(
        origin_frame=origin,
        horizon_frames=horizon,
        input_latest_source_sample=last,
        target_earliest_source_sample=first,
        target_latest_source_sample=end,
        gap_source_samples=gap,
        gap_milliseconds=1000 * gap / 48000,
        disjoint=first > last,
    )
