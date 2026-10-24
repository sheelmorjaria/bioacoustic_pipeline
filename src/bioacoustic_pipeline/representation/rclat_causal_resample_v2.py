"""Separately versioned causal FIR candidate; frozen RTPE frontend untouched."""

import numpy as np
from scipy.signal import firwin, lfilter
from .rclat_ctc import acoustic_features


def causal_decimate_v2(waveform, sample_rate):
    x = np.asarray(waveform)
    if sample_rate != 48000 or x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty mono 48 kHz source required")
    taps = firwin(193, 7000, window=("kaiser", 8.6), fs=48000)
    endpoints = np.arange(0, len(x), 3, dtype=np.int64)
    return dict(
        values=lfilter(taps, [1.0], x.astype(np.float64))[::3],
        valid=endpoints >= 192,
        earliest_source_sample=endpoints - 192,
        latest_source_sample=endpoints,
        taps=taps,
        output_rate=16000,
        delay_compensated=False,
    )


def frame_source_support_v2(frame):
    if type(frame) is not int or frame < 0:
        raise ValueError("Nonnegative integer frame required")
    return 3 * frame * 160 - 192, 3 * (frame * 160 + 639)


def target_separation_v2(origin, horizon):
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Positive horizon required")
    _, end = frame_source_support_v2(origin)
    start, last = frame_source_support_v2(origin + horizon)
    gap = start - end - 1
    return dict(
        origin=origin,
        horizon_frames=horizon,
        input_latest_source_sample=end,
        target_earliest_source_sample=start,
        target_latest_source_sample=last,
        gap_source_samples=gap,
        gap_ms=1000 * gap / 48000,
        disjoint=start > end,
    )


def mel_features_v2(waveform, sample_rate):
    decimated = causal_decimate_v2(waveform, sample_rate)
    features = acoustic_features(decimated["values"], 16000)
    frames = np.arange(len(features["mel"]), dtype=np.int64)
    first = 3 * frames * 160 - 192
    last = 3 * (frames * 160 + 639)
    support = (first >= 0)[:, None]
    return dict(
        mel=features["mel"],
        mel_valid=features["mel_valid"] & support,
        source_first_samples=first,
        source_last_samples=last,
    )
