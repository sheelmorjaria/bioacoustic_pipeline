"""Channel audit — field-recording quality assessment (Step 9).

Computes per-recording acoustic-channel quality: SNR (signal vs noise floor), reverb proxy
(tail energy decay), and spectral attenuation (high-frequency loss). These are
species-agnostic DSP metrics for flagging degraded recordings before they enter the pipeline.

Pure numpy. Firewall-clean (operates on raw audio). NOT meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def audit_channel(audio: np.ndarray, sr: int, frame_ms: float = 25.0) -> dict:
    """Assess the acoustic channel quality of a recording.

    Returns: snr_db, reverb_proxy, spectral_attenuation, quality_tier (clean/moderate/degraded).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.size < int(0.05 * sr):
        return {
            "snr_db": 0.0,
            "reverb_proxy": 0.0,
            "spectral_attenuation": 0.0,
            "quality_tier": "degraded",
        }
    # SNR: frame-level RMS, top 50% (signal) vs bottom 25% (noise floor)
    frame = max(int(sr * frame_ms / 1000), 64)
    hop = frame // 2
    rms_frames = []
    for i in range(0, max(1, len(x) - frame), hop):
        rms_frames.append(float(np.sqrt(np.mean(x[i : i + frame] ** 2))))
    rms_arr = np.sort(np.asarray(rms_frames))
    if len(rms_arr) < 4:
        return {
            "snr_db": 0.0,
            "reverb_proxy": 0.0,
            "spectral_attenuation": 0.0,
            "quality_tier": "degraded",
        }
    signal_rms = float(np.median(rms_arr[len(rms_arr) // 2 :]))  # top 50%
    noise_rms = float(np.median(rms_arr[: max(1, len(rms_arr) // 4)]))  # bottom 25%
    snr_db = 10.0 * np.log10((signal_rms**2 + 1e-12) / (noise_rms**2 + 1e-12))

    # Reverb proxy: energy decay in the final 100ms relative to the body
    tail_start = max(0, len(x) - int(0.1 * sr))
    tail_energy = float(np.sqrt(np.mean(x[tail_start:] ** 2)))
    body_energy = float(np.sqrt(np.mean(x[:tail_start] ** 2))) if tail_start > 0 else signal_rms
    reverb_proxy = tail_energy / (body_energy + 1e-12)

    # Spectral attenuation: fraction of spectral energy above sr/4
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    hf_mask = freqs >= sr / 4
    spectral_attenuation = float(mag[hf_mask].sum() / (mag.sum() + 1e-12))

    if snr_db > 20 and spectral_attenuation > 0.15:
        tier = "clean"
    elif snr_db > 10:
        tier = "moderate"
    else:
        tier = "degraded"

    return {
        "snr_db": round(snr_db, 1),
        "reverb_proxy": round(reverb_proxy, 3),
        "spectral_attenuation": round(spectral_attenuation, 3),
        "quality_tier": tier,
    }
