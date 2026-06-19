"""Audio IO utilities: load, mono-mix, resample, normalize.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf


def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono; optionally resample to target_sr."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        from scipy import signal  # noqa: PLC0415

        audio = signal.resample(audio, int(len(audio) * target_sr / sr))
        sr = target_sr
    return np.asarray(audio, dtype=np.float32), sr


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]."""
    peak = float(np.abs(audio).max())
    return audio / peak if peak > 0 else audio
