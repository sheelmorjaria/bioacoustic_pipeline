"""Taxonomy 4-arm staged comparison — Arm 1: existing crop, leakage-free.

Arm 1 is the versioned baseline: the existing 44-D extractor on blind crops,
StandardScaler fit on TRAIN ONLY, linear classifier, macro-F1 with the
chance level stated. Arms 2-4 (annotated-event, automatic extraction,
relative-feature/RQA) stack on this contract, never replacing it.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def run_arm1_existing_crop(X_train, y_train, X_test, y_test) -> dict:
    """Train-only scaling + linear classifier -> macro-F1 (+ chance level)."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    scaler = StandardScaler().fit(X_train)  # train stats ONLY
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)
    pred = clf.predict(scaler.transform(X_test))
    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "chance": 1.0 / len(set(y_train)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "arm": "arm1_existing_crop_train_scaled",
    }


def _fingerprint_of(audio: np.ndarray, sr: int) -> np.ndarray:
    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415

    return np.asarray(afp.extract(audio, sr, species="bat").combined, dtype=np.float64)


def run_arm2_annotated_event(path_or_audio, annotation, *, sr=16000):
    """Arm 2: extract the 44-D fingerprint from an ANNOTATED interval.

    ``annotation``: {"start_ms", "end_ms"}. Accepts a path or an
    (audio, sr) tuple — the annotation defines the slice, never a blind crop.
    """
    if annotation is None:
        raise ValueError("arm 2 requires an annotation interval")
    if isinstance(path_or_audio, str):
        from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

        audio, sr = load_audio(path_or_audio, target_sr=sr)
    else:
        audio, sr = path_or_audio, sr
    start = int(annotation["start_ms"] * sr / 1000.0)
    end = max(start + 1, int(annotation["end_ms"] * sr / 1000.0))
    seg = np.asarray(audio)[start:end]
    return {
        "fingerprint": _fingerprint_of(seg, sr),
        "source": "annotated_interval",
        "start_ms": float(annotation["start_ms"]),
        "end_ms": float(annotation["end_ms"]),
    }


def detect_energy_event(audio: np.ndarray, sr: int, *, window_ms: float = 500.0):
    """LABEL-BLIND event detector: the max-RMS contiguous window.

    Sees samples only — the taxon label never enters this path (pinned by a
    signature test). Returns (start_sample, end_sample) or None if the
    recording is silent throughout.
    """
    audio = np.asarray(audio)
    win = max(int(sr * window_ms / 1000.0), 1)
    if len(audio) < win:
        return None
    rms = np.convolve(audio.astype(np.float64) ** 2, np.ones(win), "valid") / win
    if rms.max() <= 1e-10:
        return None
    start = int(np.argmax(rms))
    return start, start + win


def run_arm3_automatic_event(audio, sr):
    """Arm 3: label-blind energy-gated event -> 44-D fingerprint."""
    ev = detect_energy_event(audio, sr)
    if ev is None:
        return {"event_found": False, "source": "automatic_energy_gate"}
    start, end = ev
    return {
        "event_found": True,
        "fingerprint": _fingerprint_of(np.asarray(audio)[start:end], sr),
        "source": "automatic_energy_gate",
        "start_ms": round(1000.0 * start / sr, 2),
        "end_ms": round(1000.0 * end / sr, 2),
    }
