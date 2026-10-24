"""Donor selection for Track-C caller-swapped playback stimuli (§46 Ph3).

Takes the ΔL_CPC mismatch landscape from Phase 2 plus the corpus annotations
and emits the spliced-stimulus candidate list: where the donor's translation
fails (spikes), which call types qualify (coarse control), and which donor
units are acoustically matched to the target (duration + fingerprint cosine
— the 4-stage experimental controls). Stdlib csv only.

Ceiling-bound: this is playback-stimulus PREPARATION, not semantics — the
stimuli mean what the Track-C biology says they mean, no more.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DonorCandidate:
    """One candidate unit for caller-swapped stimulus construction."""

    call_type: str
    duration_ms: float
    fingerprint: np.ndarray
    donor_id: int
    start_ms: float | None = None
    file_name: str | None = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def select_donor_candidates(
    target: DonorCandidate,
    pool: list[DonorCandidate],
    *,
    call_type: str | None = None,
    duration_tolerance_ms: float = 50.0,
    min_cosine: float = 0.9,
) -> list[DonorCandidate]:
    """Coarse call-type filter, then duration + fingerprint acoustic match.

    Stage order mirrors the experimental protocol: call type (behavioral
    class) first, then duration within ``duration_tolerance_ms`` of the
    target, then fingerprint cosine >= ``min_cosine`` — donors must be
    acoustically comparable for a caller swap to isolate identity.
    """
    wanted = call_type if call_type is not None else target.call_type
    out = []
    for c in pool:
        if c.call_type != wanted:
            continue
        if abs(c.duration_ms - target.duration_ms) > duration_tolerance_ms:
            continue
        if _cosine(c.fingerprint, target.fingerprint) < min_cosine:
            continue
        out.append(c)
    return out


def detect_mismatch_spikes(
    delta_series: np.ndarray,
    *,
    hop_ms: float,
    top_n: int = 10,
    donor_id: int | None = None,
    recipient_id: int | None = None,
    out_path: str | Path | None = None,
) -> list[dict]:
    """Top-N mismatch windows (largest ΔL first), optionally as CSV.

    Simple top-k over the per-frame ΔL landscape — deliberately not
    ``find_peaks``: for stimulus selection the k worst windows are the
    protocol, regardless of local peak shape.
    """
    delta = np.asarray(delta_series, dtype=np.float64)
    k = min(top_n, len(delta))
    order = np.argsort(-delta)[:k]
    rows = [
        {
            "start_ms": round(float(i) * hop_ms, 1),
            "end_ms": round((float(i) + 1) * hop_ms, 1),
            "delta_L": round(float(delta[i]), 4),
            "donor_id": donor_id if donor_id is not None else -1,
            "recipient_id": recipient_id if recipient_id is not None else -1,
        }
        for i in sorted(order, key=lambda i: -delta[i])
    ]
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["start_ms", "end_ms", "delta_L", "donor_id", "recipient_id"]
            )
            w.writeheader()
            w.writerows(rows)
    return rows
