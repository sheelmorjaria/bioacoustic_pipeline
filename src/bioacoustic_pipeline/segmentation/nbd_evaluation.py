"""Multi-resolution boundary F1 evaluation (#1 of the refined pipeline).

A single F1 is too coarse: an NBD "false positive" at the word level is often a
real PHONEME boundary the word-level GT can't see. This module reports:

  - word-level F1     (predicted vs word-start GT at word_tol)
  - combined F1       (word TP + word-FPs that match phoneme GT -> reclaimed)

and per-type precision/recall (hard / soft / transitional) when boundary types
are available. Tolerance-based matching (a predicted boundary within tol of a
GT boundary counts). Pure numpy / stdlib.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _near_any(t: float, gt: Sequence[float], tol: float) -> bool:
    if len(gt) == 0:
        return False
    return bool(np.min(np.abs(np.asarray(gt, dtype=float) - t)) < tol)


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf_at_tolerance(
    predicted: Sequence[float], gt: Sequence[float], tol: float = 50.0
) -> PRF:
    """Tolerance-matched precision/recall/F1 (a prediction within tol of a GT)."""
    tp = sum(1 for g in gt if _near_any(g, predicted, tol))
    fp = sum(1 for p in predicted if not _near_any(p, gt, tol))
    fn = len(gt) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return PRF(prec, rec, f1, tp, fp, fn)


@dataclass
class MultiResolutionF1:
    word: PRF
    combined: PRF
    n_word_fp: int
    n_reclaimed: int
    word_tol: float
    phoneme_tol: float

    def as_dict(self) -> dict:
        return {
            "word": {"f1": round(self.word.f1, 4), "precision": round(self.word.precision, 4),
                     "recall": round(self.word.recall, 4), "tp": self.word.tp,
                     "fp": self.word.fp, "fn": self.word.fn},
            "combined": {"f1": round(self.combined.f1, 4),
                         "precision": round(self.combined.precision, 4),
                         "recall": round(self.combined.recall, 4),
                         "tp": self.combined.tp, "fp": self.combined.fp,
                         "fn": self.combined.fn},
            "n_reclaimed": self.n_reclaimed,
            "n_word_fp": self.n_word_fp,
            "word_tol": self.word_tol,
            "phoneme_tol": self.phoneme_tol,
        }


def multi_resolution_f1(
    predicted: Sequence[float],
    word_gt: Sequence[float],
    phoneme_gt: Sequence[float] | None = None,
    *,
    word_tol: float = 50.0,
    phoneme_tol: float = 30.0,
) -> MultiResolutionF1:
    """Word-level F1 + combined F1 reclaiming word-FPs that match phoneme GT.

    A predicted boundary that misses every word-start but lands near a phoneme
    boundary is "reclaimed" — counted as a combined TP rather than an FP. This
    is the multi-resolution view: the NBD isn't wrong, it's finding sub-word
    structure.
    """
    word = prf_at_tolerance(predicted, word_gt, word_tol)

    word_fps = [p for p in predicted if not _near_any(p, word_gt, word_tol)]
    if phoneme_gt:
        reclaimed = [p for p in word_fps if _near_any(p, phoneme_gt, phoneme_tol)]
    else:
        reclaimed = []

    n_reclaimed = len(reclaimed)
    combined_tp = word.tp + n_reclaimed
    combined_fp = len(word_fps) - n_reclaimed
    combined_fn = word.fn  # words we still missed
    c_prec = combined_tp / max(combined_tp + combined_fp, 1)
    c_rec = combined_tp / max(combined_tp + combined_fn, 1)
    c_f1 = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-8)
    combined = PRF(c_prec, c_rec, c_f1, combined_tp, combined_fp, combined_fn)

    return MultiResolutionF1(
        word=word,
        combined=combined,
        n_word_fp=len(word_fps),
        n_reclaimed=n_reclaimed,
        word_tol=word_tol,
        phoneme_tol=phoneme_tol,
    )


def per_type_prf(
    predicted_typed: list[tuple[float, str]],
    gt: Sequence[float],
    tol: float = 50.0,
) -> dict[str, PRF]:
    """Per-boundary-type PRF (hard / soft / transitional).

    Args:
        predicted_typed: list of (time_ms, boundary_type).
    """
    out: dict[str, PRF] = {}
    types = sorted({t for _, t in predicted_typed}) or ["unknown"]
    for bt in types:
        preds = [p for p, t in predicted_typed if t == bt]
        out[bt] = prf_at_tolerance(preds, gt, tol)
    return out
