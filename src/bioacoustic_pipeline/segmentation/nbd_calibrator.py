"""Calibrated boundary refiner + ABSTAIN state (#1 of the refined pipeline).

Turns NBD proposals (``nbd_proposal.Proposal``) into calibrated boundaries:
features -> P(boundary) via a lightweight logistic-regression calibrator
trained against forced-alignment word boundaries (humans) — the one species
with boundary labels. Never transfer these labels to animals.

The refiner replaces the fixed dual-threshold with a probability, and adds the
ABSTAIN state the spec calls for:
  P >= keep_threshold     -> emit (hard / soft / transitional boundary)
  abstain <= P < keep      -> ABSTAIN (uncertain; could be phoneme or noise)
  P < abstain_threshold    -> drop

sklearn-only. Checkpoint = coef/intercept + scaler stats (numpy), no pickle of
arbitrary objects.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.segmentation.nbd_proposal import Proposal

# Ordered feature schema. detector -> two one-hot dims (is_energy, is_spectral);
# "both" sets both. duration is log-scaled.
FEATURE_NAMES = (
    "energy_change",
    "spectral_flux",
    "band_flux",
    "local_rms",
    "rms_slope",
    "log_duration_since_prev_ms",
    "is_energy_detector",
    "is_spectral_detector",
)


def proposal_features(proposal: Proposal) -> np.ndarray:
    """Map one Proposal to its feature vector (matches FEATURE_NAMES)."""
    det = proposal.detector
    return np.array(
        [
            proposal.energy_change,
            proposal.spectral_flux,
            proposal.band_flux,
            proposal.local_rms,
            proposal.rms_slope,
            float(np.log1p(max(proposal.duration_since_prev_ms, 0.0))),
            1.0 if det in ("energy", "both") else 0.0,
            1.0 if det in ("spectral", "both") else 0.0,
        ],
        dtype=np.float64,
    )


def proposals_to_matrix(proposals: list[Proposal]) -> np.ndarray:
    """Stack proposal feature vectors -> (n_proposals, n_features)."""
    if not proposals:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([proposal_features(p) for p in proposals])


def label_proposals(
    proposals: list[Proposal], gt_ms: list[float], tol_ms: float = 50.0
) -> np.ndarray:
    """y[i] = 1 if proposal i is within tol_ms of any ground-truth boundary."""
    gt = np.asarray(gt_ms, dtype=np.float64)
    y = np.zeros(len(proposals), dtype=np.int64)
    if len(gt) == 0:
        return y
    for i, p in enumerate(proposals):
        if np.min(np.abs(gt - p.time_ms)) < tol_ms:
            y[i] = 1
    return y


def _boundary_type(detector: str) -> str:
    if detector == "energy":
        return "hard"
    if detector == "spectral":
        return "soft"
    return "transitional"


@dataclass
class CalibratedBoundary:
    """A proposal with a calibrated confidence + boundary type."""

    time_ms: float
    confidence: float
    boundary_type: str  # hard | soft | transitional
    proposal: Proposal


class BoundaryCalibrator:
    """Logistic-regression calibrator over NBD proposal features."""

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self.feature_names: tuple[str, ...] = FEATURE_NAMES

    # ---- training ----
    def fit(self, proposals: list[Proposal], y: np.ndarray) -> "BoundaryCalibrator":
        from sklearn.linear_model import LogisticRegression

        X = proposals_to_matrix(proposals)
        if X.shape[0] != len(y):
            raise ValueError("proposals and y length mismatch")
        if len(set(y.tolist())) < 2:
            # Degenerate (all one class): still set normalization so predict runs.
            self._mean = X.mean(axis=0) if len(X) else np.zeros(len(FEATURE_NAMES))
            self._std = X.std(axis=0) if len(X) else np.ones(len(FEATURE_NAMES))
            self._std[self._std < 1e-8] = 1.0
            self._coef = np.zeros(len(FEATURE_NAMES))
            self._intercept = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
            return self
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        Xs = (X - self._mean) / self._std
        # class_weight='balanced': proposals are heavily class-imbalanced (word
        # boundaries are a minority of all candidates). Without it the model
        # under-predicts and the keep threshold keeps nothing.
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 class_weight="balanced")
        clf.fit(Xs, y)
        self._coef = clf.coef_[0]
        self._intercept = float(clf.intercept_[0])
        return self

    # ---- inference ----
    def predict_proba(self, proposals: list[Proposal]) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("calibrator is not fitted")
        X = proposals_to_matrix(proposals)
        Xs = (X - self._mean) / self._std
        logits = Xs @ self._coef + self._intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def apply(
        self,
        proposals: list[Proposal],
        *,
        keep_threshold: float = 0.5,
        abstain_threshold: float = 0.3,
    ) -> tuple[list[CalibratedBoundary], list[CalibratedBoundary]]:
        """Gate proposals into (kept, abstained). Below abstain_threshold dropped."""
        if not (0.0 <= abstain_threshold <= keep_threshold <= 1.0):
            raise ValueError("require 0 <= abstain_threshold <= keep_threshold <= 1")
        probs = self.predict_proba(proposals)
        kept: list[CalibratedBoundary] = []
        abstained: list[CalibratedBoundary] = []
        for p, prob in zip(proposals, probs):
            cb = CalibratedBoundary(
                time_ms=p.time_ms,
                confidence=float(prob),
                boundary_type=_boundary_type(p.detector),
                proposal=p,
            )
            if prob >= keep_threshold:
                kept.append(cb)
            elif prob >= abstain_threshold:
                abstained.append(cb)
        return kept, abstained

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "mean": self._mean,
                    "std": self._std,
                    "coef": self._coef,
                    "intercept": self._intercept,
                    "feature_names": self.feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryCalibrator":
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301 — trusted local checkpoint
        obj = cls()
        obj._mean = state["mean"]
        obj._std = state["std"]
        obj._coef = state["coef"]
        obj._intercept = state["intercept"]
        obj.feature_names = state["feature_names"]
        return obj

    @property
    def fitted(self) -> bool:
        return self._coef is not None
