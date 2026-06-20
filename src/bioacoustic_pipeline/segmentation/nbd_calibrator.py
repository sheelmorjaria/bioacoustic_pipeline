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
# "both" sets both. duration/contrast ratios are log-scaled. The local-context
# descriptors (rms_ratio, flux/energy_contrast, flux_asym, duration_to_next) carry
# the boundary's profile — without them the point features give AUC≈0.6.
FEATURE_NAMES = (
    "energy_change",
    "spectral_flux",
    "band_flux",
    "local_rms",
    "rms_slope",
    "log_duration_since_prev_ms",
    "is_energy_detector",
    "is_spectral_detector",
    "log_rms_ratio",
    "log_flux_contrast",
    "log_energy_contrast",
    "flux_asym",
    "log_duration_to_next_ms",
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
            float(np.log1p(max(proposal.rms_ratio, 0.0))),
            float(np.log1p(max(proposal.flux_contrast, 0.0))),
            float(np.log1p(max(proposal.energy_contrast, 0.0))),
            float(proposal.flux_asym),
            float(np.log1p(max(proposal.duration_to_next_ms, 0.0))),
        ],
        dtype=np.float64,
    )


def proposals_to_matrix(proposals: list[Proposal]) -> np.ndarray:
    """Stack proposal feature vectors -> (n_proposals, n_features)."""
    if not proposals:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([proposal_features(p) for p in proposals])


def proposal_context(
    proposals: list[Proposal],
    frames: np.ndarray,
    *,
    hop_ms: float = 10.0,
    window: int = 5,
) -> np.ndarray:
    """Learned-context features: for each proposal, the mean embedding just-before
    and just-after the boundary plus their difference, from a per-frame embedding
    array (e.g. ``CPCBackbone.embed_frames``).

    Returns ``(n_proposals, 3 * embed_dim)`` — pre || post || (post - pre). Frames
    are indexed at ``hop_ms`` (10 ms for CPC), matching the NBD proposal timescale.
    Empty windows (boundary at the very start/end) zero-fill.
    """
    frames = np.asarray(frames, dtype=np.float64)
    n = len(proposals)
    if n == 0 or frames.ndim != 2 or frames.shape[0] < 2:
        return np.zeros((n, 3 * (frames.shape[1] if frames.ndim == 2 else 0)), dtype=np.float64)
    dim = frames.shape[1]
    out = np.zeros((n, 3 * dim), dtype=np.float64)
    for i, p in enumerate(proposals):
        f = int(round(p.time_ms / hop_ms))
        lo, aft, hi = max(0, f - window), f + 1, f + 1 + window
        pre = frames[lo:f].mean(axis=0) if f > lo else np.zeros(dim)
        post = frames[aft:hi].mean(axis=0) if hi > aft and aft < frames.shape[0] else np.zeros(dim)
        out[i] = np.concatenate([pre, post, post - pre])
    return out


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
        # Tuned keep_threshold (set by refit_nbd_calibrator); apply() falls back to
        # it when the caller doesn't pass one. Defaults to 0.5 for old checkpoints.
        self.keep_threshold: float = 0.5
        # Nonlinear option: a HistGradientBoostingClassifier when fit(kind="gbt").
        # When set, predict_proba uses it instead of the logistic coef/intercept.
        self._gbt = None
        # Learned-context (e.g. CPC frame embeddings around each proposal). When
        # _use_context is True, fit/predict expect a context matrix aligned with the
        # proposals (see proposal_context) and hstack it onto the scalar features.
        self._use_context = False
        self._context_dim = 0

    # ---- training ----
    def fit(
        self,
        proposals: list[Proposal],
        y: np.ndarray,
        *,
        kind: str = "logistic",
        context: np.ndarray | None = None,
    ) -> "BoundaryCalibrator":
        """Fit the calibrator: ``kind='logistic'`` (default) or ``'gbt'`` (nonlinear).

        GBT (HistGradientBoosting) exploits feature interactions the logistic model
        misses and lifts combined F1; logistic is the lightweight fallback. If
        ``context`` is given (a matrix aligned with ``proposals``, e.g. from
        :func:`proposal_context` over CPC frame embeddings), it is hstacked onto the
        scalar features — the learned-context path that pushes combined F1 toward 0.85.
        """
        X = proposals_to_matrix(proposals)
        if context is not None:
            context = np.asarray(context, dtype=np.float64)
            if context.shape[0] != X.shape[0]:
                raise ValueError("context row count must match proposals")
            X = np.hstack([X, context])
            self._use_context = True
            self._context_dim = int(context.shape[1])
        else:
            self._use_context = False
            self._context_dim = 0
        if X.shape[0] != len(y):
            raise ValueError("proposals and y length mismatch")
        n_feat = X.shape[1]
        self._mean = X.mean(axis=0) if len(X) else np.zeros(n_feat)
        self._std = X.std(axis=0) if len(X) else np.ones(n_feat)
        self._std[self._std < 1e-8] = 1.0
        self._gbt = None
        if len(set(y.tolist())) < 2:
            # Degenerate (all one class): constant-prior logistic so predict runs.
            self._coef = np.zeros(n_feat)
            self._intercept = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
            return self
        Xs = (X - self._mean) / self._std
        if kind == "gbt":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._gbt = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.1, max_depth=4, random_state=0
            ).fit(Xs, y)
            self._coef = None
            self._intercept = 0.0
            return self
        from sklearn.linear_model import LogisticRegression
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
    def predict_proba(self, proposals: list[Proposal], *, context: np.ndarray | None = None) -> np.ndarray:
        if self._coef is None and self._gbt is None:
            raise RuntimeError("calibrator is not fitted")
        X = proposals_to_matrix(proposals)
        if self._use_context:
            if context is None:
                raise RuntimeError(
                    "calibrator was fit with learned context; pass context= "
                    "(e.g. proposal_context(..., CPCBackbone.embed_frames(...)))"
                )
            X = np.hstack([X, np.asarray(context, dtype=np.float64)])
        Xs = (X - self._mean) / self._std
        if self._gbt is not None:
            return self._gbt.predict_proba(Xs)[:, 1]
        return 1.0 / (1.0 + np.exp(-(Xs @ self._coef + self._intercept)))

    def apply(
        self,
        proposals: list[Proposal],
        *,
        keep_threshold: float | None = None,
        abstain_threshold: float = 0.3,
        context: np.ndarray | None = None,
    ) -> tuple[list[CalibratedBoundary], list[CalibratedBoundary]]:
        """Gate proposals into (kept, abstained). Below abstain_threshold dropped.

        ``keep_threshold`` defaults to the tuned ``self.keep_threshold`` so a
        refit-time threshold carries through to inference automatically. ``context``
        is the learned-context matrix (required if the calibrator was fit with one).
        """
        if keep_threshold is None:
            keep_threshold = self.keep_threshold
        if not (0.0 <= keep_threshold <= 1.0):
            raise ValueError("require 0 <= keep_threshold <= 1")
        # The ABSTAIN band cannot exceed keep by definition; clamp rather than
        # raise so a low tuned keep_threshold (e.g. 0.15) works at inference.
        abstain_threshold = max(0.0, min(abstain_threshold, keep_threshold))
        probs = self.predict_proba(proposals, context=context)
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
                    "keep_threshold": self.keep_threshold,
                    "gbt": self._gbt,
                    "use_context": self._use_context,
                    "context_dim": self._context_dim,
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
        obj.keep_threshold = state.get("keep_threshold", 0.5)  # backward-compat
        obj._gbt = state.get("gbt", None)  # None for logistic checkpoints
        obj._use_context = state.get("use_context", False)
        obj._context_dim = state.get("context_dim", 0)
        return obj

    @property
    def fitted(self) -> bool:
        return self._coef is not None or self._gbt is not None
