"""Frozen motif discovery + assignment (§46 Stage 2, context-conditioned track).

Contract: the motif vocabulary is DISCOVERED on the train split only (fit),
then FROZEN — assignment on held-out recordings never refits, so eval-split
motif usage is a measurement against a fixed vocabulary, not a moving
target. Callers are pre-filtered through the Stage-1 eligibility matrix so
only callers who can support the context contrast on both sides enter
discovery at all.

The clustering core is a KMeans surrogate for the repo's DUSTED family
discovery (fixed-length cosine subsequence clustering in
``diagnostics/motif_miner``): this class exists to enforce the
frozen-vocabulary train/eval contract with a swappable core (any object
with fit/predict maps onto ``fit``/``assign``). Swapping in the cosine
subsequence core is a drop-in and changes nothing downstream.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def filter_eligible_intervals(
    intervals,
    *,
    contexts,
    min_bouts: int,
    eligibility_matrix: dict,
):
    """Keep only callers eligible for BOTH sides of the contrast.

    Reuses the Stage-1 support matrix (``build_caller_context_support_matrix``)
    — a caller thin on either side is excluded here, not silently pooled.
    """
    c1, c2 = contexts
    eligible_callers = {
        caller
        for (caller, ctx), count in eligibility_matrix.items()
        if ctx in (c1, c2)
        and eligibility_matrix.get((caller, c1), 0) >= min_bouts
        and eligibility_matrix.get((caller, c2), 0) >= min_bouts
    }
    return [a for a in intervals if a.emitter in eligible_callers and a.context in contexts]


class FrozenDUSTEDModel:
    """Frozen motif vocabulary: fit on train, assign everywhere, never refit."""

    def __init__(self, n_clusters: int = 24, random_state: int = 42, core=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._core = core  # optional injected discovery core (fit/predict)
        self.centers_: np.ndarray | None = None

    def fit(self, train_trajectories: np.ndarray):
        """Discover motif families on TRAIN-split trajectories only."""
        from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415

        core = self._core or MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=3,
            batch_size=4096,
        )
        core.fit(train_trajectories)
        self._core = core
        self.centers_ = np.asarray(core.cluster_centers_, dtype=np.float64)
        return self

    def assign(self, trajectories: np.ndarray) -> np.ndarray:
        """Assign the FROZEN vocabulary to any trajectories (no refit)."""
        if self._core is None:
            raise RuntimeError("Model must be fitted before assignment.")
        x = np.asarray(trajectories)
        # centers may be float64 (save path) while data is float32 (the
        # dtype the core was fitted on) — align before predict
        self._core.cluster_centers_ = np.asarray(self._core.cluster_centers_, dtype=x.dtype)
        return np.asarray(self._core.predict(x))

    def save(self, path: str | Path) -> None:
        """Persist the frozen vocabulary (centers + config) as npz."""
        if self.centers_ is None:
            raise RuntimeError("Nothing to save: model not fitted.")
        np.savez_compressed(
            path,
            centers=self.centers_,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )

    def load(self, path: str | Path) -> "FrozenDUSTEDModel":
        """Load a frozen vocabulary; assignment works immediately."""
        from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415

        data = np.load(path)
        self.centers_ = data["centers"]
        self.n_clusters = int(data["n_clusters"])
        self.random_state = int(data["random_state"])
        core = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=1)
        core.cluster_centers_ = self.centers_
        core._n_threads = 1
        self._core = core
        return self
