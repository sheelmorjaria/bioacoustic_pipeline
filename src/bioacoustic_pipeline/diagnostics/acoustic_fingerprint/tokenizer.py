"""Shared cross-species fingerprint tokenizer.

A discrete tokenization of the acoustic fingerprint that is IDENTICAL across species, so
human and bat can be compared on the same token space (removing the per-species VQ-VAE
confound). Firewall-legal: the fingerprint is ``applies_to_species: []`` (all-species).

Why feature selection, not the raw ``combined`` vector: the fingerprint's spectral
descriptors are in Hz. Bat calls (~20-100 kHz) and human speech (~100-300 Hz) differ by
~1000x, so clustering on raw Hz features would separate SPECIES rather than acoustic
shape. We drop the absolute-Hz / sample-rate-scale physics features (ZCR, centroid,
spread, rolloff, dominant frequency) and cluster on the remaining DIMENSIONLESS
shape/dynamics features (crest, spectral skew/kurtosis, flatness, entropy, flux, ADSR,
and all macro/micro-texture). A token then denotes an acoustic-SHAPE class that means the
same thing in both species. ``tokenize`` is unsupervised k-means on a single shared
codebook fit on pooled human+bat fingerprints, so token id X is the same class in both.

This is an acoustic-similarity tokenization, NOT meaning (Semantic Ceiling still holds).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Physics-block indices (within the first PHYSICS_DIM entries of `combined`) that are
# absolute-Hz / sample-rate-scale and must be dropped for cross-species clustering:
# zcr(4), centroid(5), spread(6), rolloff(9), dom_freq(13).
HZ_SCALE_PHYSICS_INDICES = (4, 5, 6, 9, 13)
PHYSICS_DIM = 15


def cross_species_mask(dim: int) -> np.ndarray:
    """Boolean mask over a `combined` vector keeping the cross-species-comparable dims."""
    drop = {i for i in HZ_SCALE_PHYSICS_INDICES if i < dim}
    return np.array([i not in drop for i in range(dim)], dtype=bool)


@dataclass
class FingerprintTokenizer:
    """Shared k-means tokenizer over the dimensionless fingerprint features."""

    k: int
    feature_mask: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    centroids: np.ndarray
    n_fit: int = 0

    @property
    def n_features(self) -> int:
        return int(self.feature_mask.sum())

    @classmethod
    def fit(cls, combined_vectors, k: int = 128, *, seed: int = 0) -> "FingerprintTokenizer":
        """Fit one shared codebook on pooled (multi-species) fingerprint vectors."""
        from sklearn.cluster import KMeans  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        X = np.asarray(combined_vectors, dtype=np.float64)
        mask = cross_species_mask(X.shape[1])
        Xm = X[:, mask]
        scaler = StandardScaler().fit(Xm)
        Xs = scaler.transform(Xm)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)
        return cls(
            k=k,
            feature_mask=mask,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            centroids=km.cluster_centers_,
            n_fit=len(X),
        )

    def _project(self, combined: np.ndarray) -> np.ndarray:
        v = np.asarray(combined, dtype=np.float64)
        x = v[self.feature_mask]
        return (x - self.scaler_mean) / np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

    def tokenize(self, combined: np.ndarray) -> int:
        """Nearest shared-codebook centroid (cluster id = token). -1 if the segment was
        zeroed (silent/too-short fingerprint)."""
        v = np.asarray(combined, dtype=np.float64)
        if not np.any(v):
            return -1
        x = self._project(v)[None, :]
        d = ((x - self.centroids) ** 2).sum(axis=1)
        return int(np.argmin(d))

    def utilization(self, combined_vectors) -> dict[str, int | float]:
        """How many of the k codes this subset uses (collapse check)."""
        toks = [self.tokenize(v) for v in combined_vectors if np.any(v)]
        toks = [t for t in toks if t >= 0]
        used = len(set(toks))
        return {"used": used, "k": self.k, "utilization": round(used / self.k, 3), "n": len(toks)}

    def save(self, path: str | Path) -> None:
        np.savez(
            str(path), k=self.k, feature_mask=self.feature_mask,
            scaler_mean=self.scaler_mean, scaler_scale=self.scaler_scale,
            centroids=self.centroids, n_fit=self.n_fit,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FingerprintTokenizer":
        d = np.load(str(path), allow_pickle=True)
        return cls(
            k=int(d["k"]), feature_mask=d["feature_mask"],
            scaler_mean=d["scaler_mean"], scaler_scale=d["scaler_scale"],
            centroids=d["centroids"], n_fit=int(d["n_fit"]),
        )
