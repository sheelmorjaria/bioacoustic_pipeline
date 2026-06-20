"""Acoustic-shape motif mining — clusters fingerprints then mines cluster-ID n-grams.

Catches motifs the VQ-VAE misses: similar acoustic shapes (FM sweeps, pulse trains,
harmonic contours) that map to different VQ tokens but share a fingerprint cluster.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import numpy as np

from ..acoustic_fingerprint.similarity import normalize_batch, cosine
from .token_motifs import mine_ngrams, build_motifs


def cluster_fingerprints(fps: np.ndarray, threshold: float = 0.7) -> list[int]:
    """Greedy nearest-centroid clustering by cosine similarity.

    Returns a cluster-ID per fingerprint. Each fp joins the nearest existing centroid
    if cosine >= threshold; otherwise it starts a new cluster.
    """
    if fps.ndim != 2 or fps.shape[0] == 0:
        return []
    normed = normalize_batch(fps)
    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for row in normed:
        best, best_sim = -1, threshold
        for ci, c in enumerate(centroids):
            sim = cosine(row, c)
            if sim > best_sim:
                best_sim, best = sim, ci
        if best < 0:
            centroids.append(row)
            labels.append(len(centroids) - 1)
        else:
            centroids[best] = (centroids[best] * labels.count(best) + row) / (labels.count(best) + 1)
            labels.append(best)
    return labels


def mine_acoustic_motifs(fingerprint_sequences: list[list[np.ndarray]],
                         min_n=2, max_n=5, min_support=3, threshold=0.7):
    """Cluster fingerprints per-corpus, convert to cluster-ID sequences, mine n-grams."""
    all_fps = np.vstack([np.asarray(fp) for seq in fingerprint_sequences for fp in seq])
    if all_fps.shape[0] < 2:
        return [], {}
    labels = cluster_fingerprints(all_fps, threshold)
    # rebuild per-file cluster-ID sequences in the same order
    cluster_seqs: list[list[int]] = []
    idx = 0
    for seq in fingerprint_sequences:
        cluster_seqs.append(labels[idx:idx + len(seq)])
        idx += len(seq)
    raw = mine_ngrams(cluster_seqs, min_n, max_n, min_support)
    motifs = build_motifs(raw, cluster_seqs, motif_type="acoustic", min_support=min_support)
    for m in motifs:
        m.interpretation = "acoustic-shape motif (fingerprint cluster pattern)"
    return motifs, raw
