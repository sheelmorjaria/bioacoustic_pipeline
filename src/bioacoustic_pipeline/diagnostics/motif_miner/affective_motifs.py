"""Affective-trajectory motif mining — uses the trained β-VAE affect vectors.

Clusters 16-D affect vectors into discrete affect "states", then mines n-grams over
affect-state sequences to find recurring prosodic trajectories (e.g., rising arousal
→ sharp call → decay). Human-only (the β-VAE is trained on LibriSpeech CPC latents).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import numpy as np

from .token_motifs import mine_ngrams, build_motifs


def mine_affective_motifs(affect_sequences: list[list[np.ndarray]],
                          min_n=2, max_n=4, min_support=3, threshold=0.6):
    """Cluster affect vectors → affect-state IDs → mine n-grams.

    ``affect_sequences``: per-file lists of 16-D β-VAE affect vectors.
    Uses greedy cosine clustering (same as acoustic_motifs) to discretize into
    affect states, then mines recurring affect-state sequences.
    """
    from .acoustic_motifs import cluster_fingerprints

    all_vecs = np.vstack([np.asarray(v) for seq in affect_sequences for v in seq])
    if all_vecs.shape[0] < 2:
        return [], {}
    labels = cluster_fingerprints(all_vecs, threshold)
    # rebuild per-file affect-state sequences
    affect_seqs: list[list[int]] = []
    idx = 0
    for seq in affect_sequences:
        affect_seqs.append(labels[idx:idx + len(seq)])
        idx += len(seq)
    raw = mine_ngrams(affect_seqs, min_n, max_n, min_support)
    motifs = build_motifs(raw, affect_seqs, motif_type="affective", min_support=min_support)
    for m in motifs:
        m.interpretation = "affective-trajectory motif (prosodic state pattern)"
    return motifs, raw
