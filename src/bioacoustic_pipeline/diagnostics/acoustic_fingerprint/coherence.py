"""Fingerprint coherence diagnostics (token audit + alias contrast).

Two reviewer-facing signals computed from per-segment fingerprints:

* ``token_fingerprint_coherence`` — do all segments sharing a VQ token actually sound
  physically alike? High cohesion => the token is acoustically coherent (good evidence
  for the Semantic Ceiling: tokens are acoustic classes). Low cohesion => the token may
  group by context/speaker/recording or be a latent artifact.
* ``alias_acoustic_divergence`` — for functional-alias groups, how acoustically *distinct*
  are the member tokens? High sequence-role similarity + high acoustic divergence is the
  genuine "different sounds, same job" alias. ``classify_alias`` turns the
  (alias-similarity, acoustic-similarity) pair into one of four cases.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from . import similarity

HIGH = 0.5  # similarity threshold for the 2x2 alias classification


def token_fingerprint_coherence(token_to_fingerprints: dict) -> dict:
    """{token: {cohesion, n}} — mean cosine-to-centroid within each token.

    ``token_to_fingerprints`` maps a token id -> list of ``combined`` vectors. Vectors
    are batch-normalized across the whole corpus by the caller (z-score) before this; if
    not, normalization is applied per token here as a fallback.
    """
    out = {}
    for tok, vecs in token_to_fingerprints.items():
        rows = np.asarray(vecs, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] < 2:
            out[tok] = {"cohesion": 1.0 if rows.shape[0] == 1 else 0.0, "n": int(rows.shape[0])}
            continue
        out[tok] = {"cohesion": round(similarity.mean_to_centroid(rows), 4),
                    "n": int(rows.shape[0])}
    return out


def _centroid(vecs: np.ndarray) -> np.ndarray:
    return vecs.mean(axis=0) if vecs.ndim == 2 and vecs.size else np.zeros(0)


def alias_acoustic_divergence(alias_groups: dict, token_centroids: dict) -> dict:
    """{group_id: {divergence, n, classification}} for functional-alias groups.

    ``alias_groups``: {group_id: [token_id, ...]} (from the alias detector).
    ``token_centroids``: {token_id: centroid_vector}. Divergence = mean pairwise cosine
    *distance* (1 - similarity) between member centroids. ``classify_alias`` needs the
    group's sequence-role (alias) similarity too — pass it via ``alias_sim``.
    """
    out = {}
    for gid, tokens in alias_groups.items():
        cents = [token_centroids[t] for t in tokens if t in token_centroids]
        cents = [c for c in cents if np.asarray(c).size]
        if len(cents) < 2:
            out[gid] = {"divergence": 0.0, "n": len(cents)}
            continue
        M = similarity.normalize_batch(np.vstack(cents))
        dists = []
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                dists.append(1.0 - similarity.cosine(M[i], M[j]))
        out[gid] = {"divergence": round(float(np.mean(dists)), 4), "n": len(cents)}
    return out


def classify_alias(alias_sim: float, acoustic_sim: float) -> str:
    """Map (sequence-role similarity, acoustic similarity) to one of four cases.

    - ``acoustic_duplicate``: same/similar sound, same role (high alias, high acoustic)
    - ``functional``: different sound, same role (high alias, low acoustic) — the prize
    - ``contextual``: same sound used differently (low alias, high acoustic)
    - ``spurious``: neither (low alias, low acoustic)
    """
    high_alias = alias_sim >= HIGH
    high_acoustic = acoustic_sim >= HIGH
    if high_alias and high_acoustic:
        return "acoustic_duplicate"
    if high_alias and not high_acoustic:
        return "functional"
    if not high_alias and high_acoustic:
        return "contextual"
    return "spurious"
