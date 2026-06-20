"""Candidate selection — high-confidence motifs + per-token natural exemplars (Phase 1).

Bridges the diagnostic layer (motifs, segments) to the stimulus builder. For each VQ token it
keeps natural exemplar audio (``Segment.audio``), choosing the most acoustically typical
instance by fingerprint-to-centroid cosine. Then it selects motifs whose tokens all have
exemplars, ranked by support/significance, so every stimulus is built from real segment audio.

Species-agnostic: the caller pairs each segment's audio with its token (however obtained —
human VQ-VAE, ``SpeciesAcousticTokenizer`` for bats, or the fingerprint k-means fallback).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TokenExemplar:
    """One natural occurrence of a token: its audio + provenance + optional fingerprint."""

    token: int
    audio: np.ndarray
    sr: int
    segment_id: str = ""
    fingerprint: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_exemplar_pool(annotated_segments) -> dict[int, list[TokenExemplar]]:
    """Group token-tagged segments into ``{token: [TokenExemplar, ...]}``.

    ``annotated_segments`` is an iterable of mappings/dicts each with at least ``token``,
    ``audio``, ``sr`` (optional ``segment_id``, ``fingerprint``, ``metadata``). Tokens == -1
    (too-short) are dropped.
    """
    pool: dict[int, list[TokenExemplar]] = {}
    for seg in annotated_segments:
        tok = int(seg.get("token", -1))
        if tok < 0:
            continue
        pool.setdefault(tok, []).append(
            TokenExemplar(
                token=tok,
                audio=np.asarray(seg["audio"], dtype=np.float32),
                sr=int(seg["sr"]),
                segment_id=str(seg.get("segment_id", "")),
                fingerprint=seg.get("fingerprint"),
                metadata=dict(seg.get("metadata", {})),
            )
        )
    return pool


def typical_exemplars(
    pool: dict[int, list[TokenExemplar]], n_per_token: int = 1
) -> dict[int, list[TokenExemplar]]:
    """Pick the most acoustically-typical exemplar(s) per token (max fingerprint→centroid cosine).

    Falls back to the first exemplar when fingerprints are absent.
    """
    out: dict[int, list[TokenExemplar]] = {}
    for tok, exs in pool.items():
        if n_per_token >= len(exs):
            chosen = list(exs)
        else:
            fps = [
                e.fingerprint
                for e in exs
                if e.fingerprint is not None and np.any(np.asarray(e.fingerprint))
            ]
            if len(fps) >= len(exs):
                centroid = np.mean([np.asarray(f, dtype=np.float64) for f in fps], axis=0)
                order = sorted(
                    range(len(exs)), key=lambda i: -_cosine(exs[i].fingerprint, centroid)
                )
            else:
                order = list(range(len(exs)))
            chosen = [exs[i] for i in order[:n_per_token]]
        out[tok] = chosen
    return out


def select_candidates(
    motifs,
    pool: dict[int, list[TokenExemplar]],
    *,
    n_motifs: int = 5,
    min_support: int = 3,
) -> tuple[list, set[int]]:
    """Return ``(selected_motifs, vocab)`` — motifs whose tokens all have exemplars, top-N by support.

    ``motifs`` is a list of objects with ``.pattern`` (token sequence), ``.support``, ``.score``
    (e.g. ``diagnostics.motif_miner.Motif``). ``vocab`` is the full token set with exemplars.
    """
    vocab = set(pool.keys())
    eligible = [
        m
        for m in motifs
        if len(tuple(m.pattern)) >= 2
        and m.support >= min_support
        and set(int(t) for t in m.pattern).issubset(vocab)
    ]
    eligible.sort(key=lambda m: (getattr(m, "support", 0), getattr(m, "score", 0.0)), reverse=True)
    return eligible[:n_motifs], vocab
