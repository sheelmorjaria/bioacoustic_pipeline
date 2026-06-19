"""Immutable embedding cache — never recompute a backbone embedding.

Embedding banks (CPC + BioMAE + AVES + Perch) are expensive; changing a
diagnostic threshold must not trigger a re-encode. This maps
``(backbone, audio_hash, sample_rate) -> vector`` to disk (one .npy per key),
keyed by a BLAKE2b hash of the raw audio bytes so the cache is content-addressed
and immutable.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def audio_hash(audio: np.ndarray, sr: int) -> str:
    """Content hash of the audio array + sample rate (16-char blake2b)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(bytes(np.ascontiguousarray(audio).data))
    h.update(str(sr).encode())
    return h.hexdigest()


class EmbeddingCache:
    """Content-addressed on-disk cache of per-backbone segment embeddings."""

    def __init__(self, root: str | Path = "cache/embeddings"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, backbone: str, audio: np.ndarray, sr: int) -> Path:
        key = f"{backbone}_{sr}_{audio_hash(audio, sr)}"
        return self.root / f"{key}.npy"

    def get(self, backbone: str, audio: np.ndarray, sr: int) -> np.ndarray | None:
        p = self._path(backbone, audio, sr)
        if p.exists():
            return np.load(p)
        return None

    def put(self, backbone: str, audio: np.ndarray, sr: int, vector: np.ndarray) -> None:
        np.save(self._path(backbone, audio, sr), np.asarray(vector))

    def embed(
        self,
        backbone: str,
        audio: np.ndarray,
        sr: int,
        compute_fn,
    ) -> np.ndarray:
        """Get-or-compute: return cached embedding, else call compute_fn and store."""
        cached = self.get(backbone, audio, sr)
        if cached is not None:
            return cached
        vec = np.asarray(compute_fn(audio, sr))
        self.put(backbone, audio, sr, vec)
        return vec
