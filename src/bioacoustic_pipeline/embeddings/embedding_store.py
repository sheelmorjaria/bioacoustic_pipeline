"""Whole-item embedding cache for corpus-level evals (§45/§46).

``cache/s45_eval/`` (gitignored) — one compressed npz per (arm, version, item
list). The key binds the REPRESENTATION VERSION (checkpoint content hash for
learned arms, extractor version for handcrafted ones) and the full item list,
so a retrained checkpoint or a changed item set invalidates its cache instead
of silently serving stale embeddings — the correctness property a name-only
key lacks for freshly-trained models.

Items are corpus-dependent: whole files for the bat per-bout corpus, call
INTERVALS for marmoset (file + start + end + caller). Callers pass an opaque
identifier per item; only its exact list matters.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "cache" / "s45_eval"


def checkpoint_version(checkpoint_dir: str | Path) -> str:
    """Content hash of ``<dir>/best_model.pt`` ('none' if absent)."""
    p = Path(checkpoint_dir) / "best_model.pt"
    if not p.exists():
        return "none"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def cached_embeddings(
    arm_name: str,
    item_ids: list[str],
    compute_one: Callable[[int], np.ndarray],
    *,
    version_key: str,
    cache_dir: str | Path | None = None,
    log=print,
) -> np.ndarray:
    """Embeddings for ``len(item_ids)`` items, cached on (arm, version, items).

    ``compute_one(i)`` embeds item ``i`` (e.g. closure over audio slices);
    called only on cache miss. A stale cache (item-list mismatch) recomputes.
    """
    digest = hashlib.sha256(("\x1f".join(item_ids) + "\x1e" + version_key).encode()).hexdigest()[
        :16
    ]
    directory = Path(cache_dir) if cache_dir else CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm_name}_{digest}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if list(data["items"]) == list(item_ids):
            log(f"{arm_name}: embeddings from cache ({path.name})")
            return data["embeddings"].astype(np.float64)
        log(f"{arm_name}: cache stale (item list mismatch) — recomputing")
    emb = np.asarray([np.asarray(compute_one(i), dtype=np.float64) for i in range(len(item_ids))])
    np.savez_compressed(path, embeddings=emb, items=np.asarray(item_ids))
    log(f"{arm_name}: embeddings computed and cached ({path.name})")
    return emb
