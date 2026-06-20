"""Token-sequence motif variants for playback stimulus design (Phase 1).

Each variant is a CONTROLLED perturbation of a discovered motif's token pattern, used to ask
"does this structure matter to the animal?" These are falsifiable sequence-perturbation
hypotheses — NOT a grammar or semantics claim. Pure stdlib; deterministic given a seed.

The stimulus builder maps a variant token sequence to per-token exemplar audio.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def reverse(pattern):
    """C B A — tests directionality / order sensitivity."""
    return list(reversed(list(pattern)))


def shuffle(pattern, seed: int = 0):
    """Random permutation of the same tokens — tests sequence vs bag-of-tokens."""
    rng = np.random.default_rng(seed)
    p = list(pattern)
    rng.shuffle(p)
    return p


def token_delete(pattern, seed: int = 0):
    """Drop one token (A C from A B C) — tests whether every position is load-bearing."""
    p = list(pattern)
    if len(p) <= 1:
        return p
    rng = np.random.default_rng(seed)
    drop = int(rng.integers(0, len(p)))
    return [t for i, t in enumerate(p) if i != drop]


def token_insert(pattern, vocab, seed: int = 0):
    """Insert one random in-vocab token (A X B C) — tests rejection of foreign units."""
    if not vocab:
        return list(pattern)
    rng = np.random.default_rng(seed)
    new_tok = int(rng.choice(list(vocab)))
    pos = int(rng.integers(0, len(pattern) + 1))
    return list(pattern[:pos]) + [new_tok] + list(pattern[pos:])


def random_control(vocab, n: int, seed: int = 0):
    """X Q Z — a length-matched random sequence as the null/control condition."""
    if not vocab or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    return [int(rng.choice(list(vocab))) for _ in range(n)]
