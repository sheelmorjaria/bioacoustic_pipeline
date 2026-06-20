"""Turn-taking motif mining — cross-speaker n-grams when speaker identity is available.

Generalizes the pipeline's turn-bigram diagnostics beyond bigrams. Requires speaker-
tagged token sequences (not yet in the default pipeline; will activate for Track C
dyadic/social corpora).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

from .token_motifs import mine_ngrams, build_motifs


def build_turn_events(tokens: list, speakers: list) -> list[tuple]:
    """Zip tokens with speaker tags -> [(speaker, token), ...]."""
    return list(zip(speakers, tokens))


def mine_turn_motifs(turn_sequences: list[list[tuple]], min_n=2, max_n=4, min_support=3):
    """Mine cross-speaker n-grams over (speaker, token) event sequences."""
    raw = mine_ngrams(turn_sequences, min_n, max_n, min_support)
    motifs = build_motifs(raw, turn_sequences, motif_type="turn", min_support=min_support)
    for m in motifs:
        m.interpretation = "turn-taking motif (cross-speaker pattern)"
    return motifs, raw
