"""Alias-collapsed motif mining — replaces tokens with functional-alias group IDs.

If motifs become STRONGER after collapsing acoustically-distinct-but-functionally-
equivalent tokens, the sequence structure is FUNCTIONAL, not purely acoustic.
alias_gain = support_after / support_before.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations
from .token_motifs import mine_ngrams, build_motifs


def collapse_to_alias(sequences, alias_map):
    """Replace each token with its alias group ID (or itself if unmapped)."""
    return [[alias_map.get(tok, tok) for tok in seq] for seq in sequences]


def mine_alias_motifs(sequences, alias_map, min_n=2, max_n=5, min_support=3):
    """Mine motifs after collapsing tokens → alias IDs. Returns (motifs, alias_gain_dict)."""
    raw_counts = mine_ngrams(sequences, min_n, max_n, min_support)
    collapsed = collapse_to_alias(sequences, alias_map)
    alias_counts = mine_ngrams(collapsed, min_n, max_n, min_support)
    alias_motifs = build_motifs(alias_counts, collapsed, motif_type="alias", min_support=min_support)
    # alias_gain per pattern: support_collapsed / support_raw
    total_raw = sum(raw_counts.values()) or 1
    total_alias = sum(alias_counts.values()) or 1
    gain = total_alias / total_raw
    for m in alias_motifs:
        m.alias_gain = round(gain, 4)
        m.interpretation = (
            f"functional-role motif (alias_gain {gain:.2f})"
            if gain > 1.0 else
            "alias-collapsed motif (no gain)"
        )
    return alias_motifs, gain
