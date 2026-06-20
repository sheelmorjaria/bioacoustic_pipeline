"""Token n-gram motif mining over VQ token sequences.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations
from collections import Counter
from .scoring import Motif, npmi, entropy_reduction, score_motif


def mine_ngrams(sequences, min_n=2, max_n=5, min_support=3):
    """Mine n-grams from a list of per-file token sequences. Returns {pattern: support}."""
    counts: Counter = Counter()
    for seq in sequences:
        for n in range(min_n, max_n + 1):
            for i in range(len(seq) - n + 1):
                counts[tuple(seq[i:i+n])] += 1
    return {p: c for p, c in counts.items() if c >= min_support}


def build_motifs(raw_counts, sequences, motif_type="token", min_support=3):
    """Score n-gram counts into Motif objects with NPMI + entropy reduction."""
    flat = [t for seq in sequences for t in seq]
    unigram = Counter(flat)
    total = len(flat) or 1
    motifs = []
    for i, (pattern, support) in enumerate(
        sorted(raw_counts.items(), key=lambda x: -x[1])
    ):
        npmi_val = npmi(pattern, support, unigram, total)
        ent_red = entropy_reduction(flat, pattern, support)
        sc = score_motif(pattern, support, unigram, total, flat, stability=0.5)
        motifs.append(Motif(
            motif_id=f"{motif_type}_{i}", motif_type=motif_type,
            pattern=pattern, support=support, score=sc,
            npmi=npmi_val, entropy_reduction=ent_red, stability=0.5,
            interpretation=f"recurrent {motif_type} sequence (support {support})",
        ))
    return motifs
