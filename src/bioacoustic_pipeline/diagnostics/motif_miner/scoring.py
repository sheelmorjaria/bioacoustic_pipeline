"""Motif scoring: dataclasses + information-theoretic metrics for recurring patterns.

A motif is a recurring pattern (token n-gram, alias-role sequence, fingerprint cluster
sequence, etc.) that survives null-model testing. Motifs are NOT syntax — they are
statistically tested sequence hypotheses requiring behavioral validation.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

EPS = 1e-12


@dataclass
class Motif:
    """A discovered recurring pattern."""

    motif_id: str = ""
    motif_type: str = ""  # token | alias | acoustic | affective | turn
    pattern: tuple = ()
    support: int = 0
    examples: list = field(default_factory=list)
    score: float = 0.0
    p_value: float | None = None
    null_percentile: float | None = None
    stability: float = 0.0
    npmi: float = 0.0
    entropy_reduction: float = 0.0
    alias_gain: float | None = None  # alias_motifs: support_after / support_before
    interpretation: str = ""

    def as_dict(self) -> dict:
        d = {
            "motif_id": self.motif_id, "motif_type": self.motif_type,
            "pattern": list(self.pattern), "support": self.support,
            "score": round(self.score, 4), "npmi": round(self.npmi, 4),
            "stability": round(self.stability, 4),
        }
        if self.p_value is not None:
            d["p_value"] = round(self.p_value, 4)
        if self.null_percentile is not None:
            d["null_percentile"] = round(self.null_percentile, 2)
        if self.alias_gain is not None:
            d["alias_gain"] = round(self.alias_gain, 4)
        d["interpretation"] = self.interpretation
        return d


@dataclass
class MotifReport:
    """Full output of the Motif Miner."""

    segment_count: int = 0
    token_motifs: list[Motif] = field(default_factory=list)
    alias_motifs: list[Motif] = field(default_factory=list)
    acoustic_motifs: list[Motif] = field(default_factory=list)
    turn_motifs: list[Motif] = field(default_factory=list)
    null_results: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "segment_count": self.segment_count,
            "n_token_motifs": len(self.token_motifs),
            "n_alias_motifs": len(self.alias_motifs),
            "n_acoustic_motifs": len(self.acoustic_motifs),
            "n_turn_motifs": len(self.turn_motifs),
            "top_token_motifs": [m.as_dict() for m in self.token_motifs[:5]],
            "top_alias_motifs": [m.as_dict() for m in self.alias_motifs[:5]],
            "summary": self.summary,
        }


def npmi(pattern: tuple, pattern_support: int, unigram_counts: Counter, total_tokens: int) -> float:
    """Normalized PMI for an n-gram pattern.

    NPMI = PMI / -log P(pattern), bounded [-1, 1]. Positive = more than chance.
    Uses the product of individual token probabilities as the expected co-occurrence.
    """
    if pattern_support < 1 or total_tokens < 1:
        return 0.0
    p_pattern = pattern_support / total_tokens
    p_expected = 1.0
    for tok in pattern:
        p_expected *= unigram_counts.get(tok, 0) / total_tokens
    if p_expected < EPS:
        return 0.0
    pmi = math.log(p_pattern / p_expected + EPS)
    npmi_val = pmi / (-math.log(p_pattern) + EPS)
    return max(-1.0, min(1.0, npmi_val))


def entropy_reduction(tokens: list, pattern: tuple, support: int) -> float:
    """How much knowing the pattern reduces uncertainty about what follows.

    H(token) - H(token | preceded by pattern). Positive = predictive.
    """
    if support < 2 or len(pattern) < 1:
        return 0.0
    raw_entropy = _shannon_entropy(Counter(tokens))
    # conditional: tokens that FOLLOW each occurrence of the pattern
    followers = []
    n = len(pattern)
    for i in range(len(tokens) - n):
        if tuple(tokens[i : i + n]) == pattern:
            if i + n < len(tokens):
                followers.append(tokens[i + n])
    if len(followers) < 2:
        return 0.0
    cond_entropy = _shannon_entropy(Counter(followers))
    return max(0.0, raw_entropy - cond_entropy)


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total < 1:
        return 0.0
    return -sum((c / total) * math.log2(c / total + EPS) for c in counts.values())


def score_motif(pattern: tuple, support: int, unigram_counts: Counter,
                total_tokens: int, tokens: list,
                null_percentile: float | None = None,
                stability: float = 0.0) -> float:
    """Composite motif score: support + NPMI + entropy_reduction + null + stability."""
    npmi_val = npmi(pattern, support, unigram_counts, total_tokens)
    ent_red = entropy_reduction(tokens, pattern, support)
    null_bonus = (null_percentile - 50) / 50 if null_percentile is not None else 0.0
    return max(0.0, npmi_val + 0.3 * ent_red + 0.5 * null_bonus + 0.2 * stability)
