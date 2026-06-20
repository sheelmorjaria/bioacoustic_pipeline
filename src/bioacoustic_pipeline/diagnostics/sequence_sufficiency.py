"""Sequence-sufficiency gate — abstention guard for sequence diagnostics.

No sequence-level diagnostic (functional aliasing, motif mining, LZ complexity,
turn syntax) may return a BIOLOGICAL NULL when the underlying token sequences
are too sparse to test. A conservative NBD profile can collapse multi-unit bouts
into zero or one segment (e.g. the bat holophrastic prior: debounce 200 ms +
spectral floor 0.50), which starves every sequence diagnostic of bigrams. Without
this gate those diagnostics emit a technically-correct "absent" that reads as a
biological finding (a false null). The gate forces them to ABSTAIN with a stated
machine reason, so the verdict reports ``not_evaluable`` rather than a misleading
"absent".

Operates on per-bout token sequences — the universal input to all sequence
diagnostics — so it is decoupled from segmentation / tokenization specifics and
unit-testable without audio.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVALUABLE = "evaluable"
NOT_EVALUABLE = "not_evaluable"

# Machine-readable reason codes (the verdict composes the human-readable line).
REASON_SEGMENTATION_COLLAPSE = "segmentation_collapse"  # most bouts yielded 0 tokens
REASON_INSUFFICIENT_TOKENS = "insufficient_token_sequence"  # bouts too short on average
REASON_INSUFFICIENT_BIGRAMS = "insufficient_bigram_support"  # not enough bigrams to fit contexts

HUMAN_REASONS = {
    REASON_SEGMENTATION_COLLAPSE: (
        "segmentation collapsed most bouts to zero tokens — no sequence to test"
    ),
    REASON_INSUFFICIENT_TOKENS: (
        "median tokens per bout below floor — sequences too short to test"
    ),
    REASON_INSUFFICIENT_BIGRAMS: (
        "too few bigrams to estimate token context distributions"
    ),
}


@dataclass
class SufficiencyResult:
    """Outcome of the sufficiency gate."""

    status: str  # EVALUABLE | NOT_EVALUABLE
    reason: str | None  # None when evaluable, else a REASON_* code
    median_tokens_per_bout: float
    nonzero_bout_rate: float
    total_tokens: int
    total_bigrams: int
    total_trigrams: int
    usable_bout_count: int  # bouts with >= 2 tokens (the ones that contribute bigrams)
    n_bouts: int
    thresholds: dict = field(default_factory=dict)

    @property
    def evaluable(self) -> bool:
        return self.status == EVALUABLE

    @property
    def reason_detail(self) -> str:
        """Human-readable reason (empty when evaluable)."""
        return HUMAN_REASONS.get(self.reason or "", "")

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "median_tokens_per_bout": round(self.median_tokens_per_bout, 3),
            "nonzero_bout_rate": round(self.nonzero_bout_rate, 3),
            "total_tokens": self.total_tokens,
            "total_bigrams": self.total_bigrams,
            "total_trigrams": self.total_trigrams,
            "usable_bout_count": self.usable_bout_count,
            "n_bouts": self.n_bouts,
            "thresholds": self.thresholds,
        }


def evaluate_sufficiency(
    sequences,
    *,
    min_nonzero_bout_rate: float = 0.8,
    min_median_tokens_per_bout: float = 2.0,
    min_total_bigrams: int = 500,
) -> SufficiencyResult:
    """Gate: are these token sequences dense enough to test sequence diagnostics?

    Args:
        sequences: per-bout token lists (e.g. one list of token ids per audio file).
        min_nonzero_bout_rate: floor on the fraction of bouts yielding >= 1 token.
            Below this the segmentation is collapsing bouts and any sequence
            diagnostic would be a false null.
        min_median_tokens_per_bout: floor on the median bout length.
        min_total_bigrams: floor on the corpus bigram count (the raw material for
            context-vector / alias / motif statistics).

    Returns:
        A :class:`SufficiencyResult`. ``evaluable`` is True iff all three floors hold.

    Checks ordered by root-causality: a segmentation that drops most bouts is the
    underlying problem, not the downstream bigram count it produces.
    """
    seqs = [list(int(t) for t in s) for s in sequences]
    n_bouts = len(seqs)
    lengths = np.array([len(s) for s in seqs], dtype=float) if seqs else np.array([0.0])
    nonzero = int((lengths >= 1).sum())
    nonzero_rate = nonzero / max(n_bouts, 1)
    median_tokens = float(np.median(lengths)) if len(lengths) else 0.0
    total_tokens = int(lengths.sum())
    total_bigrams = int(sum(max(0, len(s) - 1) for s in seqs))
    total_trigrams = int(sum(max(0, len(s) - 2) for s in seqs))
    usable = int((lengths >= 2).sum())

    thresholds = {
        "min_nonzero_bout_rate": min_nonzero_bout_rate,
        "min_median_tokens_per_bout": min_median_tokens_per_bout,
        "min_total_bigrams": min_total_bigrams,
    }

    if nonzero_rate < min_nonzero_bout_rate:
        reason = REASON_SEGMENTATION_COLLAPSE
    elif median_tokens < min_median_tokens_per_bout:
        reason = REASON_INSUFFICIENT_TOKENS
    elif total_bigrams < min_total_bigrams:
        reason = REASON_INSUFFICIENT_BIGRAMS
    else:
        reason = None

    return SufficiencyResult(
        status=EVALUABLE if reason is None else NOT_EVALUABLE,
        reason=reason,
        median_tokens_per_bout=median_tokens,
        nonzero_bout_rate=nonzero_rate,
        total_tokens=total_tokens,
        total_bigrams=total_bigrams,
        total_trigrams=total_trigrams,
        usable_bout_count=usable,
        n_bouts=n_bouts,
        thresholds=thresholds,
    )
