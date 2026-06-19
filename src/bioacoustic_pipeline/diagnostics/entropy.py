"""Phase 5.1: Entropy-reduction validation.

Proves the detected segmentation is doing information work: the conditional
entropy of the token stream GIVEN segment labels is lower than the raw token
entropy (segmenting by CPC boundaries makes subsequent tokens more predictable).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class EntropyReduction:
    """How much segmentation reduces token-stream uncertainty."""

    raw_entropy_bits: float
    segmented_entropy_bits: float  # H(token | segment_id)
    reduction_bits: float  # raw - segmented
    reduction_ratio: float  # reduction / raw


def shannon_entropy(token_counts: dict[int, int]) -> float:
    """Shannon entropy (bits) of a token-count distribution."""
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in token_counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def conditional_entropy_given_segments(
    token_stream: list[int], segment_ids: list[int]
) -> float:
    """H(token | segment) = sum_s P(s) * H(token | segment=s)."""
    if len(token_stream) != len(segment_ids):
        raise ValueError("token_stream and segment_ids must be the same length")
    total = len(token_stream)
    if total == 0:
        return 0.0

    seg_counts: Counter[int] = Counter()
    seg_token_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for tok, sid in zip(token_stream, segment_ids):
        seg_counts[sid] += 1
        seg_token_counts[sid][tok] += 1

    h = 0.0
    for sid, counts in seg_token_counts.items():
        p_seg = seg_counts[sid] / total
        h += p_seg * shannon_entropy(dict(counts))
    return h


def _collapse_ngrams(
    token_stream: list[int], segment_ids: list[int], ngram_dict: set[tuple[int, ...]]
) -> tuple[list, list]:
    """Collapse listed n-grams into single symbols (keeps segments aligned)."""
    if not ngram_dict:
        return list(token_stream), list(segment_ids)
    max_len = max(len(g) for g in ngram_dict)
    out_tokens: list = []
    out_segs: list = []
    i = 0
    n = len(token_stream)
    while i < n:
        matched = None
        for L in range(max_len, 1, -1):
            if i + L <= n and tuple(token_stream[i : i + L]) in ngram_dict:
                matched = L
                break
        if matched is not None:
            out_tokens.append(tuple(token_stream[i : i + matched]))
            out_segs.append(segment_ids[i])
            i += matched
        else:
            out_tokens.append(token_stream[i])
            out_segs.append(segment_ids[i])
            i += 1
    return out_tokens, out_segs


def measure_entropy_reduction(
    token_stream: list[int],
    segment_ids: list[int],
    *,
    ngram_dict: set[tuple[int, ...]] | None = None,
) -> EntropyReduction:
    """Measure how much segmentation (and optional n-gram collapse) reduces entropy.

    Args:
        token_stream: tokens.
        segment_ids: segment label per token (from CPC boundaries).
        ngram_dict: n-grams to collapse to single symbols before measuring.
    """
    stream, sids = _collapse_ngrams(token_stream, segment_ids, ngram_dict or set())

    raw_h = shannon_entropy(dict(Counter(stream)))
    seg_h = conditional_entropy_given_segments(stream, sids)
    reduction = raw_h - seg_h
    ratio = reduction / raw_h if raw_h > 0 else 0.0
    return EntropyReduction(
        raw_entropy_bits=raw_h,
        segmented_entropy_bits=seg_h,
        reduction_bits=reduction,
        reduction_ratio=ratio,
    )
