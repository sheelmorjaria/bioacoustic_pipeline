"""Phase 5.2: Lempel-Ziv sequence complexity.

LZ76 production count (number of phrases) as a diagnostic that differentiates
combinatorial streams (rich recombination -> high normalized complexity) from
holistic/repetitive streams (low normalized complexity). Pure token-stream
analysis; no model deps.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LZComplexity:
    """LZ76 complexity of a token stream."""

    c_n: int  # raw production (phrase) count
    normalized: float  # c_n / (n / log2(n)) — dimensionless
    structure_class: str  # combinatorial | holistic | mixed


def _is_sublist(needle: list[int], hay: list[int]) -> bool:
    if not needle:
        return True
    nl = len(needle)
    for j in range(len(hay) - nl + 1):
        if hay[j : j + nl] == needle:
            return True
    return False


def _lz76_count(seq: list[int]) -> int:
    """Number of LZ76 phrases (production complexity)."""
    n = len(seq)
    i = 0
    c = 0
    while i < n:
        w = 0
        # longest prefix seq[i:i+w] that already occurred in seq[:i]
        while i + w < n and _is_sublist(seq[i : i + w + 1], seq[:i]):
            w += 1
        # phrase = w repeated chars + 1 novel char (or end-of-stream)
        i += w + 1
        c += 1
    return c


def lempel_ziv_complexity(token_stream: list[int]) -> LZComplexity:
    """Compute LZ76 complexity and a coarse structure class.

    Normalization: ``c_n / (n / log2(n))``. Combinatorial streams approach/beat
    1.0; highly repetitive streams fall well below 0.4.
    """
    seq = list(token_stream)
    n = len(seq)
    c_n = _lz76_count(seq)

    if n <= 1:
        normalized = 0.0
    else:
        normalized = c_n / (n / math.log2(n))

    if normalized > 0.9:
        structure_class = "combinatorial"
    elif normalized < 0.4:
        structure_class = "holistic"
    else:
        structure_class = "mixed"

    return LZComplexity(c_n=c_n, normalized=normalized, structure_class=structure_class)
