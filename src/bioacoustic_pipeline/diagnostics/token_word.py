"""Token<->word association — the human calibration / Semantic Ceiling test.

LibriSpeech is the CALIBRATION corpus: it has transcripts, so forced alignment gives
word spans, and we can ask what an *unsupervised* VQ-VAE token recovers when true
word labels exist. This module is the pure-statistics core (no audio, no torch):
temporal-overlap word assignment + token/word contingency metrics. The driver that
runs forced alignment + segmentation + tokenization lives in
``scripts/token_word_calibration.py``.

Scientific framing (the Semantic Ceiling): if a supervised linear probe on the CPC
latent recovers word identity far better than unsupervised VQ token<->word
consistency, then word information EXISTS in the representation but unsupervised
tokenization does NOT recover lexical identity — tokens are acoustic-similarity
classes, not words. Never say "token 42 means TOM"; say "token 42 is acoustically
enriched for segments aligned to TOM".

Pure stdlib. Unit-testable without audio.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

# A compact closed-class English function-word set (uppercased, with common
# LibriSpeech contractions). Everything else is treated as a content word. This is
# a coarse split — only used to report the two classes separately, not as a decision
# threshold.
FUNCTION_WORDS = frozenset(
    """A AN THE AND OR BUT IF THEN OF IN TO ON AT FOR WITH FROM BY ABOUT AS INTO
    OVER UNDER UP DOWN OUT OFF IS WAS ARE WERE BE BEEN BEING AM DO DOES DID DONE
    HAVE HAS HAD WOULD COULD SHOULD WILL WOULD CAN MAY MIGHT MUST SHALL NOT NO NOR
    I YOU HE SHE IT WE THEY ME HIM HER US THEM MY YOUR HIS ITS OUR THEIR THIS THAT
    THESE THOSE THERE HERE WHERE WHEN WHY HOW WHAT WHO WHOM WHICH SO VERY JUST ALSO
    TOO THAN BECAUSE WHILE THOUGH ALTHOUGH UNLESS ERE O OH AH WELL NOW STILL EVEN
    I'M YOU'RE HE'S SHE'S IT'S WE'RE THEY'RE I'VE YOU'VE WE'VE THEY'VE I'LL YOU'LL
    HE'LL SHE'LL IT'LL WE'LL THEY'LL I'D YOU'D HE'D SHE'D IT'D WE'D THEY'D DON'T
    DOESN'T DIDN'T WON'T WOULDN'T CAN'T COULDN'T SHOULDN'T ISN'T AREN'T WASN'T
    WEREN'T HASN'T HAVEN'T HADN'T THAT'S THERE'S HERE'S WHAT'S WHO'S HOW'S""".split()
)


def overlap_ms(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Temporal overlap of two intervals in ms (0 if disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_word_by_overlap(
    seg_start: float,
    seg_end: float,
    word_spans: Iterable[tuple[float, float, str]],
    *,
    policy: str = "strict",
    min_ratio: float = 0.5,
) -> tuple[str, float] | None:
    """Assign a segment to a word by temporal overlap.

    Args:
        word_spans: iterable of (start_ms, end_ms, word).
        policy: "midpoint" (A — word covering the segment midpoint), "max_overlap"
            (B — word with largest overlap), or "strict" (C — max-overlap accepted
            only if its ratio >= ``min_ratio`` AND it is the only word overlapping
            that much). ``strict`` is the headline policy; ``max_overlap`` is
            exploratory; ``midpoint`` suits short segments.
        min_ratio: overlap / segment_duration floor for the strict policy.

    Returns:
        (word, overlap_ratio) or None if no assignment under the policy.
    """
    seg_dur = max(seg_end - seg_start, 1e-6)
    spans = list(word_spans)
    if not spans:
        return None

    if policy == "midpoint":
        mid = (seg_start + seg_end) / 2.0
        for ws, we, w in spans:
            if ws <= mid <= we:
                return w, overlap_ms(seg_start, seg_end, ws, we) / seg_dur
        return None

    # max_overlap (B) and strict (C) both need the best-overlap word.
    best_w, best_ov = None, 0.0
    for ws, we, w in spans:
        ov = overlap_ms(seg_start, seg_end, ws, we)
        if ov > best_ov:
            best_ov, best_w = ov, w
    if best_w is None:
        return None
    ratio = best_ov / seg_dur

    if policy == "max_overlap":
        return best_w, ratio
    if policy == "strict":
        # Accept only if the best word covers >= min_ratio AND no OTHER word also
        # covers >= min_ratio (i.e. the segment is unambiguously inside one word).
        if ratio < min_ratio:
            return None
        n_covering = sum(
            1 for ws, we, _w in spans
            if overlap_ms(seg_start, seg_end, ws, we) / seg_dur >= min_ratio
        )
        return (best_w, ratio) if n_covering == 1 else None
    raise ValueError(f"unknown policy {policy!r} (use midpoint|max_overlap|strict)")


def is_function_word(word: str) -> bool:
    return word.upper() in FUNCTION_WORDS


# --------------------------------------------------------------------------- #
# Contingency + consistency metrics
# --------------------------------------------------------------------------- #


def build_token_word_table(records: Iterable[dict]) -> dict[int, Counter]:
    """token -> Counter(word) from records with 'token' and 'word' (word may be None)."""
    table: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        w = r.get("word")
        if w is None:
            continue
        table[int(r["token"])][w] += 1
    return table


def token_purity(table: dict[int, Counter]) -> dict[int, dict]:
    """Per-token: dominant word, its purity (dom/total), and total count."""
    out: dict[int, dict] = {}
    for tok, wc in table.items():
        total = sum(wc.values())
        if total == 0:
            continue
        top_word, top_count = wc.most_common(1)[0]
        out[tok] = {"top_word": top_word, "purity": top_count / total,
                    "count": total, "n_words": len(wc)}
    return out


def word_consistency(records: Iterable[dict]) -> dict[str, dict]:
    """Per-word: dominant token, its consistency (dom/total), and total count."""
    word_to_tokens: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        w = r.get("word")
        if w is None:
            continue
        word_to_tokens[w][int(r["token"])] += 1
    out: dict[str, dict] = {}
    for w, tc in word_to_tokens.items():
        total = sum(tc.values())
        top_token, top_count = tc.most_common(1)[0]
        out[w] = {"top_token": top_token, "consistency": top_count / total,
                  "count": total, "n_tokens": len(tc)}
    return out


def global_consistency(records: Iterable[dict]) -> float:
    """Micro consistency: sum(per-word dominant-token count) / total labeled segments.

    Dominated by frequent words; pair with :func:`macro_word_consistency`.
    """
    wc = word_consistency(records)
    total = sum(v["count"] for v in wc.values())
    if total == 0:
        return 0.0
    dominant = sum(round(v["consistency"] * v["count"]) for v in wc.values())
    return dominant / total


def macro_word_consistency(records: Iterable[dict], *, min_count: int = 5) -> float:
    """Mean per-word consistency over words seen >= ``min_count`` times."""
    wc = word_consistency(records)
    vals = [v["consistency"] for v in wc.values() if v["count"] >= min_count]
    return sum(vals) / len(vals) if vals else 0.0


def consistency_by_class(records: Iterable[dict], *, min_count: int = 5) -> dict[str, float]:
    """Macro word consistency split into content vs function words."""
    wc = word_consistency(records)
    out: dict[str, list[float]] = {"content": [], "function": []}
    for w, v in wc.items():
        if v["count"] < min_count:
            continue
        out["function" if is_function_word(w) else "content"].append(v["consistency"])
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in out.items()}


def chance_baseline(records: Iterable[dict]) -> float:
    """1 / n_distinct_words over labeled segments (the random-assignment floor)."""
    words = {r["word"] for r in records if r.get("word") is not None}
    return 1.0 / max(len(words), 1)
