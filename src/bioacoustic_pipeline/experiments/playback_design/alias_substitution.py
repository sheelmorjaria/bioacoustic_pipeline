"""Functional-alias substitution variants (Phase 1, token-level).

Tests whether functionally-equivalent tokens (``FunctionalAliasGroup.member_tokens``) are
behaviorally interchangeable: swap one token in a motif for an alias-group member, and
contrast with a non-alias control (a token from a *different* group). Token-level only —
the stimulus builder maps the resulting pattern to exemplar audio.

This is a falsifiable test of the alias detector, NOT a claim that the tokens "mean the same
thing" (Semantic Ceiling holds — equivalence is distributional/functional, not semantic).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def _group_of(alias_report, token):
    g = alias_report.group_of(int(token))
    return g.group_id if g is not None else None


def alias_substitute_pattern(pattern, alias_report, *, position: int | None = None, seed: int = 0):
    """Replace one token with a same-group alias member.

    Returns ``(new_pattern, info)`` where info = ``{position, original, substituted,
    group_id}``, or ``(None, None)`` if no position in ``pattern`` has an alias alternative.
    """
    rng = np.random.default_rng(seed)
    p = [int(t) for t in pattern]
    # positions eligible for substitution: token has a group with >1 member
    eligible = [
        i
        for i, t in enumerate(p)
        if (
            _group_of(alias_report, t) is not None
            and len(alias_report.group_of(t).member_tokens) > 1
        )
    ]
    if not eligible:
        return None, None
    pos = position if position in eligible else int(rng.choice(eligible))
    grp = alias_report.group_of(p[pos])
    others = [int(t) for t in grp.member_tokens if int(t) != p[pos]]
    if not others:
        return None, None
    sub = int(rng.choice(others))
    new = list(p)
    new[pos] = sub
    return new, {"position": pos, "original": p[pos], "substituted": sub, "group_id": grp.group_id}


def nonalias_control_pattern(
    pattern, alias_report, vocab, *, position: int | None = None, seed: int = 0
):
    """Replace one token with a token from a DIFFERENT alias group (or a singleton).

    The control: a structurally-similar substitution that is NOT a functional alias, so a
    matching behavioral response would weaken the alias-equivalence hypothesis. Returns
    ``(None, None)`` if no suitable control token exists.
    """
    rng = np.random.default_rng(seed)
    p = [int(t) for t in pattern]
    pos = position if position is not None else int(rng.integers(0, len(p)))
    grp_id = _group_of(alias_report, p[pos])
    # candidates: different group, or a singleton (in vocab but no group)
    candidates = []
    for t in vocab:
        gi = _group_of(alias_report, int(t))
        if int(t) == p[pos]:
            continue
        if gi is None or gi != grp_id:
            candidates.append(int(t))
    if not candidates:
        return None, None
    sub = int(rng.choice(candidates))
    new = list(p)
    new[pos] = sub
    return new, {"position": pos, "original": p[pos], "substituted": sub, "is_alias": False}
