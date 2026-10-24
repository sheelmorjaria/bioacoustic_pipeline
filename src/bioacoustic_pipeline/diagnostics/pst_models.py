"""PST M0/M1/M2 nested model comparison — held-out NLL endpoint (§46).

Motivated by the near-zero CPC history-interaction contrast (T_y ~ 0): does a
caller-specific DISCRETE transition model improve held-out prediction beyond
a shared PST with a caller token bias? M0 = shared PST; M1 = M0 + caller
token bias (rebalanced unigrams); M2 = caller transitions with backoff to M1.
U_y = NLL(M0) - NLL(M1); S_y = NLL(M1) - NLL(M2). This is cross-method
triangulation of T_y, NOT independent replication.

Design: bout-safe counting (contexts never cross bouts); contexts retained
only when they beat their suffix (depth is a cap, not a target); smoothed
probabilities everywhere so unseen transitions give finite NLL; fit on train
bouts, score on held-out bouts only.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import math
from collections import Counter

_ALPHA = 0.5  # add-alpha smoothing
_MARGIN = 1e-12  # strict-improvement margin for context retention


def _smooth(counts: Counter, n_symbols: int, alpha: float = _ALPHA) -> dict:
    total = sum(counts.values()) + alpha * n_symbols
    return {s: (counts.get(s, 0) + alpha) / total for s in range(n_symbols)}


class PSTModel:
    """Shared PST with suffix backoff over bout-safe contexts."""

    def __init__(self, contexts: dict, n_symbols: int, max_depth: int):
        self.contexts = contexts  # {tuple: Counter(next)} RETAINED contexts + root ()
        self.n_symbols = n_symbols
        self.max_depth = max_depth

    # ---- queries ----
    def _match(self, context):
        """Longest retained suffix of ``context`` (root () always matches)."""
        for k in range(min(len(context), self.max_depth), -1, -1):
            suffix = tuple(context[-k:]) if k else ()
            if suffix in self.contexts:
                return suffix
        return ()

    def predict_proba(self, *, context, next_token) -> float:
        probs = _smooth(self.contexts[self._match(list(context))], self.n_symbols)
        return probs[int(next_token)]

    def entropy_rate(self) -> float:
        """Usage-weighted conditional entropy over depth-1 retained contexts."""
        total_usg = sum(sum(c.values()) for ctx, c in self.contexts.items() if len(ctx) == 1)
        if total_usg == 0:
            return 0.0
        acc = 0.0
        for ctx, counts in self.contexts.items():
            if len(ctx) != 1:
                continue
            p = _smooth(counts, self.n_symbols)
            usg = sum(counts.values())
            acc += usg * -sum(v * math.log(v) for v in p.values())
        return acc / total_usg

    def _cond_entropy(self, ctx: tuple) -> float:
        p = _smooth(self.contexts[ctx], self.n_symbols)
        return -sum(v * math.log(v) for v in p.values())

    def n_contexts_improving_over_suffix(self, depth_min: int = 1) -> int:
        n = 0
        for ctx in self.contexts:
            if len(ctx) < max(depth_min, 1):
                continue
            suffix = ctx[1:]  # depth-1 contexts compare against the root ()
            if suffix not in self.contexts:
                continue
            if self._cond_entropy(ctx) < self._cond_entropy(suffix) - _MARGIN:
                n += 1
        return n

    @property
    def n_significant_nodes(self) -> int:
        return sum(1 for ctx in self.contexts if len(ctx) >= 1)

    def all_observed_transitions(self) -> set:
        return {
            (ctx[-1], nxt)
            for ctx, counts in self.contexts.items()
            if len(ctx) == 1
            for nxt in counts
        }


def fit_shared_pst(sequences, *, max_depth: int, n_clusters: int) -> PSTModel:
    """Count bout-safe contexts; retain a context when it beats its suffix."""
    raw: dict = {(): Counter()}
    for seq in sequences:
        s = list(seq)
        for i, tok in enumerate(s):
            raw[()][tok] += 1
        for d in range(1, max_depth + 1):
            for i in range(d, len(s)):
                ctx = tuple(s[i - d : i])
                raw.setdefault(ctx, Counter())[s[i]] += 1
    # retention: root always; depth d kept if entropy < suffix entropy
    kept = {(): raw[()]}
    for ctx in sorted((c for c in raw if c), key=len):
        suffix = ctx[1:]
        if suffix not in kept:
            continue  # only grow from retained suffixes (PST property)

        def ent(c):
            p = _smooth(raw[c], n_clusters)
            return -sum(v * math.log(v) for v in p.values())

        if ent(ctx) < ent(suffix) - _MARGIN or len(ctx) == 1:
            kept[ctx] = raw[ctx]
    return PSTModel(kept, n_clusters, max_depth)


class BiasedPST:
    """M1: shared PST * caller token bias (renormalized)."""

    def __init__(self, base: PSTModel, bias: dict):
        self.base = base
        self.bias = bias  # {token: prob} caller unigram (smoothed)

    def predict_proba(self, *, context, next_token) -> float:
        p0 = self.base.predict_proba(context=context, next_token=next_token)
        b = self.bias[int(next_token)]
        score = p0 * b
        norm = sum(
            self.base.predict_proba(context=context, next_token=t) * self.bias[t]
            for t in range(self.base.n_symbols)
        )
        return score / norm if norm > 0 else p0


def fit_static_bias_model(m0: PSTModel, caller_train) -> BiasedPST:
    uni = Counter()
    for seq in caller_train:
        uni.update(seq)
    n = m0.n_symbols
    total = sum(uni.values()) + _ALPHA * n
    bias = {t: (uni.get(t, 0) + _ALPHA) / total for t in range(n)}
    return BiasedPST(m0, bias)


class CallerTransitionModel:
    """M2: caller transitions with backoff to M1 (never worse than a step)."""

    def __init__(self, m1: BiasedPST, caller_ctx: dict, max_depth: int):
        self.m1 = m1
        self.caller_ctx = caller_ctx  # {tuple: Counter}
        self.max_depth = max_depth

    def predict_proba(self, *, context, next_token) -> float:
        for k in range(min(len(context), self.max_depth), 0, -1):
            ctx = tuple(context[-k:])
            if ctx in self.caller_ctx and sum(self.caller_ctx[ctx].values()) >= 3:
                p = _smooth(self.caller_ctx[ctx], self.m1.base.n_symbols)
                return p[int(next_token)]
        return self.m1.predict_proba(context=context, next_token=next_token)


def fit_caller_transition_model(m0: PSTModel, caller_train, *, max_depth: int):
    m1 = fit_static_bias_model(m0, caller_train)
    ctx: dict = {}
    for seq in caller_train:
        s = list(seq)
        for d in range(1, max_depth + 1):
            for i in range(d, len(s)):
                key = tuple(s[i - d : i])
                ctx.setdefault(key, Counter())[s[i]] += 1
    return CallerTransitionModel(m1, ctx, max_depth)


def score_heldout_nll(model, sequences) -> float:
    """Mean per-token negative log-likelihood under ``model`` (bout-safe)."""
    total, n = 0.0, 0
    for seq in sequences:
        s = list(seq)
        for i, tok in enumerate(s):
            p = model.predict_proba(context=s[:i], next_token=tok)
            total -= math.log(max(p, 1e-300))
            n += 1
    return total / max(n, 1)


def _perm_p_value(observed: float, perm_values) -> float:
    """Phipson–Smyth corrected p: (1 + #{null >= obs}) / (B + 1)."""
    perms = list(perm_values)
    return (1.0 + sum(1 for v in perms if v >= observed)) / (len(perms) + 1.0)
