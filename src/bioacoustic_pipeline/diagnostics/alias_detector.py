"""Phase 6: Functional Alias Detector — distributional equivalence, unsupervised.

Two VQ-VAE tokens that are ACOUSTICALLY distinct (different waveforms →
different tokens) can still be FUNCTIONALLY identical: they do the same job
in the sequence (e.g. a male "the" and a female "the" both precede a noun).
This module discovers those equivalences from token-stream statistics alone —
no supervised word labels — closing the Semantic Ceiling the right way: by
what tokens *do*, not what they *sound like*.

Method (distributional semantics, a la Harris 1954 "distributional hypothesis"):
  1. For every token t, build a context vector = PPMI profile of the tokens
     that follow it. PPMI = max(0, log P(following=v | t) / P(v)).
  2. Two tokens with near-identical following-token distributions have high
     cosine similarity → they are functional aliases.
  3. Cluster tokens above a cosine threshold (connected components over the
     similarity graph) into FunctionalAliasGroups.

Pure numpy. No torch / CPC. Word labels are never used for detection — only
optionally to *label* a discovered group after the fact.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FunctionalAliasGroup:
    """A cluster of acoustically-distinct but functionally-equivalent tokens."""

    group_id: int
    member_tokens: list[int]
    centroid_context: np.ndarray
    cohesion: float
    total_instances: int
    label: str | None = None

    @property
    def n_members(self) -> int:
        return len(self.member_tokens)


@dataclass
class FunctionalAliasReport:
    """Full output of the Functional Alias Detector."""

    token_ids: list[int]
    context_matrix: np.ndarray
    token_counts: dict[int, int]
    groups: list[FunctionalAliasGroup]
    singleton_tokens: list[int]
    similarity_threshold: float
    min_instances: int
    direction: str

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def n_merged(self) -> int:
        return sum(g.n_members for g in self.groups)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def group_of(self, token: int) -> FunctionalAliasGroup | None:
        for g in self.groups:
            if token in g.member_tokens:
                return g
        return None

    def summary(self) -> str:
        return (
            f"FunctionalAliasReport: {self.n_groups} groups "
            f"({self.n_merged}/{self.n_tokens} tokens merged), "
            f"{len(self.singleton_tokens)} singletons"
        )


def _as_sequences(token_stream) -> list[list[int]]:
    """Normalize input to a list of token sequences (no cross-sequence bigrams)."""
    if len(token_stream) == 0:
        return []
    first = token_stream[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return [list(int(t) for t in seq) for seq in token_stream]
    return [list(int(t) for t in token_stream)]


def build_context_matrix(
    token_stream,
    vocab: list[int] | None = None,
    direction: str = "following",
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """Build a PPMI context matrix (n_tokens × n_vocab) from a token stream.

    Each row is the Positive-PMI profile of token t over the tokens that
    neighbor it (default: follow it). PPMI = max(0, log P(ctx|t) / P(ctx)).

    No additive smoothing is applied: a (t, ctx) pair that never co-occurs
    contributes exactly 0. Smoothing would inject a shared positive floor into
    every row and inflate the cosine between truly-orthogonal distributions.

    Args:
        token_stream: flat list[int] OR list of per-sequence lists (per-file).
            Sequences are treated independently — no bigrams across boundaries.
        vocab: column vocabulary (sorted token ids). If None, inferred.
        direction: "following", "preceding", or "both" (sum of the two profiles).

    Returns:
        (context_matrix [n_tokens, n_vocab], token_ids, token_counts)
    """
    sequences = _as_sequences(token_stream)
    if direction not in ("following", "preceding", "both"):
        raise ValueError(f"direction must be following|preceding|both, got {direction!r}")

    # Build co-occurrence counts: cooc[t][v] = # times v is the directed neighbor of t
    cooc: dict[int, dict[int, float]] = {}
    for seq in sequences:
        n = len(seq)
        for i in range(n - 1):
            t, v = seq[i], seq[i + 1]
            cooc.setdefault(t, {})
            if direction == "following":
                cooc[t][v] = cooc[t].get(v, 0.0) + 1.0
            elif direction == "preceding":
                cooc.setdefault(v, {})
                cooc[v][t] = cooc[v].get(t, 0.0) + 1.0
            else:  # both
                cooc[t][v] = cooc[t].get(v, 0.0) + 1.0
                cooc.setdefault(v, {})
                cooc[v][t] = cooc[v].get(t, 0.0) + 1.0

    token_ids = sorted(cooc.keys())
    if vocab is None:
        vocab = sorted({v for nbrs in cooc.values() for v in nbrs} | set(token_ids))
    vocab_index = {v: j for j, v in enumerate(vocab)}

    n_tokens = len(token_ids)
    n_vocab = len(vocab)
    counts = np.zeros((n_tokens, n_vocab), dtype=np.float64)
    for i, t in enumerate(token_ids):
        for v, c in cooc[t].items():
            if v in vocab_index:
                counts[i, vocab_index[v]] = c

    token_counts = {t: int(counts[i].sum()) for i, t in enumerate(token_ids)}

    # P(ctx | t): raw conditional (0 where t never precedes ctx).
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard (tokens always have >=1 neighbor here)
    cond = counts / row_sums
    # P(ctx) marginal over the directed-neighbor mass (always > 0 for vocab ctx).
    col_mass = counts.sum(axis=0)
    total = col_mass.sum()
    total = total if total > 0 else 1.0
    p_ctx = col_mass / total
    # PPMI = max(0, log P(ctx|t) / P(ctx)). log(0) -> -inf -> clipped to 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(cond) - np.log(p_ctx)[None, :]
    ppmi = np.maximum(pmi, 0.0)
    ppmi[~np.isfinite(ppmi)] = 0.0
    return ppmi, token_ids, token_counts


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows (token context vectors)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / (norms + 1e-12)
    return unit @ unit.T


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_functional_aliases(
    context_matrix: np.ndarray,
    token_ids: list[int],
    token_counts: dict[int, int],
    similarity_threshold: float = 0.85,
    min_instances: int = 5,
) -> tuple[list[FunctionalAliasGroup], list[int]]:
    """Cluster tokens into functional alias groups by context-vector cosine.

    Connected components over edges where cosine > similarity_threshold,
    restricted to tokens seen at least min_instances times (rare tokens are
    noise, not stable functional classes).

    Returns:
        (groups, singleton_tokens). A singleton is an eligible (>= min_instances)
        token that did not merge with any other token.
    """
    n = len(token_ids)
    eligible = [
        i for i, t in enumerate(token_ids)
        if token_counts.get(t, 0) >= min_instances
    ]
    if len(eligible) < 2:
        return [], eligible_keep_singletons(token_ids, token_counts, min_instances)

    sim = cosine_similarity_matrix(context_matrix)
    uf = _UnionFind(n)
    for ai in range(len(eligible)):
        for bi in range(ai + 1, len(eligible)):
            i, j = eligible[ai], eligible[bi]
            if sim[i, j] >= similarity_threshold:
                uf.union(i, j)

    # Collect components (size >= 2 → group; size 1 → singleton)
    comps: dict[int, list[int]] = {}
    for i in eligible:
        comps.setdefault(uf.find(i), []).append(i)

    groups: list[FunctionalAliasGroup] = []
    singletons: list[int] = []
    for members in comps.values():
        if len(members) >= 2:
            ids = [token_ids[i] for i in members]
            centroid = context_matrix[members].mean(axis=0)
            sub = sim[np.ix_(members, members)]
            if len(members) > 1:
                upper = sub[np.triu_indices_from(sub, k=1)]
                cohesion = float(upper.mean())
            else:
                cohesion = 1.0
            total = sum(token_counts[t] for t in ids)
            groups.append(
                FunctionalAliasGroup(
                    group_id=len(groups),
                    member_tokens=sorted(ids),
                    centroid_context=centroid,
                    cohesion=cohesion,
                    total_instances=total,
                )
            )
        else:
            singletons.append(token_ids[members[0]])

    groups.sort(key=lambda g: -g.total_instances)
    for idx, g in enumerate(groups):
        g.group_id = idx
    return groups, sorted(singletons)


def eligible_keep_singletons(token_ids, token_counts, min_instances):
    return sorted(t for t in token_ids if token_counts.get(t, 0) >= min_instances)


def detect_functional_aliases(
    token_stream,
    similarity_threshold: float = 0.85,
    min_instances: int = 5,
    direction: str = "following",
    vocab: list[int] | None = None,
) -> FunctionalAliasReport:
    """Full pipeline: token stream → context vectors → alias groups → report."""
    context_matrix, token_ids, token_counts = build_context_matrix(
        token_stream, vocab=vocab, direction=direction
    )
    groups, singletons = group_functional_aliases(
        context_matrix, token_ids, token_counts,
        similarity_threshold=similarity_threshold,
        min_instances=min_instances,
    )
    return FunctionalAliasReport(
        token_ids=token_ids,
        context_matrix=context_matrix,
        token_counts=token_counts,
        groups=groups,
        singleton_tokens=singletons,
        similarity_threshold=similarity_threshold,
        min_instances=min_instances,
        direction=direction,
    )


# --------------------------------------------------------------------------- #
# Speaker-controlled alias null
# --------------------------------------------------------------------------- #


def alias_density(
    token_stream,
    similarity_threshold: float = 0.80,
    min_instances: int = 10,
    direction: str = "following",
) -> float:
    """Fraction of eligible tokens merged into functional-alias groups.

    Eligible = tokens seen >= ``min_instances`` times (singletons + merged). This is
    the statistic the permutation nulls below reshuffle. Matches the density the
    verdict engine consumes (orchestration.synthesize_signals).
    """
    rep = detect_functional_aliases(
        token_stream, similarity_threshold=similarity_threshold,
        min_instances=min_instances, direction=direction,
    )
    eligible = len(rep.singleton_tokens) + rep.n_merged
    return rep.n_merged / max(eligible, 1)


def shuffle_sequences_within_group(sequences, group, rng) -> list[list[int]]:
    """Cross-file shuffle restricted to bouts sharing a group value (e.g. Emitter).

    Preserves each group's token multiset + every bout length, but destroys
    sequential structure within the group. Used so a positive alias result must
    survive a speaker (or context) stratified null — preventing "different speakers
    produce acoustically different variants" from being misread as "different tokens
    fill the same functional role".
    """
    out = [list(s) for s in sequences]
    group = list(group)
    for g in dict.fromkeys(group):  # unique groups, order-stable
        idx = [i for i, gg in enumerate(group) if gg == g]
        if len(idx) < 2:
            continue
        lengths = [len(out[i]) for i in idx]
        pool = [t for i in idx for t in out[i]]
        rng.shuffle(pool)
        k = 0
        for i, ln in zip(idx, lengths):
            out[i] = pool[k : k + ln]
            k += ln
    return out


def controlled_alias_tests(
    sequences,
    *,
    emitter=None,
    context=None,
    similarity_threshold: float = 0.80,
    min_instances: int = 10,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Alias density vs a family of structure-destroying nulls.

    A positive alias result is reported only if it survives these controls:
      * ``within_file``   — permute tokens inside each bout (baseline sequence null)
      * ``frequency``     — cross-file/unigram shuffle (preserves global token freq)
      * ``within_emitter``— shuffle within Emitter bouts (speaker confound)
      * ``within_context``— shuffle within Context bouts (behavioral-context confound)

    Returns ``{null_name: NullTestResult}`` (only controls whose grouping is supplied
    are included; within_file + frequency always run).

    Raises nothing on degenerate input — returns whatever permutation_test yields.
    """
    from bioacoustic_pipeline.diagnostics.null_models import (  # noqa: PLC0415
        cross_file_shuffle,
        permutation_test,
        within_file_shuffle,
    )

    stat = lambda seqs: alias_density(  # noqa: E731
        seqs, similarity_threshold=similarity_threshold, min_instances=min_instances
    )
    results: dict[str, object] = {}

    results["within_file"] = permutation_test(
        sequences, stat, within_file_shuffle, n_permutations=n_permutations, seed=seed,
        null_type="within_file",
    )
    results["frequency"] = permutation_test(
        sequences, stat, cross_file_shuffle, n_permutations=n_permutations, seed=seed,
        null_type="frequency",
    )
    if emitter is not None:
        em = list(emitter)
        results["within_emitter"] = permutation_test(
            sequences, stat,
            lambda seqs, rng, _em=em: shuffle_sequences_within_group(seqs, _em, rng),
            n_permutations=n_permutations, seed=seed, null_type="within_emitter",
        )
    if context is not None:
        ctx = list(context)
        results["within_context"] = permutation_test(
            sequences, stat,
            lambda seqs, rng, _ctx=ctx: shuffle_sequences_within_group(seqs, _ctx, rng),
            n_permutations=n_permutations, seed=seed, null_type="within_context",
        )
    return results
