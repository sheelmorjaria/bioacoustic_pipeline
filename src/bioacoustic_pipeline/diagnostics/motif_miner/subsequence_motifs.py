"""DUSTED-style recurrent-subsequence motif discovery for animal vocalisations.

Adapts spoken-term-discovery (DUSTED) to *recurrent acoustic terms* in animal
vocalisations — NOT words. The transferable target is recurrent motifs / sub-call
phrases / sequence templates / vocalisation-internal units. Semantics is never
asserted: a discovered motif is a statistically tested recurring acoustic structure
that needs Track-C playback/context evidence to gain meaning.

Why this exists alongside ``token_motifs`` / ``acoustic_motifs``: those mine *exact*
cluster-ID n-grams. They are approximate at the single-unit level (fingerprint
clustering) but **exact at the sequence level**, so they *split* motif variants that
differ by one internal part — ``[7,10,0]`` vs ``[7,11,1]`` (a substituted unit) vs
``[7,10]`` (a dropped unit) vs ``[7,3,10,0]`` (an inserted unit). This module performs
**indel-tolerant, length-normalised, role-aware DTW** over fingerprint vectors so those
variants are recovered as ONE recurrent motif family.

Pipeline (constrained cascade — full all-pairs DTW is never run):

    HECB-typed sub-call units  →  window generation (L = min_n..max_n)
      →  cheap window signature (centroid ‖ first/last cluster ‖ role seq ‖ length)
      →  prefilter (centroid-cosine radius + length-ratio + role-compatible ends)
      →  length-normalised role-aware DTW  (only on surviving candidate pairs)
      →  motif graph (edge = DTW dist < threshold)  →  connected-component families
      →  per-motif metrics + null-tested enrichment

DTW cost (biologically grounded in the onset/body/offset positional-role result):

    substitution(i, j) = (1 − cosine(fp_i, fp_j))
                       + role_penalty · 𝟙[role_i ≠ role_j]
                       + relpos_lambda · |rel_mid_i − rel_mid_j|
    indel(role)        = indel_edge if role ∈ {onset, offset} else indel_body
    distance           = total_alignment_cost / max(len(a), len(b))

A fixed-length cosine baseline (``fixed_length_cosine_clusters``) is provided as an
ablation: it catches substitutions but, having no indels, misses length-mismatched
variants — the case DTW is designed to recover.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from ..acoustic_fingerprint.similarity import cosine, normalize_batch

# ─── roles ───────────────────────────────────────────────────────────────────
_EDGE_ROLES = {"onset", "offset", "only"}


def _roles_for_len(n: int) -> list[str]:
    """onset / body / offset / only by position index (mirrors build_vocalisation)."""
    if n <= 1:
        return ["only"]
    roles = ["body"] * n
    roles[0] = "onset"
    roles[-1] = "offset"
    return roles


def _rel_mids_for_len(n: int) -> list[float]:
    return [(i + 0.5) / n for i in range(n)]


def _indel_cost(role: str, indel_body: float, indel_edge: float) -> float:
    return indel_edge if role in _EDGE_ROLES else indel_body


# ─── data classes ────────────────────────────────────────────────────────────


@dataclass
class CallUnits:
    """One vocalisation's ordered internal units (sub-calls)."""

    fingerprints: list[np.ndarray]
    clusters: list[int] = field(default_factory=list)
    calltype: object | None = None
    caller: object | None = None
    file_id: str = ""
    call_idx: int = -1

    def __post_init__(self) -> None:
        if not self.clusters:
            self.clusters = [-1] * len(self.fingerprints)

    @property
    def n_units(self) -> int:
        return len(self.fingerprints)


@dataclass
class SubSequence:
    """A window of consecutive internal units (a candidate motif instance)."""

    fps: list[np.ndarray]
    clusters: list[int]
    roles: list[str]
    rel_mids: list[float]
    call_idx: int = -1
    calltype: object | None = None
    caller: object | None = None
    file_id: str = ""

    @property
    def length(self) -> int:
        return len(self.fps)

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(np.asarray(self.fps, dtype=np.float32), axis=0)

    @property
    def signature(self) -> tuple:
        """Cheap comparable signature: (first_cluster, last_cluster, role_tuple, length)."""
        return (self.clusters[0], self.clusters[-1], tuple(self.roles), self.length)


@dataclass
class SubsequenceMotif:
    """A discovered recurrent acoustic term (family of approximately-matching windows)."""

    motif_id: str = ""
    support: int = 0  # number of member windows
    n_calls: int = 0  # distinct vocalisations containing the motif
    n_callers: int = 0
    n_files: int = 0
    signature_pattern: tuple = ()  # most common cluster-tuple among members
    mean_role_sequence: tuple = ()  # most common role-tuple among members
    coverage: float = 0.0  # n_calls / total_calls
    cohesion: float = 0.0  # mean in-component edge DTW distance (lower = tighter)
    label_purity: float = 0.0  # fraction of members sharing the modal cluster-tuple
    calltype_enrichment: dict = field(default_factory=lambda: {"V": 0.0, "p": 1.0})
    caller_enrichment: dict = field(default_factory=lambda: {"V": 0.0, "p": 1.0})
    prototype: SubSequence | None = None
    interpretation: str = ""

    def as_dict(self) -> dict:
        return {
            "motif_id": self.motif_id,
            "support": self.support,
            "n_calls": self.n_calls,
            "n_callers": self.n_callers,
            "n_files": self.n_files,
            "signature_pattern": list(self.signature_pattern),
            "mean_role_sequence": list(self.mean_role_sequence),
            "coverage": round(self.coverage, 4),
            "cohesion": round(self.cohesion, 4),
            "label_purity": round(self.label_purity, 4),
            "calltype_V": round(self.calltype_enrichment["V"], 4),
            "calltype_p": round(self.calltype_enrichment["p"], 4),
            "caller_V": round(self.caller_enrichment["V"], 4),
            "caller_p": round(self.caller_enrichment["p"], 4),
            "interpretation": self.interpretation,
        }


# ─── windowing ───────────────────────────────────────────────────────────────


def window_subsequences(
    calls: Sequence[CallUnits], min_n: int = 2, max_n: int = 6
) -> list[SubSequence]:
    """All length-[min_n, max_n] contiguous windows across all calls."""
    windows: list[SubSequence] = []
    for call in calls:
        n = call.n_units
        if n < min_n:
            continue
        for L in range(min_n, min(max_n, n) + 1):
            roles_L = _roles_for_len(L)
            mids_L = _rel_mids_for_len(L)
            for s in range(n - L + 1):
                windows.append(
                    SubSequence(
                        fps=list(call.fingerprints[s : s + L]),
                        clusters=list(call.clusters[s : s + L]),
                        roles=list(roles_L),
                        rel_mids=list(mids_L),
                        call_idx=call.call_idx,
                        calltype=call.calltype,
                        caller=call.caller,
                        file_id=call.file_id,
                    )
                )
    return windows


def _normalize_calls(calls: Sequence[CallUnits]) -> list[CallUnits]:
    """Per-feature z-score all unit fingerprints across the corpus (fair cosine)."""
    all_fps = [np.asarray(fp, dtype=np.float32) for c in calls for fp in c.fingerprints]
    if not all_fps:
        return list(calls)
    normed = normalize_batch(np.asarray(all_fps, dtype=np.float32))
    out, idx = [], 0
    for c in calls:
        n = len(c.fingerprints)
        out.append(replace(c, fingerprints=[normed[idx + k] for k in range(n)]))
        idx += n
    return out


# ─── DTW ─────────────────────────────────────────────────────────────────────


def dtw_distance(
    a: SubSequence,
    b: SubSequence,
    *,
    indel_body: float = 0.25,
    indel_edge: float = 0.45,
    role_penalty: float = 0.15,
    relpos_lambda: float = 0.2,
    bandwidth: float = 0.4,
) -> float:
    """Length-normalised, role-aware DTW cost over fingerprint vectors ∈ [0, ~1+].

    Substitution = (1 − cosine) + role-mismatch penalty + relative-position penalty.
    Indel cost is role-aware: deleting an onset/offset unit (``indel_edge``) costs more
    than deleting a body unit (``indel_body``) — respects the onset/body/offset result.
    Normalised by ``max(len(a), len(b))``.

    ``bandwidth``: Sakoe-Chiba band (0–1). Limits |i/na − j/nb| < bandwidth, preventing
    DTW from stretching a short unit to match a long graded sweep — a major source of
    over-merging in continuous bioacoustics. Default 0.4 (generous); 1.0 = no band.
    """
    na, nb = a.length, b.length
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return 1.0
    inf = np.inf
    D = np.full((na + 1, nb + 1), inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        D[i, 0] = D[i - 1, 0] + _indel_cost(a.roles[i - 1], indel_body, indel_edge)
    for j in range(1, nb + 1):
        D[0, j] = D[0, j - 1] + _indel_cost(b.roles[j - 1], indel_body, indel_edge)
    for i in range(1, na + 1):
        # Sakoe-Chiba band: skip cells outside |i/na − j/nb| < bandwidth
        if bandwidth < 1.0:
            lo = max(1, int(nb * (i / na - bandwidth)))
            hi = min(nb, int(nb * (i / na + bandwidth)) + 1)
        else:
            lo, hi = 1, nb
        for j in range(lo, hi + 1):
            sub = max(0.0, 1.0 - cosine(a.fps[i - 1], b.fps[j - 1]))
            if a.roles[i - 1] != b.roles[j - 1]:
                sub += role_penalty
            sub += relpos_lambda * abs(a.rel_mids[i - 1] - b.rel_mids[j - 1])
            D[i, j] = min(
                D[i - 1, j] + _indel_cost(a.roles[i - 1], indel_body, indel_edge),
                D[i, j - 1] + _indel_cost(b.roles[j - 1], indel_body, indel_edge),
                D[i - 1, j - 1] + sub,
            )
    return max(0.0, float(D[na, nb]) / max(na, nb))


# ─── prefilter ───────────────────────────────────────────────────────────────


def _prefilter_pairs(
    windows: list[SubSequence],
    *,
    centroid_radius: float = 0.25,
    length_ratio: float = 1.5,
) -> list[tuple[int, int]]:
    """Candidate window pairs: centroid-cosine NN + length-ratio gate.

    Uses ``sklearn.neighbors.NearestNeighbors`` (cosine) on per-window centroids so
    full all-pairs DTW is never run. (Positional roles are length-deterministic —
    onset-first / offset-last for every window — so a role-end gate would be a no-op;
    the meaningful role logic lives inside DTW: role-mismatch penalty + role-aware indel.)
    """
    if len(windows) < 2:
        return []
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    cents = np.asarray([w.centroid for w in windows], dtype=np.float32)
    nn = NearestNeighbors(radius=centroid_radius, metric="cosine").fit(cents)
    graph = nn.radius_neighbors_graph(mode="connectivity").tocoo()
    pairs = []
    for i, j in zip(graph.row, graph.col):
        if j <= i:
            continue
        wa, wb = windows[i], windows[j]
        la, lb = wa.length, wb.length
        if max(la, lb) / max(min(la, lb), 1) > length_ratio:
            continue
        pairs.append((int(i), int(j)))
    return pairs


# ─── graph clustering + motif metrics ────────────────────────────────────────


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _enrichment(has: list[int], labels: list) -> dict:
    """Cramér's V + χ² p + ARI for binary motif-presence vs a categorical label.

    ARI (Adjusted Rand Index) corrects for chance and handles unbalanced categories
    better than V — reviewer-safe complementary metric for "association without semantics".
    """
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415
    from scipy.stats import chi2_contingency  # noqa: PLC0415

    ulabels = sorted(set(labels))
    if len(ulabels) < 2:
        return {"V": 0.0, "p": 1.0, "ari": 0.0}
    idx = {lab: k for k, lab in enumerate(ulabels)}
    table = np.zeros((2, len(ulabels)))
    for h, lab in zip(has, labels):
        table[int(bool(h)), idx[lab]] += 1
    if table.sum() < 10 or table.sum(axis=0).min() == 0:
        return {"V": 0.0, "p": 1.0, "ari": 0.0}
    try:
        chi2, p, _, _ = chi2_contingency(table)
        denom = table.sum() * (min(2, len(ulabels)) - 1)
        V = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
        # ARI: binary presence vs label partition (complementary to V)
        label_codes = [idx[lab] for lab in labels]
        ari = float(adjusted_rand_score([int(bool(h)) for h in has], label_codes))
        return {"V": min(V, 1.0), "p": float(p), "ari": ari}
    except Exception:  # noqa: BLE001
        return {"V": 0.0, "p": 1.0, "ari": 0.0}


def _build_motifs(
    windows: list[SubSequence],
    edges: list[tuple[int, int, float]],
    components: dict[int, list[int]],
    *,
    min_support: int,
    min_calls: int,
    total_calls: int,
    calltypes_by_call: dict[int, list],
    callers_by_call: dict[int, list],
) -> list[SubsequenceMotif]:
    motifs: list[SubsequenceMotif] = []
    for ci, members in sorted(components.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_support:
            continue
        member_calls = sorted({windows[m].call_idx for m in members if windows[m].call_idx >= 0})
        if len(member_calls) < min_calls:
            continue
        # internal edges for cohesion + prototype
        mset = set(members)
        in_edges = [(i, j, d) for (i, j, d) in edges if i in mset and j in mset]
        cohesion = float(np.mean([d for _, _, d in in_edges])) if in_edges else 0.0
        # prototype = member with min mean edge-distance to in-component neighbours
        nbr_dist: dict[int, list[float]] = defaultdict(list)
        for i, j, d in in_edges:
            nbr_dist[i].append(d)
            nbr_dist[j].append(d)
        prototype_idx = min(
            members, key=lambda m: (np.mean(nbr_dist[m]) if nbr_dist.get(m) else cohesion)
        )
        # signature / role patterns (most common cluster-tuple / role-tuple)
        cluster_tuples = Counter(tuple(windows[m].clusters) for m in members)
        role_tuples = Counter(tuple(windows[m].roles) for m in members)
        sig = cluster_tuples.most_common(1)[0][0]
        role_seq = role_tuples.most_common(1)[0][0]
        label_purity = cluster_tuples.most_common(1)[0][1] / len(members)
        # enrichment at the call level (binary presence vs all calls)
        has_calltype = [1 if ci_ in member_calls else 0 for ci_ in calltypes_by_call]
        ct_labels = [calltypes_by_call[c][0] for c in calltypes_by_call]
        has_caller = [1 if ci_ in member_calls else 0 for ci_ in callers_by_call]
        ca_labels = [callers_by_call[c][0] for c in callers_by_call]
        member_call_objs = [c for c in member_calls]
        n_callers = len({callers_by_call[c][0] for c in member_call_objs})
        n_files = len({windows[m].file_id for m in members})
        motifs.append(
            SubsequenceMotif(
                motif_id=f"subseq_{len(motifs)}",
                support=len(members),
                n_calls=len(member_calls),
                n_callers=n_callers,
                n_files=n_files,
                signature_pattern=sig,
                mean_role_sequence=role_seq,
                coverage=len(member_calls) / max(total_calls, 1),
                cohesion=cohesion,
                label_purity=label_purity,
                calltype_enrichment=_enrichment(has_calltype, ct_labels),
                caller_enrichment=_enrichment(has_caller, ca_labels),
                prototype=windows[prototype_idx],
                interpretation="recurrent acoustic term (DUSTED subsequence motif)",
            )
        )
    return motifs


def _hac_refine(
    windows: list[SubSequence],
    edges: list[tuple[int, int, float]],
    rough_ccs: dict[int, list[int]],
    dist_threshold: float,
    max_full_hac: int = 300,
) -> dict[int, list[int]]:
    """Refine rough CCs with average-linkage HAC (prevents single-linkage chaining).

    Single-linkage (connected components) merges via one noisy bridge edge. Average-
    linkage HAC considers the mean inter-cluster distance, so bridge edges don't chain.
    For CCs ≤ ``max_full_hac``: HAC on DTW distance submatrix. For larger CCs: HAC on
    centroid-cosine distances (fast approximation).
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    edge_lookup: dict[tuple[int, int], float] = {}
    for i, j, d in edges:
        edge_lookup[(min(i, j), max(i, j))] = d

    refined: dict[int, list[int]] = {}
    next_id = 0
    for members in rough_ccs.values():
        n = len(members)
        if n <= 3:
            refined[next_id] = members
            next_id += 1
            continue
        if n <= max_full_hac:
            # Build pairwise DTW distance matrix from verified edges
            dist_mat = np.full((n, n), 2.0, dtype=np.float64)
            np.fill_diagonal(dist_mat, 0.0)
            m_idx = {m: i for i, m in enumerate(members)}
            m_set = set(members)
            for i, j, d in edges:
                if i in m_set and j in m_set:
                    mi, mj = m_idx[i], m_idx[j]
                    dist_mat[mi, mj] = d
                    dist_mat[mj, mi] = d
        else:
            # Large CC → approximate with centroid-cosine distances (fast)
            cents = np.asarray([windows[m].centroid for m in members], dtype=np.float32)
            normed = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
            dist_mat = 1.0 - np.clip(normed @ normed.T, 0.0, 1.0)
            np.fill_diagonal(dist_mat, 0.0)
        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=dist_threshold, criterion="distance")
        for lab in set(labels.tolist()):
            refined[next_id] = [members[i] for i in range(n) if labels[i] == lab]
            next_id += 1
    return refined


def _discover(
    windows: list[SubSequence],
    *,
    dist_threshold: float,
    min_support: int,
    min_calls: int,
    total_calls: int,
    calltypes_by_call: dict,
    callers_by_call: dict,
    prefilter_kws: dict,
    dtw_kws: dict,
    clustering: str = "hac",
) -> list[SubsequenceMotif]:
    """Shared discovery core: prefilter → DTW-verify → graph → motifs.

    ``clustering``: ``'connected'`` = raw connected components (single-linkage, fast,
    original); ``'hac'`` = average-linkage HAC refinement (prevents single-linkage
    chaining that over-merges graded corpora; default).
    """
    cand = _prefilter_pairs(windows, **prefilter_kws)
    uf = _UnionFind(len(windows))
    edges: list[tuple[int, int, float]] = []
    for i, j in cand:
        d = dtw_distance(windows[i], windows[j], **dtw_kws)
        if d < dist_threshold:
            edges.append((i, j, d))
            uf.union(i, j)
    rough_ccs: dict[int, list[int]] = defaultdict(list)
    for w_idx in range(len(windows)):
        rough_ccs[uf.find(w_idx)].append(w_idx)
    if clustering == "hac":
        comp_map = _hac_refine(windows, edges, rough_ccs, dist_threshold)
    else:
        comp_map = rough_ccs
    return _build_motifs(
        windows,
        edges,
        comp_map,
        min_support=min_support,
        min_calls=min_calls,
        total_calls=total_calls,
        calltypes_by_call=calltypes_by_call,
        callers_by_call=callers_by_call,
    )


# ─── public discovery (DTW with indels) ──────────────────────────────────────


def discover_recurrent_subsequences(
    calls: Sequence[CallUnits],
    *,
    min_n: int = 2,
    max_n: int = 6,
    dist_threshold: float = 0.35,
    min_support: int = 3,
    min_calls: int = 2,
    centroid_radius: float = 0.25,
    length_ratio: float = 1.5,
    normalize: bool = True,
    indel_body: float = 0.25,
    indel_edge: float = 0.45,
    role_penalty: float = 0.15,
    relpos_lambda: float = 0.2,
    clustering: str = "hac",
) -> list[SubsequenceMotif]:
    """DUSTED-style DTW discovery of recurrent acoustic terms.

    Threshold tiers (tune against ``discovery_null``): ``<0.20`` near-duplicate,
    ``<0.35`` variant (default), ``<0.50`` loose family.
    """
    calls = list(calls)
    if normalize:
        calls = _normalize_calls(calls)
    windows = window_subsequences(calls, min_n, max_n)
    calltypes_by_call = {c.call_idx: [c.calltype] for c in calls if c.call_idx >= 0}
    callers_by_call = {c.call_idx: [c.caller] for c in calls if c.call_idx >= 0}
    return _discover(
        windows,
        dist_threshold=dist_threshold,
        min_support=min_support,
        min_calls=min_calls,
        total_calls=len({c.call_idx for c in calls if c.call_idx >= 0}),
        calltypes_by_call=calltypes_by_call,
        callers_by_call=callers_by_call,
        prefilter_kws=dict(centroid_radius=centroid_radius, length_ratio=length_ratio),
        dtw_kws=dict(
            indel_body=indel_body,
            indel_edge=indel_edge,
            role_penalty=role_penalty,
            relpos_lambda=relpos_lambda,
        ),
        clustering=clustering,
    )


# ─── baseline: fixed-length cosine (ablation) ────────────────────────────────


def _fixed_length_pairs(
    windows: list[SubSequence],
    *,
    centroid_radius: float,
    length_ratio: float,
) -> list[tuple[int, int]]:
    """Same-length windows only, gated exactly like the DTW prefilter."""
    cand = _prefilter_pairs(windows, centroid_radius=centroid_radius, length_ratio=length_ratio)
    return [(i, j) for (i, j) in cand if windows[i].length == windows[j].length]


def _fixed_length_distance(a: SubSequence, b: SubSequence) -> float:
    """Mean per-position cosine distance between same-length windows (no indels)."""
    return float(np.mean([1.0 - cosine(a.fps[k], b.fps[k]) for k in range(a.length)]))


def fixed_length_cosine_clusters(
    calls: Sequence[CallUnits],
    *,
    min_n: int = 2,
    max_n: int = 6,
    dist_threshold: float = 0.35,
    min_support: int = 3,
    min_calls: int = 2,
    centroid_radius: float = 0.25,
    normalize: bool = True,
) -> list[SubsequenceMotif]:
    """Baseline/ablation: same-length windows, mean per-position cosine distance.

    Catches substitutions but, with no indels, cannot recover length-mismatched motif
    variants — the case DTW is designed to catch.
    """
    calls = list(calls)
    if normalize:
        calls = _normalize_calls(calls)
    windows = window_subsequences(calls, min_n, max_n)
    calltypes_by_call = {c.call_idx: [c.calltype] for c in calls if c.call_idx >= 0}
    callers_by_call = {c.call_idx: [c.caller] for c in calls if c.call_idx >= 0}
    cand = _fixed_length_pairs(windows, centroid_radius=centroid_radius, length_ratio=1.0)
    uf = _UnionFind(len(windows))
    edges: list[tuple[int, int, float]] = []
    for i, j in cand:
        d = _fixed_length_distance(windows[i], windows[j])
        if d < dist_threshold:
            edges.append((i, j, d))
            uf.union(i, j)
    comp_map: dict[int, list[int]] = defaultdict(list)
    for w_idx in range(len(windows)):
        comp_map[uf.find(w_idx)].append(w_idx)
    return _build_motifs(
        windows,
        edges,
        comp_map,
        min_support=min_support,
        min_calls=min_calls,
        total_calls=len({c.call_idx for c in calls if c.call_idx >= 0}),
        calltypes_by_call=calltypes_by_call,
        callers_by_call=callers_by_call,
    )


# ─── baseline: exact n-grams (ablation) ──────────────────────────────────────


def exact_ngram_clusters(
    calls: Sequence[CallUnits],
    *,
    min_n: int = 2,
    max_n: int = 6,
    min_support: int = 3,
    min_calls: int = 2,
    normalize: bool = True,
) -> list[SubsequenceMotif]:
    """Baseline/ablation: windows match iff their cluster-ID tuples are identical.

    This is what ``token_motifs`` / ``acoustic_motifs`` effectively do — exact at the
    sequence level. Highest precision, lowest recall: splits every substituted/dropped/
    inserted variant into a separate family. Provided so the DTW method can be compared
    against the exact-n-gram ceiling/floor on identical metrics.
    """
    calls = list(calls)
    if normalize:
        calls = _normalize_calls(calls)
    windows = window_subsequences(calls, min_n, max_n)
    calltypes_by_call = {c.call_idx: [c.calltype] for c in calls if c.call_idx >= 0}
    callers_by_call = {c.call_idx: [c.caller] for c in calls if c.call_idx >= 0}
    groups: dict[tuple, list[int]] = defaultdict(list)
    for wi, w in enumerate(windows):
        groups[tuple(w.clusters)].append(wi)
    comp_map = {gid: idxs for gid, idxs in enumerate(groups.values())}
    return _build_motifs(
        windows,
        [],
        comp_map,
        min_support=min_support,
        min_calls=min_calls,
        total_calls=len({c.call_idx for c in calls if c.call_idx >= 0}),
        calltypes_by_call=calltypes_by_call,
        callers_by_call=callers_by_call,
    )


# ─── nulls + stability ───────────────────────────────────────────────────────


def _shuffle_call_order(calls: Sequence[CallUnits], rng: random.Random) -> list[CallUnits]:
    """Permute internal-unit order within each call (preserves the fp multiset)."""
    out = []
    for c in calls:
        order = list(range(c.n_units))
        rng.shuffle(order)
        out.append(
            replace(
                c,
                fingerprints=[c.fingerprints[k] for k in order],
                clusters=[c.clusters[k] for k in order],
            )
        )
    return out


def discovery_null(
    calls: Sequence[CallUnits],
    *,
    dist_threshold: float = 0.35,
    min_support: int = 3,
    min_calls: int = 2,
    n_perm: int = 50,
    seed: int = 0,
    discover_fn=None,
    **discover_kws,
) -> dict:
    """Order-shuffle null on TOTAL motif support (sensitive to motif strength).

    Shuffles internal-unit order within each call (preserves the fingerprint multiset,
    destroys sequence structure), re-discovers, and reports the observed total support's
    percentile against the null. Total support (not motif count) is the right statistic:
    a strong motif (support ~30) and a weak shuffled relic (support ~4) both count as one
    motif, but their support separates them decisively.
    """
    fn = discover_fn or discover_recurrent_subsequences

    def _total_support(c):
        return sum(
            m.support
            for m in fn(
                c,
                dist_threshold=dist_threshold,
                min_support=min_support,
                min_calls=min_calls,
                **discover_kws,
            )
        )

    observed = _total_support(calls)
    rng = random.Random(seed)
    null_counts = [_total_support(_shuffle_call_order(calls, rng)) for _ in range(n_perm)]
    null = np.asarray(null_counts, dtype=np.float64)
    ge = int(np.sum(null >= observed))
    p = (1 + ge) / (len(null) + 1)
    pct = float(np.searchsorted(np.sort(null), observed) / len(null) * 100) if len(null) else 100.0
    return {
        "observed_support": observed,
        "null_mean": float(null.mean()) if len(null) else 0.0,
        "null_std": float(null.std()) if len(null) else 0.0,
        "percentile": pct,
        "p": p,
    }


def _motif_key(motif: SubsequenceMotif) -> tuple:
    return tuple(motif.signature_pattern)


def bootstrap_stability(
    calls: Sequence[CallUnits],
    *,
    dist_threshold: float = 0.35,
    min_support: int = 3,
    min_calls: int = 2,
    n_boot: int = 20,
    top_k: int = 10,
    seed: int = 0,
    discover_fn=None,
    **discover_kws,
) -> float:
    """Resample calls with replacement, re-discover top-K motifs, mean pairwise Jaccard."""
    fn = discover_fn or discover_recurrent_subsequences
    calls = list(calls)
    rng = random.Random(seed)
    top_sets: list[set] = []
    for _ in range(n_boot):
        sample = [calls[k] for k in (rng.randrange(len(calls)) for _ in range(len(calls)))]
        motifs = fn(
            sample,
            dist_threshold=dist_threshold,
            min_support=min_support,
            min_calls=min_calls,
            **discover_kws,
        )
        top = {_motif_key(m) for m in sorted(motifs, key=lambda m: -m.support)[:top_k]}
        top_sets.append(top)
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = top_sets[i] | top_sets[j]
            if union:
                jaccards.append(len(top_sets[i] & top_sets[j]) / len(union))
    return float(np.mean(jaccards)) if jaccards else 0.0


# ─── markov surrogate null (preserves bigram transitions) ────────────────────


def _markov_surrogate(calls: Sequence[CallUnits], rng: random.Random) -> list[CallUnits]:
    """Generate 1st-order Markov-chain surrogate sequences (preserves bigram transitions).

    Unlike order-shuffle (preserves the per-call multiset), the Markov null generates
    entirely new cluster sequences from a corpus-fitted transition matrix. This preserves
    unit frequency + bigram transition probabilities but destroys trigram/4-gram structure,
    specifically testing whether higher-order sequences exceed what local transitions predict.
    """
    from dataclasses import replace as _replace

    transitions: dict[int, Counter] = defaultdict(Counter)
    starts: Counter = Counter()
    for c in calls:
        clusters = c.clusters
        if len(clusters) < 2:
            continue
        starts[clusters[0]] += 1
        for k in range(len(clusters) - 1):
            transitions[clusters[k]][clusters[k + 1]] += 1
    # Normalize transition probabilities
    trans_probs: dict[int, list[tuple[int, float]]] = {}
    for src, counts in transitions.items():
        total = sum(counts.values())
        trans_probs[src] = [(k, v / total) for k, v in counts.items()]
    # Normalize start probabilities
    total_starts = sum(starts.values())
    start_items = [(k, v / total_starts) for k, v in starts.items()]
    all_units = [k for k, _ in start_items]

    out = []
    for c in calls:
        clusters = c.clusters
        if len(clusters) < 2 or not all_units:
            out.append(c)
            continue
        n = len(clusters)
        seq = []
        # Start: sample from start distribution
        s = rng.choices(all_units, weights=[p for _, p in start_items], k=1)[0]
        seq.append(s)
        for _ in range(n - 1):
            src = seq[-1]
            if src in trans_probs:
                dests = [k for k, _ in trans_probs[src]]
                probs = [p for _, p in trans_probs[src]]
                seq.append(rng.choices(dests, weights=probs, k=1)[0])
            else:
                seq.append(rng.choices(all_units, weights=[p for _, p in start_items], k=1)[0])
        out.append(_replace(c, clusters=seq))
    return out


def markov_null(
    calls: Sequence[CallUnits],
    *,
    dist_threshold: float = 0.35,
    min_support: int = 3,
    min_calls: int = 2,
    n_perm: int = 50,
    seed: int = 0,
    discover_fn=None,
    **discover_kws,
) -> dict:
    """1st-order Markov-chain null: does motif support exceed what bigram transitions predict?

    Preserves unit frequency + bigram transition probabilities but destroys higher-order
    (trigram/4-gram) structure. Specifically tests whether discovered motifs carry signal
    beyond what local transitions predict — the right null for `[X,X]`-style repeat motifs
    that survive the order-shuffle null by construction.
    """
    fn = discover_fn or discover_recurrent_subsequences

    def _total_support(c):
        return sum(
            m.support
            for m in fn(
                c,
                dist_threshold=dist_threshold,
                min_support=min_support,
                min_calls=min_calls,
                **discover_kws,
            )
        )

    observed = _total_support(calls)
    rng = random.Random(seed)
    null_counts = [_total_support(_markov_surrogate(calls, rng)) for _ in range(n_perm)]
    null = np.asarray(null_counts, dtype=np.float64)
    ge = int(np.sum(null >= observed))
    p = (1 + ge) / (len(null) + 1)
    pct = float(np.searchsorted(np.sort(null), observed) / len(null) * 100) if len(null) else 100.0
    return {
        "observed_support": observed,
        "null_mean": float(null.mean()) if len(null) else 0.0,
        "null_std": float(null.std()) if len(null) else 0.0,
        "percentile": pct,
        "p": p,
    }


# ─── cosine-HAC clustering utility (alternative to k-means for V/NMI) ────────


def cosine_hac_tokens(vecs: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Cosine-affinity agglomerative clustering → token IDs (alternative to k-means).

    k-means assumes spherical clusters; cosine-HAC handles the elliptical structure of
    fingerprint spaces better. Use in `_recovery()` / `association_test()` as a drop-in
    replacement for k-means to get fairer V/NMI when the representation is cosine-based.
    """
    from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

    n = vecs.shape[0]
    if n < k:
        return np.zeros(n, dtype=int)
    ac = AgglomerativeClustering(n_clusters=min(k, n), metric="cosine", linkage="average")
    return ac.fit_predict(vecs)


def gmm_tokens(vecs: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """GMM clustering → token IDs (alternative to k-means for elliptical clusters).

    k-means assumes spherical clusters; GMM handles elliptical/correlated feature
    distributions better. Use alongside ``cosine_hac_tokens`` for fairer V/NMI when
    the representation has non-spherical cluster structure (pooled features, etc.).
    """
    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    n = vecs.shape[0]
    if n < k:
        return np.zeros(n, dtype=int)
    gmm = GaussianMixture(
        n_components=min(k, n), random_state=seed, covariance_type="diag", n_init=1, max_iter=100
    ).fit(vecs)
    return gmm.predict(vecs)


def statistics_pool(fingerprints: list[np.ndarray]) -> np.ndarray:
    """Statistical pooling: mean + std over a sub-call sequence (temporal signature).

    Mean-pooling washes out temporal voice signatures (RookID §18: whole-call beats
    HECB mean-pooled). Statistical pooling (mean+std) captures the *spread* of sub-call
    variation — a lightweight temporal-attention proxy that weights identity-discriminative
    sub-calls without learning attention weights. Returns a 2× concatenated vector.
    """
    if not fingerprints:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(fingerprints, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).astype(np.float32)


# ─── reporting ───────────────────────────────────────────────────────────────


def summarize_discovery(motifs: list[SubsequenceMotif], n_calls: int) -> dict:
    """Corpus-level summary of a discovered motif inventory."""
    if not motifs:
        return {
            "n_motifs": 0,
            "coverage": 0.0,
            "mean_calltype_V": 0.0,
            "mean_caller_V": 0.0,
            "mean_label_purity": 0.0,
            "wmean_label_purity": 0.0,
            "family_entropy": 0.0,
            "family_entropy_norm": 0.0,
            "top_family_support_fraction": 0.0,
        }
    total_support = sum(m.support for m in motifs) or 1
    supports = [m.support for m in motifs]
    probs = [s / total_support for s in supports]
    family_entropy = float(-sum(p * np.log2(p) for p in probs if p > 0))
    n = len(motifs)
    family_entropy_norm = float(family_entropy / np.log2(n)) if n > 1 else 0.0
    top_family_support_fraction = float(max(supports) / total_support)
    return {
        "n_motifs": n,
        "total_support": sum(m.support for m in motifs),
        "coverage": max(m.coverage for m in motifs),
        "mean_calltype_V": float(np.mean([m.calltype_enrichment["V"] for m in motifs])),
        "mean_caller_V": float(np.mean([m.caller_enrichment["V"] for m in motifs])),
        "frac_calltype_sig": float(np.mean([m.calltype_enrichment["p"] < 0.05 for m in motifs])),
        "mean_label_purity": float(np.mean([m.label_purity for m in motifs])),
        "wmean_label_purity": float(
            sum(m.label_purity * m.support for m in motifs) / total_support
        ),
        # diverse-inventory vs one-dominant-repeat diagnostics:
        #   high entropy/norm-entropy + low top-fraction = diverse (birdsong-like);
        #   low entropy + high top-fraction = degenerate, one dominant (marmoset-like).
        "family_entropy": family_entropy,
        "family_entropy_norm": family_entropy_norm,
        "top_family_support_fraction": top_family_support_fraction,
    }


def format_motif_table(motifs: list[SubsequenceMotif], top: int = 12) -> str:
    """Render the top motifs as a readable table (recurrent acoustic terms, NOT words)."""
    lines = [
        f"{'motif':>10} {'support':>7} {'n_calls':>7} {'n_callers':>9} {'cov':>5} "
        f"{'cohes':>6} {'purity':>6} {'ct_V':>5} {'ct_p':>7} {'ca_V':>5} {'signature':>16}",
        "-" * 99,
    ]
    for m in sorted(motifs, key=lambda x: -x.support)[:top]:
        lines.append(
            f"{m.motif_id:>10} {m.support:7d} {m.n_calls:7d} {m.n_callers:9d} "
            f"{m.coverage:5.3f} {m.cohesion:6.3f} {m.label_purity:6.3f} "
            f"{m.calltype_enrichment['V']:5.3f} "
            f"{m.calltype_enrichment['p']:7.4f} {m.caller_enrichment['V']:5.3f} "
            f"{str(list(m.signature_pattern)):>16}"
        )
    return "\n".join(lines)
