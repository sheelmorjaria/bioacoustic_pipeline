"""Transition network topology — finite-state-machine view of vocal sequences.

Architecture v1.0 "alternative structure" technique. Treat the vocalization as a directed graph:
nodes = sub-call token IDs (k-means cluster IDs from the fingerprint trajectory), directed edges =
observed transitions (weighted by count). Network topology metrics then quantify the repertoire's
"shape" — a sparse hub-and-spoke (infant marmoset: few nodes, one dominant cycle) vs a richly
modular network (Bengalese: many communities / phrases the bird moves between). Visually intuitive
and cross-species comparable; loses continuous acoustic similarity (transitions are weighted counts).

`networkx` is imported lazily (``available``/``load_error`` per convention). Firewall-clean: tokens
are acoustic-similarity cluster IDs, NOT learned semantics. NOT meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from collections import Counter

try:  # lazy — stays importable without the extra
    import networkx as nx  # noqa: F401

    _AVAILABLE = True
    _ERR = ""
except Exception as exc:  # noqa: BLE001
    nx = None
    _AVAILABLE = False
    _ERR = repr(exc)

available = _AVAILABLE
load_error = _ERR


def build_transition_graph(sequences: list[list[int]]):
    """Token sequences → weighted directed transition graph (networkx.DiGraph). Edges weighted by count."""
    if not _AVAILABLE:
        return None
    g = nx.DiGraph()
    trans: Counter = Counter()
    for seq in sequences:
        s = list(seq)
        for a, b in zip(s[:-1], s[1:]):
            trans[(a, b)] += 1
    for (a, b), w in trans.items():
        g.add_edge(a, b, weight=w)
    return g


def transition_network_metrics(sequences: list[list[int]]) -> dict:
    """Topology metrics of the transition graph built from token sequences.

    Returns: n_nodes, n_edges, density, mean_degree, clustering (directed, unweighted),
        n_components (weakly connected), modularity_communities (greedy on the weighted undirected
        projection), mean_path_length (harmonic, on the largest weakly connected component).
    """
    if not _AVAILABLE:
        return {"available": False, "load_error": _ERR}
    g = build_transition_graph(sequences)
    if g is None or g.number_of_nodes() == 0:
        return {
            "available": True,
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "mean_degree": 0.0,
            "clustering": 0.0,
            "n_components": 0,
            "modularity_communities": 0,
            "mean_path_length": 0.0,
        }
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = nx.density(g)
    mean_degree = (sum(dict(g.in_degree()).values()) + sum(dict(g.out_degree()).values())) / max(
        n_nodes, 1
    )
    clustering = nx.average_clustering(g.to_undirected()) if n_nodes > 1 else 0.0
    n_comp = nx.number_weakly_connected_components(g)
    # modularity: greedy communities on the weighted undirected projection
    n_communities = 0
    try:
        ug = g.to_undirected()
        if ug.number_of_edges() > 0:
            comms = nx.community.greedy_modularity_communities(ug, weight="weight")
            n_communities = len(comms)
    except Exception:  # noqa: BLE001 — older networkx / singleton graph
        n_communities = n_comp
    # mean harmonic path length on the largest weakly connected component
    mpl = 0.0
    try:
        largest = max(nx.weakly_connected_components(g), key=len)
        sub = g.subgraph(largest)
        if sub.number_of_nodes() > 1:
            mpl = nx.average_shortest_path_length(sub.to_undirected())
    except Exception:  # noqa: BLE001
        mpl = 0.0
    return {
        "available": True,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(density),
        "mean_degree": float(mean_degree),
        "clustering": float(clustering),
        "n_components": int(n_comp),
        "modularity_communities": int(n_communities),
        "mean_path_length": float(mpl),
    }
