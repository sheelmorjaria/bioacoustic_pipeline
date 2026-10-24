"""Dyadic eligibility — emitter x recipient x context support audit (P2).

Before any recipient-conditioned claim: which emitters have comparable,
NON-CONFOUNDED recipient data across the context contrast? Disjoint
recipient sets across contexts = perfect recipient-context confound (the
recipient effect and the context effect are inseparable by design); missing
either context's support = not evaluable. Sentinel addressees (code 0,
20% of identified-emitter bouts) are excluded like emitter sentinels.

Keeps the three dyadic questions separable: recipient variation (Q1),
pair structure (Q2), temporal response dependence (Q3 — the T_y track).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

SENTINEL_ADDRESSEE = 0


def build_dyadic_support_matrix(annotations) -> dict:
    """Bout counts per (emitter, addressee, context, recording) cell.

    Sentinel addressees (0) are dropped; the recording key is the
    annotation's file (per-bout corpus) or recording group when present.
    """
    matrix: dict = {}
    for a in annotations:
        if a.addressee == SENTINEL_ADDRESSEE:
            continue
        rec = a.recording_group or a.file_name
        key = (a.emitter, a.addressee, a.context, rec)
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


def check_recipient_evaluability(
    matrix: dict,
    *,
    emitter,
    c1,
    c2,
    min_bouts: int = 5,
) -> dict:
    """Can this emitter support a recipient contrast across c1 vs c2?

    Requires >= ``min_bouts`` in BOTH contexts and OVERLAPPING recipient
    sets (disjoint sets = the confound: recipient and context vary together
    perfectly, so no design separates them).
    """
    recipient_counts: dict = {}
    for (e, r, c, _f), count in matrix.items():
        if e == emitter:
            recipient_counts[(r, c)] = recipient_counts.get((r, c), 0) + count
    total_c1 = sum(v for (r, c), v in recipient_counts.items() if c == c1)
    total_c2 = sum(v for (r, c), v in recipient_counts.items() if c == c2)
    if total_c1 < min_bouts or total_c2 < min_bouts:
        return {
            "evaluable": False,
            "reason": "missing_context_data",
            "n_c1": total_c1,
            "n_c2": total_c2,
        }
    recipients_c1 = {r for (r, c) in recipient_counts if c == c1}
    recipients_c2 = {r for (r, c) in recipient_counts if c == c2}
    if recipients_c1.isdisjoint(recipients_c2):
        return {
            "evaluable": False,
            "reason": "recipient_context_confound",
            "n_c1": total_c1,
            "n_c2": total_c2,
            "recipients_c1": sorted(recipients_c1),
            "recipients_c2": sorted(recipients_c2),
        }
    return {
        "evaluable": True,
        "reason": "adequate_support",
        "n_c1": total_c1,
        "n_c2": total_c2,
        "recipients_c1": sorted(recipients_c1),
        "recipients_c2": sorted(recipients_c2),
        "shared_recipients": sorted(recipients_c1 & recipients_c2),
    }


def audit_dyadic_support(annotations, *, c1, c2, min_bouts: int = 5) -> dict:
    """Full emitter-level audit for the context contrast."""
    matrix = build_dyadic_support_matrix(annotations)
    emitters = sorted({e for (e, _r, _c, _f) in matrix})
    results = []
    for e in emitters:
        res = check_recipient_evaluability(matrix, emitter=e, c1=c1, c2=c2, min_bouts=min_bouts)
        results.append(
            {
                "emitter": e,
                **{k: v for k, v in res.items() if k != "evaluable"},
                "evaluable": res["evaluable"],
            }
        )
    n_ok = sum(1 for r in results if r["evaluable"])
    return {
        "contexts": [c1, c2],
        "min_bouts": min_bouts,
        "n_emitters_with_addressees": len(emitters),
        "n_evaluable": n_ok,
        "results": results,
        "framing": "dyadic support audit — data eligibility only, no inference; NOT meaning",
    }


def exposure_overlap(annotations, *, emitter, c1, c2):
    """O_e = sum_r min[P(r|e,c1), P(r|e,c2)] — SUBSTANTIAL shared exposure.

    Identity overlap (shared recipient IDs) is necessary but not sufficient:
    a 99/1 recipient asymmetry shares IDs with almost no controlled exposure.
    Returns (O_e, (P(.|c1), P(.|c2)) dicts), or (None, (dicts)) when either
    context lacks non-sentinel support. Sentinel addressees (0) excluded.
    """
    counts: dict = {c1: {}, c2: {}}
    for a in annotations:
        if a.emitter != emitter or a.addressee == SENTINEL_ADDRESSEE:
            continue
        if a.context in counts:
            counts[a.context][a.addressee] = counts[a.context].get(a.addressee, 0) + 1
    dists = {}
    for c, per in counts.items():
        total = sum(per.values())
        dists[c] = {r: n / total for r, n in per.items()} if total > 0 else {}
    if not dists[c1] or not dists[c2]:
        return None, dists
    shared = set(dists[c1]) & set(dists[c2])
    o_e = sum(min(dists[c1][r], dists[c2][r]) for r in shared)
    return o_e, dists
