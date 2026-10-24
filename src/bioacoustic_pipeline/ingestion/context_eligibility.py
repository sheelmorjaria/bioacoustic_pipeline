"""Caller-context eligibility — the Stage-1 identifiability audit (§46).

Before any context-conditioned claim: which (caller, context-contrast) cells
have enough bouts to support a caller-controlled design? A caller missing
one side of a contrast is NOT evaluable for it — flagged, never silently
dropped or pooled. Failure to support a contrast is a data fact, not a
negative result.

Annotations are any objects exposing ``.emitter`` and ``.context`` (the bat
corpus's CallAnnotation: per-bout rows with integer Context codes).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations


def build_caller_context_support_matrix(annotations) -> dict:
    """Count eligible bouts per (caller, context) cell.

    The full observed caller × context cross-product is materialized with
    zeros, so absent cells are explicit facts rather than KeyErrors.
    """
    counts: dict = {}
    callers, contexts = set(), set()
    for ann in annotations:
        key = (ann.emitter, ann.context)
        counts[key] = counts.get(key, 0) + 1
        callers.add(ann.emitter)
        contexts.add(ann.context)
    return {(e, c): counts.get((e, c), 0) for e in callers for c in contexts}


def check_evaluability(
    matrix: dict,
    *,
    caller,
    c1,
    c2,
    min_bouts: int = 1,
) -> dict:
    """Can this caller support the c1-vs-c2 contrast?

    Evaluable requires >= ``min_bouts`` on BOTH sides. The returned counts
    travel with the verdict so thin cells are visible, not just flagged.
    """
    n_c1 = matrix.get((caller, c1), 0)
    n_c2 = matrix.get((caller, c2), 0)
    if n_c1 < min_bouts or n_c2 < min_bouts:
        return {
            "evaluable": False,
            "reason": "missing_context_data",
            "n_c1": n_c1,
            "n_c2": n_c2,
        }
    return {
        "evaluable": True,
        "reason": "adequate_support",
        "n_c1": n_c1,
        "n_c2": n_c2,
    }


def audit_all_contrasts(annotations, *, min_bouts: int = 5) -> dict:
    """Full eligibility audit: every caller × context-pair, with counts.

    Returns {contexts, callers, evaluable_pairs, not_evaluable_pairs,
    n_evaluable_calls} — the pre-registered map of what the caller-controlled
    design can and cannot test on this corpus.
    """
    matrix = build_caller_context_support_matrix(annotations)
    contexts = sorted({c for (_, c) in matrix})
    callers = sorted({e for (e, _) in matrix})
    evaluable, not_evaluable = [], []
    for i, c1 in enumerate(contexts):
        for c2 in contexts[i + 1 :]:
            for caller in callers:
                res = check_evaluability(matrix, caller=caller, c1=c1, c2=c2, min_bouts=min_bouts)
                row = {
                    "caller": caller,
                    "c1": c1,
                    "c2": c2,
                    "n_c1": res["n_c1"],
                    "n_c2": res["n_c2"],
                }
                (evaluable if res["evaluable"] else not_evaluable).append(row)
    return {
        "min_bouts": min_bouts,
        "contexts": contexts,
        "callers": callers,
        "evaluable": evaluable,
        "not_evaluable": not_evaluable,
        "n_evaluable_calls": len(evaluable),
    }
