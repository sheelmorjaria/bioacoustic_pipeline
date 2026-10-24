#!/usr/bin/env python3
"""Candidate-ranking matcher intervention on the diagnosed Bengalese failure.

PREREGISTRATION (frozen before any score was computed):

  HYPOTHESIS: some missed natural acoustic correspondences are present
  among multiple supported local candidate alignments but are ranked below
  generic background intervals by the raw-distance criterion. A
  candidate-specific ranking rule may recover correct intervals without
  exceeding the declared incorrect-acceptance limits. This is a PROPOSED
  explanation for the measured selection-before-threshold failure — not a
  conclusion.

  STAGE 1 — ORACLE DIAGNOSTIC: for each query-host pair, retain K=10
  DISTINCT local candidates (non-overlapping; the original best interval
  always included) via the full-DP end-cost row. Measure, on annotated
  DEVELOPMENT recordings, class-balanced
      R_top1     (old decision correct)
      R_oracle@K (a correct annotated interval is AMONG the candidates)
  A substantial gap = room for a ranking intervention; little gap = the
  proposal did not expose correct intervals. Oracle is a DIAGNOSTIC;
  reference labels never choose the reported prediction.

  STAGE 2 — RANKING INTERVENTION (background-relative): for candidate I,
      S(q, I) = B(I) - d(q, I),   B(I) = min_{b in bank} d(b, I)
  with a FROZEN development-derived background bank of REAL gap
  trajectories (representation-supported; never gated zero vectors),
  duration-matched to the query grid; d = fixed-interval DTW under the
  declared conventions. Arms on the SAME candidate set:
      baseline  : lowest original DTW cost
      proposed  : highest frozen background-relative score
      oracle    : diagnostic only (not deployable)
  A monotone rescaling of one query's costs cannot change which interval
  ranks first — the intervention changes ranking via B(I), and 'more
  candidates' alone is accounted by the oracle arm, never claimed as
  improvement.

  STAGE 3 — FOUR CAPABILITIES, separate denominators:
      matching   correct-correspondence recall on with-class hosts;
                 wrong-class and gap acceptance on without-class hosts
      localisation one-to-one interval recovery under the frozen >=50%
                 overlap criterion; boundary errors alongside misses
      grouping   accepted same-class occurrences grouped under the frozen
                 exemplar radius without merging wrong classes
      coverage   unique eligible query-host pairs and reference classes
                 reached (no overlapping-window double counting)

  ACCEPTANCE: per arm, a frozen threshold (grid from correct-candidate
  score quantiles on the calibration-select half) subject to the retained
  requirements F_gap <= 0.10 and class-balanced R >= 0.25; abstention if
  unsatisfiable. Increased acceptance alone is not success.

  PROVENANCE: annotation-calibrated matching (dev labels define the
  calibration material); evaluation on the disjoint dev-eval half;
  reserved validation stratum untouched; marmoset portability is a LATER
  step, not part of this study. Recovered acoustic classes are NOT
  linguistic units or meanings.

Usage:
    python scripts/bengalese_candidate_ranking.py [--birds 0 1 2]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import bengalese_annotation_calibration as bac

K_CANDIDATES = 10
TOP_CLASSES = 6
MAX_HOSTS = 8
N_BANK = 40
BETA = 0.10
R_MIN = 0.25
GRID_N = 41
SEED = 0


def dp_matrix(q, c):
    """Full subsequence-DP matrix (free starts) + backpointer-free backtrack."""
    q, c = np.asarray(q, float), np.asarray(c, float)
    n, m = len(q), len(c)
    big = 1e12
    D = np.full((n + 1, m + 1), big)
    D[0, :] = 0.0
    for i in range(1, n + 1):
        step = np.sqrt(((q[i - 1] - c) ** 2).sum(axis=1))
        seg = np.minimum(np.minimum(D[i - 1, :m], D[i - 1, 1:]), D[i, :m])
        D[i, 1:] = np.minimum(D[i, 1:], seg + step)
    return D


def backtrack_start(D, j):
    n = D.shape[0] - 1
    i, jj, c_start = n, j, j
    big = 1e12
    while i > 0:
        cands = []
        for di, dj in ((-1, 0), (0, -1), (-1, -1)):
            ii, jjj = i + di, jj + dj
            if ii >= 0 and jjj >= 0 and D[ii, jjj] < big:
                cands.append((D[ii, jjj], ii, jjj))
        if not cands:
            break
        _, i, jj = min(cands)
        c_start = min(c_start, jj)
    return c_start


def kbest_candidates(query, host, k=K_CANDIDATES, min_sep=None):
    """Up to k distinct non-overlapping local candidates, best-first.

    The original global best is always the first returned element.
    """
    D = dp_matrix(query, host)
    n = D.shape[0] - 1
    end_costs = D[n, 1:]
    order = np.argsort(end_costs)
    if min_sep is None:
        min_sep = max(2, len(query) // 2)
    out = []
    taken: list = []
    for j in order:
        if len(out) >= k:
            break
        j = int(j) + 1
        if not np.isfinite(D[n, j]) or D[n, j] >= 1e12:
            continue
        s0 = backtrack_start(D, j)
        e0 = j
        if e0 - s0 + 1 < max(1, len(query) // 2):
            continue  # degeneracy guard, as the frozen matcher
        if any(not (e0 < s - min_sep or s0 > e + min_sep) for s, e in taken):
            continue  # distinct-local suppression
        taken.append((s0, e0))
        out.append({"cost": float(D[n, j] / n), "interval": [int(s0), int(e0)]})
    return out, D


def dtw_fixed(a, b):
    """Fixed-interval DTW (both boundaries supplied), mean per-step cost."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    big = 1e12
    D = np.full((n + 1, m + 1), big)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        step = np.sqrt(((a[i - 1] - b) ** 2).sum(axis=1))
        seg = np.minimum(np.minimum(D[i - 1, :m], D[i - 1, 1:]), D[i, :m])
        D[i, 1:] = np.minimum(D[i, 1:], seg + step)
    return float(D[n, m] / (n + m))


def _overlap_kind(cand, anns, query_class):
    s, e = cand["interval"]
    length = max(e - s, 1)
    best = ("gap", 0.0)
    for lab, a_s, a_e in anns:
        ov = (min(e, a_e) - max(s, a_s)) / length
        if ov > best[1]:
            best = (lab, ov)
    if best[1] < 0.5:
        return "gap"
    return "correct" if best[0] == query_class else "wrong-class"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    args = ap.parse_args()

    report = {
        "preregistration": "see module docstring (frozen)",
        "provenance": "annotation-calibrated matching; dev material "
        "only; reserved validation untouched; recovered classes are "
        "not linguistic units",
        "per_bird": {},
    }

    for bird in args.birds:
        dev = bac.load_bird_dev(bird, args.bouts)
        if not dev:
            continue
        for i, b in enumerate(dev):
            b["bout_id"] = f"{b['wav']}#{i}"
        sel = [b for b in dev if b["half"] == "calibration-select"]
        ev = [b for b in dev if b["half"] == "dev-eval"]
        if len(sel) < 4 or len(ev) < 4:
            continue

        occ: dict = {}
        for b in sel:
            for a in b["ann"]:
                occ.setdefault(a["label"], []).append((b, a["s"], a["e"]))
        top = sorted(occ, key=lambda k: -len(occ[k]))[:TOP_CLASSES]

        # ---- frozen background bank: REAL gap trajectories from dev,
        # keyed by each class's ACTUAL query length (annotation-derived,
        # not on a fixed grid); frozen: first-by-wav-order, size-capped
        bank: dict = {}

        def _gaps_of_length(L):
            gaps = []
            for b in sorted(sel, key=lambda x: x["wav"]):
                spans = sorted((a["s"], a["e"]) for a in b["ann"])
                prev = 0
                for s, e in spans + [(10**9, 10**9)]:
                    if s - prev >= L:
                        gaps.append(b["z"][prev : prev + L])
                    prev = e
            return gaps[:N_BANK]

        def B_of(cand_interval_z, L):
            return min(dtw_fixed(b, cand_interval_z) for b in bank[L])

        def _run(half_bouts, tag):
            """Arms + oracle over the same candidate sets."""
            rows = []
            for cls in top:
                items = sorted(occ[cls], key=lambda x: x[0]["wav"])
                if not items:
                    continue
                qb, qs, qe = items[0]
                q = qb["z"][qs:qe]
                L = qe - qs
                if L not in bank:
                    bank[L] = _gaps_of_length(L)
                if not bank[L]:
                    continue
                with_b = [
                    b
                    for b in half_bouts
                    if any(a["label"] == cls for a in b["ann"]) and b is not qb
                ][:MAX_HOSTS]
                without_b = [b for b in half_bouts if not any(a["label"] == cls for a in b["ann"])][
                    :MAX_HOSTS
                ]
                for b in with_b + without_b:
                    cands, _D = kbest_candidates(q, b["z"])
                    if not cands:
                        continue
                    anns = [(a["label"], a["s"], a["e"]) for a in b["ann"]]
                    kind_top1 = _overlap_kind(cands[0], anns, cls)
                    oracle_hit = any(_overlap_kind(c, anns, cls) == "correct" for c in cands)
                    for c in cands:
                        z_I = b["z"][c["interval"][0] : c["interval"][1]]
                        c["S"] = B_of(z_I, L) - c["cost"]
                    best_S = max(cands, key=lambda c: c["S"])
                    rows.append(
                        {
                            "cls": cls,
                            "host": b["bout_id"],
                            "is_with_class": any(b is x for x in with_b),
                            "kind_top1": kind_top1,
                            "kind_S": _overlap_kind(best_S, anns, cls),
                            "oracle_hit": oracle_hit,
                            "cost_top1": cands[0]["cost"],
                            "S_top1": best_S["S"],
                            "correct_S_score": max(
                                (c["S"] for c in cands if _overlap_kind(c, anns, cls) == "correct"),
                                default=None,
                            ),
                        }
                    )
            return rows

        sel_rows = _run(sel, "calibration-select")

        def _metrics(rows, thr_cost, thr_S):
            with_r = [r for r in rows if r["is_with_class"]]
            without = [r for r in rows if not r["is_with_class"]]

            def _acc(r, arm):
                return r["cost_top1"] <= thr_cost if arm == "baseline" else r["S_top1"] >= thr_S

            def _kind(r, arm):
                return r["kind_top1"] if arm == "baseline" else r["kind_S"]

            out = {}
            for arm in ("baseline", "proposed"):
                rec = [
                    1.0 if (_acc(r, arm) and _kind(r, arm) == "correct") else 0.0 for r in with_r
                ]
                acc = [r for r in without if _acc(r, arm)]
                f_gap = (
                    (sum(1 for r in acc if _kind(r, arm) == "gap") / len(without))
                    if without
                    else 0.0
                )
                f_wrong = (
                    (sum(1 for r in acc if _kind(r, arm) == "wrong-class") / len(without))
                    if without
                    else 0.0
                )
                out[arm] = {
                    "R_class": float(np.mean(rec)) if rec else 0.0,
                    "F_gap": f_gap,
                    "F_wrong_class": f_wrong,
                }
            # oracle ceilings (diagnostic; no acceptance rule)
            per_cls_r1, per_cls_ok = {}, {}
            for r in with_r:
                per_cls_r1.setdefault(r["cls"], []).append(
                    1.0 if r["kind_top1"] == "correct" else 0.0
                )
                per_cls_ok.setdefault(r["cls"], []).append(1.0 if r["oracle_hit"] else 0.0)
            out["oracle"] = {
                "R_top1_class_balanced": float(np.mean([np.mean(v) for v in per_cls_r1.values()])),
                "R_oracle_at_K_class_balanced": float(
                    np.mean([np.mean(v) for v in per_cls_ok.values()])
                ),
            }
            return out

        # ---- frozen threshold grids from correct-candidate scores
        pos_cost = [
            r["cost_top1"] for r in sel_rows if r["is_with_class"] and r["kind_top1"] == "correct"
        ]
        pos_S = [r["S_top1"] for r in sel_rows if r["is_with_class"] and r["kind_S"] == "correct"]
        cost_grid = np.unique(np.quantile(pos_cost or [1.0], np.linspace(0.025, 0.975, GRID_N)))
        S_grid = np.unique(np.quantile(pos_S or [0.0], np.linspace(0.025, 0.975, GRID_N)))
        best = {}
        for arm, grid in (("baseline", cost_grid), ("proposed", S_grid)):
            cand_points = []
            for i, thr in enumerate(grid):
                m = _metrics(
                    sel_rows,
                    thr if arm == "baseline" else 1e9,
                    1e9 if arm == "baseline" else thr,
                )
                if m[arm]["F_gap"] <= BETA and m[arm]["R_class"] >= R_MIN:
                    J = 0.5 * (m[arm]["R_class"] + 1 - m[arm]["F_wrong_class"])
                    cand_points.append((J, i, thr, m))
            if cand_points:
                J, i, thr, m = max(cand_points)
                best[arm] = {"threshold": float(thr), "J": round(J, 4), "grid_point": i}
            else:
                best[arm] = None  # abstention for this arm

        ev_rows = _run(ev, "dev-eval")
        ev_metrics = _metrics(
            ev_rows,
            (best["baseline"] or {}).get("threshold", 1e9),
            (best["proposed"] or {}).get("threshold", -1e9),
        )
        oracle_sel = _metrics(sel_rows, 1e9, -1e9)["oracle"]
        oracle_ev = _metrics(ev_rows, 1e9, -1e9)["oracle"]

        report["per_bird"][str(bird)] = {
            "bank_sizes": {str(L): len(v) for L, v in bank.items()},
            "n_rows_sel": len(sel_rows),
            "n_rows_eval": len(ev_rows),
            "oracle_diagnostic": {
                "calibration_select": oracle_sel,
                "dev_eval": oracle_ev,
                "gap_R_top1_to_oracle": round(
                    oracle_sel["R_oracle_at_K_class_balanced"]
                    - oracle_sel["R_top1_class_balanced"],
                    4,
                ),
            },
            "calibrated_thresholds": best,
            "dev_eval_metrics": ev_metrics,
            "coverage": {
                "unique_query_host_pairs": len(sel_rows) + len(ev_rows),
                "classes": len(top),
            },
            "outcome": None,  # filled below
        }
        r = report["per_bird"][str(bird)]
        gap = r["oracle_diagnostic"]["gap_R_top1_to_oracle"]
        prop = best.get("proposed")
        base = best.get("baseline")
        if gap < 0.05:
            r["outcome"] = (
                "correct intervals largely ABSENT from the "
                "candidate set — reranking cannot address "
                "the main problem"
            )
        elif prop is None and base is None:
            r["outcome"] = (
                "ABSTENTION both arms — no operating point meets F_gap<=0.10 with R>=0.25"
            )
        elif prop is None:
            r["outcome"] = "baseline operating point exists; proposed ranking abstains"
        else:
            de = ev_metrics["proposed"]
            r["outcome"] = (
                f"proposed ranking calibrated (J={prop['J']}); "
                f"dev-eval R={de['R_class']:.3f} "
                f"F_gap={de['F_gap']:.3f} "
                f"F_wrong={de['F_wrong_class']:.3f}"
            )
        print(
            f"[Bird{bird}] oracle gap {gap} | top1 "
            f"{oracle_sel['R_top1_class_balanced']:.3f} -> @K "
            f"{oracle_sel['R_oracle_at_K_class_balanced']:.3f} | "
            f"{r['outcome']}",
            flush=True,
        )

    out = ROOT / "outputs" / "bengalese_candidate_ranking.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
