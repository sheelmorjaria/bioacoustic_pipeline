#!/usr/bin/env python3
"""Development-set score diagnostics for the Bengalese natural-query failure.

PURPOSE (review-directed): BEFORE any new calibration rule, separate "poor
scores" from "a poorly placed cutoff" on DEVELOPMENT recordings only
(discovery+calibration strata; validation untouched). Two diagnostics:

  D1 — ANNOTATED-SEGMENT COMPARISON (reference boundaries supplied on both
       sides): full-alignment DTW costs for (a) same-class renditions from
       separate recordings, (b) duration-matched different-class pairs,
       (c) duration-matched inter-syllable GAP alternatives (reported with
       their own denominator, separate from vocal-class alternatives).
       Output: score distributions + rank separation + where the frozen
       radius tau sits in each.

  D2 — FULL-HOST LOCAL SEARCH (query supplied, location free): subsequence
       DTW over whole development hosts. Retained per host: the TRUE
       annotated interval's cost AND the best returned interval (so a
       background minimum cannot conceal a reasonably-similar correct
       rendition). Output: localization-hit fraction (argmin inside the
       true interval), true-interval acceptance under tau, and the
       best-vs-true score gap.

  FOUR-WAY MAPPING (diagnostic outcomes, assigned AFTER examining scores):
    same-class ranks well + localises, but exceeds tau -> acceptance
      calibration;
    same/different overlap strongly even with reference boundaries ->
      representation/metric discrimination;
    reference-boundary comparison works but free-start picks the wrong
      interval -> localisation / path scoring / candidate eligibility;
    (known-query works but blind discovery fails -> seeding — not testable
      here; blind discovery already failed and is downstream.)

  NO NEW THRESHOLD IS CHOSEN in this script. The subsequent
  annotation-calibrated rule (if warranted) is a separately versioned
  analysis with its criterion declared before new evaluation results.

Usage:
    python scripts/bengalese_score_diagnostics.py [--birds 0 1 2] [--bouts 90]

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

import bengalese_natural_benchmark as bnb
import subcall_term_discovery as std

DATA = bnb.DATA
TOP_CLASSES = 6
MAX_PAIRS_PER_COND = 30
SEED = 0


def dtw_full(a, b):
    """Fixed-endpoint DTW, mean per-step cost (both boundaries supplied)."""
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


def load_bird(bird: int, bouts: int):
    """Same deterministic material + strata as the benchmark."""
    from concurrent.futures import ProcessPoolExecutor

    bdir = DATA / f"Bird{bird}"
    xml = bdir / "Annotation.xml"
    if not xml.exists():
        return None
    bouts_raw = bnb.parse_annotation(xml)[:bouts]
    hop, win = int(bnb.SR * bnb.HOP_MS / 1000), int(bnb.SR * bnb.WIN_MS / 1000)
    jobs = [(bdir / "Wave" / w, notes, hop, win, w) for w, notes in bouts_raw]
    with ProcessPoolExecutor(max_workers=24) as ex:
        loaded = [r for r in ex.map(bnb._bout_job, jobs, chunksize=2) if r]
    uniq = sorted({b["wav"] for b in loaded})
    strata = {
        w: ("discovery" if i % 5 in (0, 1) else "calibration" if i % 5 == 2 else "validation")
        for i, w in enumerate(uniq)
    }
    for b in loaded:
        b["stage"] = strata[b["wav"]]
    dev = [b for b in loaded if b["stage"] in ("discovery", "calibration")]
    return dev


def _gap_intervals(b, n_frames):
    """Duration-matched inter-syllable gap trajectories from one bout."""
    out = []
    spans = sorted((a["s"], a["e"]) for a in b["ann"])
    prev_end = 0
    for s, e in spans + [(10**9, 10**9)]:
        if s - prev_end >= n_frames:
            out.append(b["z"][prev_end : prev_end + n_frames])
        prev_end = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    report = {
        "purpose": __doc__.split("PURPOSE")[1].split("Usage:")[0].strip(),
        "material": "development strata only (discovery+calibration); validation untouched",
        "per_bird": {},
    }

    for bird in args.birds:
        dev = load_bird(bird, args.bouts)
        if not dev or len(dev) < 6:
            continue
        # frozen per-bird radius, re-derived exactly as the benchmark
        disc = [b for b in dev if b["stage"] == "discovery"]
        cal = [b for b in dev if b["stage"] == "calibration"]
        frames = np.vstack([b["track"] for b in disc])
        mu, sd = frames.mean(0), frames.std(0) + 1e-9
        for b in dev:
            b["z"] = (b["track"] - mu) / sd
        costs = []
        for i in range(40):
            p = cal[rng.randint(0, len(cal))]
            L = bnb.LENGTHS[i % len(bnb.LENGTHS)]
            if len(p["z"]) < L + 2:
                continue
            s0 = int(rng.randint(0, len(p["z"]) - L))
            for k2 in rng.randint(0, len(cal), size=6):
                p2 = cal[k2]
                if p2["wav"] == p["wav"]:
                    continue
                costs.append(std.subseq_dtw(p["z"][s0 : s0 + L], p2["z"])[0])
        tau = float(np.percentile([c for c in costs if np.isfinite(c)], bnb.CTRL_PCT))

        # syllable occurrences by class (dev, cross-recording)
        occ: dict = {}
        for b in dev:
            for a in b["ann"]:
                occ.setdefault(a["label"], []).append((b, a["s"], a["e"]))
        top = sorted(occ, key=lambda k: -len(occ[k]))[:TOP_CLASSES]

        # ---- D1: reference-boundary costs
        same, diff, gap = [], [], []
        for lab in top:
            items = occ[lab]
            pairs = [
                (i, j)
                for i in range(len(items))
                for j in range(i + 1, len(items))
                if items[i][0]["wav"] != items[j][0]["wav"]
            ]
            rng.shuffle(pairs)
            for i, j in pairs[: MAX_PAIRS_PER_COND // TOP_CLASSES * 2]:
                ba, sa, ea = items[i]
                bb, sb, eb = items[j]
                same.append(dtw_full(ba["z"][sa:ea], bb["z"][sb:eb]))
                # duration-matched different-class pair (robust: direct
                # draw from other-class occurrences within +-3 frames)
                n = ea - sa
                pool = [
                    x
                    for lab2 in top
                    if lab2 != lab
                    for x in occ[lab2]
                    if abs((x[2] - x[1]) - n) <= 3 and x[0]["wav"] != ba["wav"]
                ]
                if pool:
                    b2, s2, e2 = pool[rng.randint(0, len(pool))]
                    diff.append(dtw_full(ba["z"][sa:ea], b2["z"][s2:e2]))
                # duration-matched gap alternative
                gaps = _gap_intervals(bb, n) + _gap_intervals(ba, n)
                if gaps:
                    gap.append(dtw_full(ba["z"][sa:ea], gaps[rng.randint(0, len(gaps))]))
        same = np.array(same)
        diff = np.array(diff)
        gap = np.array(gap)
        # rank separation: P(same < diff)
        sep = float((same[:, None] < diff[None]).mean()) if len(same) and len(diff) else None
        sep_gap = float((same[:, None] < gap[None]).mean()) if len(same) and len(gap) else None

        # ---- D2: full-host local search
        # CORRECTED 2026-10-20: hosts are BOUTS. The original grouped class
        # occurrences by WAV, searched only the first bout's track, and
        # tested containment / true-interval costs against spans from ALL
        # bouts of that wav in their own coordinate frames — a cross-bout
        # frame mismatch that inflated localization_hit_fraction to
        # 0.8333/0.9062/0.9697 (original values preserved in the artifact's
        # correction record). Occurrence spans are now taken from the SAME
        # bout that was searched.
        loc_hits, true_accepted, best_minus_true = [], [], []
        for lab in top:
            items = occ[lab]
            query = items[rng.randint(0, len(items))]
            qb, qs, qe = query
            q = qb["z"][qs:qe]
            host_bouts, seen_wav = [], set()
            for b, _s, _e in items:
                if b["wav"] == qb["wav"] or b["wav"] in seen_wav:
                    continue
                seen_wav.add(b["wav"])
                host_bouts.append(b)
                if len(host_bouts) >= 6:
                    break
            for b in host_bouts:
                occs = [(s, e) for bb, s, e in items if bb is b]
                cost_best, s_best, e_best = std.subseq_dtw(q, b["z"])
                # true-interval cost (fixed ends within the host)
                trues = [dtw_full(q, b["z"][s:e]) for (s, e) in occs]
                t_best = min(trues)
                loc_hits.append(
                    float(any(s_best >= s - 1 and e_best <= e + 1 for s, e in occs))
                    if np.isfinite(cost_best)
                    else 0.0
                )
                true_accepted.append(float(t_best <= tau))
                if np.isfinite(cost_best):
                    best_minus_true.append(float(cost_best - t_best))
        report["per_bird"][str(bird)] = {
            "tau_frozen": round(tau, 4),
            "D1_annotated_segment": {
                "n_same": len(same),
                "n_diff": len(diff),
                "n_gap": len(gap),
                "same_class_median": round(float(np.median(same)), 4) if len(same) else None,
                "diff_class_median": round(float(np.median(diff)), 4) if len(diff) else None,
                "gap_median": round(float(np.median(gap)), 4) if len(gap) else None,
                "P_same_lt_diff": round(sep, 4) if sep is not None else None,
                "P_same_lt_gap": round(sep_gap, 4) if sep_gap is not None else None,
                "tau_position": {
                    "frac_same_below_tau": round(float((same <= tau).mean()), 4)
                    if len(same)
                    else None,
                    "frac_diff_below_tau": round(float((diff <= tau).mean()), 4)
                    if len(diff)
                    else None,
                    "frac_gap_below_tau": round(float((gap <= tau).mean()), 4)
                    if len(gap)
                    else None,
                },
            },
            "D2_full_host_local_search": {
                "n_hosts": len(loc_hits),
                "localization_hit_fraction": round(float(np.mean(loc_hits)), 4)
                if loc_hits
                else None,
                "true_interval_acceptance_under_tau": round(float(np.mean(true_accepted)), 4)
                if true_accepted
                else None,
                "median_best_minus_true_cost": round(float(np.median(best_minus_true)), 4)
                if best_minus_true
                else None,
                "note": "true-interval cost retained alongside the best "
                "returned interval — a background minimum cannot conceal a "
                "reasonably similar correct rendition",
            },
        }
        r = report["per_bird"][str(bird)]
        print(
            f"[Bird{bird}] tau={tau:.4f} | D1 P(same<diff)="
            f"{r['D1_annotated_segment']['P_same_lt_diff']} "
            f"P(same<gap)={r['D1_annotated_segment']['P_same_lt_gap']} | "
            f"D2 loc={r['D2_full_host_local_search']['localization_hit_fraction']} "
            f"true<=tau={r['D2_full_host_local_search']['true_interval_acceptance_under_tau']}",
            flush=True,
        )

    # four-way mapping (assigned from the measured numbers)
    def _m(b):
        d1 = b["D1_annotated_segment"]
        d2 = b["D2_full_host_local_search"]
        good_sep = (d1["P_same_lt_diff"] or 0) > 0.75
        good_loc = (d2["localization_hit_fraction"] or 0) > 0.5
        accepts = (d1["tau_position"]["frac_same_below_tau"] or 0) > 0.5
        if not good_sep:
            return (
                "representation/metric discrimination (same/different "
                "overlap even with reference boundaries)"
            )
        if good_sep and not good_loc:
            return (
                "localisation / path scoring (reference-boundary works, "
                "free-start picks wrong interval)"
            )
        if good_sep and good_loc and not accepts:
            return "acceptance calibration (scores rank + localise, cutoff rejects positives)"
        return "mixed — inspect per-bird values"

    report["four_way_mapping"] = {b: _m(v) for b, v in report["per_bird"].items()}
    report["no_new_threshold_note"] = "no threshold chosen here; any "
    "annotation-calibrated rule is a separately versioned analysis with its "
    "criterion declared before new evaluation results"
    report["correction_2026_10_20"] = {
        "what": "D2 host construction grouped class occurrences by WAV, "
        "searched only the first bout's track, and tested containment / "
        "true-interval costs against spans from ALL bouts of that wav in "
        "their own coordinate frames (Sequence bouts inside one wav have "
        "different offsets) — a cross-bout frame mismatch",
        "effect": "localization_hit_fraction was inflated; containment "
        "could pass numerically against an unrelated bout's span (matches "
        "landing in gaps or other classes satisfied 'containment')",
        "original_values_preserved": {
            "0": {
                "localization_hit_fraction": 0.8333,
                "n_hosts": 36,
                "true_interval_acceptance_under_tau": 0.0,
            },
            "1": {
                "localization_hit_fraction": 0.9062,
                "n_hosts": 32,
                "true_interval_acceptance_under_tau": 0.8125,
            },
            "2": {
                "localization_hit_fraction": 0.9697,
                "n_hosts": 33,
                "true_interval_acceptance_under_tau": 0.0,
            },
        },
        "original_four_way_mapping_preserved": {
            "0": "representation/metric discrimination (same/different "
            "overlap even with reference boundaries)",
            "1": "acceptance calibration (scores rank + localise, cutoff rejects positives)",
            "2": "acceptance calibration (scores rank + localise, cutoff rejects positives)",
        },
        "found_by": "natural-unit reference benchmark reconciliation "
        "(outputs/natural_unit_reference_benchmark.json + "
        "outputs/nurb_reconciliation.json): same-criterion same-population "
        "replication could not reproduce 0.83-0.97; instrumented replay "
        "showed hit=True matches landing where no same-class element "
        "exists in the searched track",
        "correction": "occurrence spans are now taken from the SAME bout "
        "that was searched (hosts iterated per bout)",
    }
    out = ROOT / "outputs" / "bengalese_score_diagnostics.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["four_way_mapping"], indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
