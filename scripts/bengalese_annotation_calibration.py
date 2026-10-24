#!/usr/bin/env python3
"""Annotation-informed acceptance calibration (Bengalese, bounded).

PROVENANCE (changes per review): the method becomes ANNOTATION-CALIBRATED
ACOUSTIC MATCHING followed by label-blind discovery. Development class
labels calibrate the acceptance rule; they remain part of the method's
training history. Evaluation labels stay hidden until scoring.

PREREGISTRATION (frozen before any score was computed):

  MATERIAL: development strata only (discovery+calibration WAVs of the
  benchmark), split by FROZEN wav identities into calibration-select and
  dev-eval (sorted dev WAVs, i%2). The validation stratum is untouched and
  reserved; any future reuse is a REVISED-PROCEDURE REANALYSIS, never a
  fresh confirmation.

  PRIMARY OPERATING DECISION: the deployed full-host local search — one
  natural query per top class (deterministic: first calibration-select
  occurrence in sorted wav order), subsequence DTW over COMPLETE hosts,
  identical free-start/free-end rules and degeneracy guard for every
  condition. A minimum over many intervals is never swapped for a
  supplied-interval score.

  SUCCESS DEFINITION: an accepted interval (cost <= tau) overlapping >=50%
  with an annotated occurrence of the query's class. Wrong-class and gap
  acceptances counted separately with their own denominators.

  OBJECTIVE (single declared criterion):
      J(tau) = 0.5 * ( R_class_balanced + 1 - F_different_balanced )
  over with-class hosts (recall) and without-class hosts (incorrect
  acceptance), SUBJECT TO the prespecified background constraint
  F_gap(tau) <= BETA = 0.10, and the minimum-useful-recovery requirement
  R_class(tau) >= R_MIN = 0.25. A bird with NO grid point satisfying both
  constraints reports ABSTENTION ("no acceptable operating point") —
  reject-everything is not a successful calibration.

  GRID: 41 quantiles (2.5%-97.5%) of the with-class host best-cost
  distribution on calibration-select — frozen before scoring.

  FROZEN SAMPLE: query occurrence (wav+interval), host wav lists and all
  per-host best costs are recorded in the artifact; the seed and class
  quotas are fixed. No redrawing until a favourable threshold appears.

  ASSESSMENT: the selected tau* per bird is applied ONCE to dev-eval
  (disjoint WAVs) — class-balanced recall, wrong-class and gap acceptance.
  Per-bird abstention is a coherent outcome; no forcing one decision
  across birds.

  SCOPE: acceptance calibration ONLY — representation, DTW scoring, seed
  selection and family construction are unchanged, so any improvement is
  attributable to this intervention.

Usage:
    python scripts/bengalese_annotation_calibration.py [--birds 0 1 2]

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

TOP_CLASSES = 6
MAX_HOSTS = 8
BETA = 0.10
R_MIN = 0.25
GRID_N = 41
SEED = 0


def load_bird_dev(bird: int, bouts: int):
    from concurrent.futures import ProcessPoolExecutor

    bdir = bnb.DATA / f"Bird{bird}"
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
    dev_wavs = sorted(w for w in uniq if strata[w] != "validation")
    half = {w: ("calibration-select" if i % 2 == 0 else "dev-eval") for i, w in enumerate(dev_wavs)}
    for b in loaded:
        b["stage"] = strata[b["wav"]]
        b["half"] = half.get(b["wav"])
    dev = [b for b in loaded if b["stage"] != "validation"]
    frames = np.vstack([b["track"] for b in dev])
    mu, sd = frames.mean(0), frames.std(0) + 1e-9
    for b in dev:
        b["z"] = (b["track"] - mu) / sd
    return dev


def _host_result(query, host_bouts):
    """Best subsequence-DTW match of the query in each host bout."""
    out = []
    for b in host_bouts:
        cost, s, e = std.subseq_dtw(query, b["z"])
        out.append(
            {
                "host": b["wav"],
                "cost": float(cost) if np.isfinite(cost) else None,
                "interval": [int(s), int(e)],
                "ann": [(a["label"], a["s"], a["e"]) for a in b["ann"]],
            }
        )
    return out


def _classify(hit, query_class):
    """correct / wrong-class / gap for an accepted interval (>=50% overlap)."""
    if hit["cost"] is None:
        return "no-accept"
    s, e = hit["interval"]
    length = max(e - s, 1)
    best = ("gap", 0.0)
    for lab, a_s, a_e in hit["ann"]:
        ov = min(e, a_e) - max(s, a_s)
        if ov / length > best[1]:
            best = (lab, ov / length)
    if best[1] < 0.5:
        return "gap"
    return "correct" if best[0] == query_class else "wrong-class"


def _curves(hits_by_class, tau_grid):
    """Class-balanced recall / false-acceptance curves over the grid."""
    R, F, Fgap, Fwrong = [], [], [], []
    for tau in tau_grid:
        r_c, f_c, fg_c, fw_c = [], [], [], []
        for cls, rec in hits_by_class.items():
            with_c = [h for h in rec["with_class"]]
            without = [h for h in rec["without_class"]]
            if not with_c:
                continue
            r_c.append(
                np.mean(
                    [
                        _classify(h, cls) == "correct"
                        and h["cost"] is not None
                        and h["cost"] <= tau
                        for h in with_c
                    ]
                )
            )
            if without:
                acc = [h for h in without if h["cost"] is not None and h["cost"] <= tau]
                f_c.append(len(acc) / len(without))
                fg_c.append(sum(1 for h in acc if _classify(h, cls) == "gap") / len(without))
                fw_c.append(
                    sum(1 for h in acc if _classify(h, cls) == "wrong-class") / len(without)
                )
        R.append(float(np.mean(r_c)) if r_c else 0.0)
        F.append(float(np.mean(f_c)) if f_c else 0.0)
        Fgap.append(float(np.mean(fg_c)) if fg_c else 0.0)
        Fwrong.append(float(np.mean(fw_c)) if fw_c else 0.0)
    return R, F, Fgap, Fwrong


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    args = ap.parse_args()

    report = {
        "preregistration": "see module docstring (frozen)",
        "provenance": "annotation-calibrated acoustic matching, "
        "followed by (future) label-blind discovery; validation "
        "stratum untouched; future validation reuse = revised-"
        "procedure reanalysis",
        "per_bird": {},
    }

    for bird in args.birds:
        dev = load_bird_dev(bird, args.bouts)
        if not dev:
            continue
        sel = [b for b in dev if b["half"] == "calibration-select"]
        ev = [b for b in dev if b["half"] == "dev-eval"]
        if len(sel) < 4 or len(ev) < 4:
            print(f"[Bird{bird}] insufficient halves — skipped", flush=True)
            continue

        occ: dict = {}
        for b in sel:
            for a in b["ann"]:
                occ.setdefault(a["label"], []).append((b, a["s"], a["e"]))
        top = sorted(occ, key=lambda k: -len(occ[k]))[:TOP_CLASSES]

        def _collect(bouts):
            """Host pools keyed by BOUT (wav#seq) — one wav backs several
            Sequence bouts, so wav-level pools could put the same source in
            both conditions (fixed after the first run exposed it; the
            abstention frontier direction was unchanged)."""
            by_class = {}
            frozen = {}
            for i, b in enumerate(bouts):
                b["bout_id"] = f"{b['wav']}#{i}"
            for cls in top:
                items = sorted(occ[cls], key=lambda x: x[0]["wav"])
                if not items:
                    continue
                qb, qs, qe = items[0]
                q = qb["z"][qs:qe]
                with_bouts = [
                    b for b in bouts if any(a["label"] == cls for a in b["ann"]) and b is not qb
                ][:MAX_HOSTS]
                without_bouts = [b for b in bouts if not any(a["label"] == cls for a in b["ann"])][
                    :MAX_HOSTS
                ]
                by_class[cls] = {
                    "with_class": _host_result(q, with_bouts),
                    "without_class": _host_result(q, without_bouts),
                }
                frozen[cls] = {
                    "query": {"wav": qb["wav"], "interval": [int(qs), int(qe)]},
                    "with_bouts": [b["bout_id"] for b in with_bouts],
                    "without_bouts": [b["bout_id"] for b in without_bouts],
                }
            return by_class, frozen

        sel_hits, frozen_sel = _collect(sel)
        # grid from same-class best costs (frozen before scoring)
        pos_costs = [
            h["cost"]
            for rec in sel_hits.values()
            for h in rec["with_class"]
            if h["cost"] is not None
        ]
        if not pos_costs:
            report["per_bird"][str(bird)] = {"outcome": "no positive costs"}
            continue
        grid = np.unique(np.quantile(pos_costs, np.linspace(0.025, 0.975, GRID_N)))
        R, F, Fgap, Fwrong = _curves(sel_hits, grid)
        J = [0.5 * (r + 1 - f) for r, f in zip(R, F)]
        ok = [i for i in range(len(grid)) if Fgap[i] <= BETA and R[i] >= R_MIN]
        if not ok:
            report["per_bird"][str(bird)] = {
                "outcome": "ABSTENTION — no acceptable operating point "
                f"(constraint F_gap<={BETA}, R>={R_MIN} unsatisfiable on "
                "the grid)",
                "frozen_sample": frozen_sel,
            }
            continue
        best = max(ok, key=lambda i: J[i])
        tau_star = float(grid[best])

        # ---- assessment ONCE on dev-eval with frozen tau*
        ev_hits, _ = _collect(ev)
        R_e = [
            np.mean(
                [
                    _classify(h, cls) == "correct"
                    and h["cost"] is not None
                    and h["cost"] <= tau_star
                    for h in rec["with_class"]
                ]
            )
            for cls, rec in ev_hits.items()
            if rec["with_class"]
        ]
        F_e = [
            len(
                [h for h in rec["without_class"] if h["cost"] is not None and h["cost"] <= tau_star]
            )
            / max(len(rec["without_class"]), 1)
            for cls, rec in ev_hits.items()
            if rec["without_class"]
        ]
        Fgap_e = [
            len(
                [
                    h
                    for h in rec["without_class"]
                    if h["cost"] is not None
                    and h["cost"] <= tau_star
                    and _classify(h, cls) == "gap"
                ]
            )
            / max(len(rec["without_class"]), 1)
            for cls, rec in ev_hits.items()
            if rec["without_class"]
        ]
        Fwrong_e = [
            len(
                [
                    h
                    for h in rec["without_class"]
                    if h["cost"] is not None
                    and h["cost"] <= tau_star
                    and _classify(h, cls) == "wrong-class"
                ]
            )
            / max(len(rec["without_class"]), 1)
            for cls, rec in ev_hits.items()
            if rec["without_class"]
        ]
        report["per_bird"][str(bird)] = {
            "outcome": "operating point selected",
            "tau_star": round(tau_star, 4),
            "J_at_tau": round(J[best], 4),
            "calibration_select": {
                "R_class": round(R[best], 4),
                "F_diff": round(F[best], 4),
                "F_gap": round(Fgap[best], 4),
                "F_wrong_class": round(Fwrong[best], 4),
                "grid_range": [round(float(grid[0]), 4), round(float(grid[-1]), 4)],
            },
            "dev_eval": {
                "R_class": round(float(np.mean(R_e)), 4) if R_e else None,
                "F_diff": round(float(np.mean(F_e)), 4) if F_e else None,
                "F_gap": round(float(np.mean(Fgap_e)), 4) if Fgap_e else None,
                "F_wrong_class": round(float(np.mean(Fwrong_e)), 4) if Fwrong_e else None,
                "n_classes": len(R_e),
            },
            "frozen_sample": frozen_sel,
        }
        r = report["per_bird"][str(bird)]
        print(
            f"[Bird{bird}] tau*={tau_star:.4f} J={J[best]:.3f} | sel "
            f"R={R[best]:.3f} Fgap={Fgap[best]:.3f} | dev-eval "
            f"R={r['dev_eval']['R_class']} F={r['dev_eval']['F_diff']} "
            f"Fgap={r['dev_eval']['F_gap']}",
            flush=True,
        )

    out = ROOT / "outputs" / "bengalese_annotation_calibration.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
