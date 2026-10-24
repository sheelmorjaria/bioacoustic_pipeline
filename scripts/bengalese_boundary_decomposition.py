#!/usr/bin/env python3
"""Bengalese boundary decomposition — Stage A (NBD boundaries) + Stage B
(oracle-boundary retrieval ceiling). Advisor-directed 2026-10-20.

ARCHITECTURE UNDER TEST (review framing): NBD answers "where might a unit
boundary be?"; the fingerprint answers "are these candidate units
acoustically the same kind?". The old procedure asked free-start DTW to do
both at once and measured selection-before-threshold failures. This script
separates them:

  STAGE A — boundary/localization benchmark (raw fixed-threshold NBD):
    proposals from segmentation.nbd_proposal (default floors 0.05/0.03,
    debounce 30 ms — the species-portable route; NO calibrator: the NBD
    calibrator is human-CPC-dependent and firewall-barred on animals, and
    an annotation-trained animal calibrator is a separately versioned
    circularity-audited step, NOT this benchmark), scored against
    annotated syllable boundaries on the SAME bouts/strata as the natural
    benchmarks:
      boundary precision / recall / F1 at tolerance {1,2,3} frames
      onset & offset error of matched boundaries (ms)
      segment IoU (segments = consecutive proposals vs annotated spans)
      over-segmentation: mean proposal boundaries inside an annotated
        syllable; gap-pollution: fraction of proposals in inter-syllable
        gaps; coverage: annotated syllables with >=1 internal proposal
      R_proposal CEILING = recall at the primary tolerance (3 frames).
    HISTORICAL COMPARATOR (quoted from prior artifacts, not recomputed):
    corrected free-start DTW localisation 0.528/0.250/0.500 (dev strata)
    and benchmark known-query IoU>=0.5 localisation 0.230/0.144/0.118
    (validation strata).

  STAGE B — oracle-boundary same-class retrieval ceiling (R_oracle):
    TRUE annotated syllable spans -> 12-feature fingerprint trajectory ->
    fixed-endpoint full DTW ranking across bouts (no search, no
    thresholds): Recall@1 / Recall@10 same-class, cross-bout only, plus
    P(same < diff) on validation pairs. This separates segmentation
    failure from representation/matching failure.

  CEILINGS TABLE: R_oracle (Stage B), R_proposal (Stage A recall),
  R_selected (DEFERRED: typed-NBD/HECB selection trained within Bengalese
  development material — a separately preregistered arm, not run here).
  R_oracle - R_proposal characterises segmentation/proposal loss;
  R_proposal - R_selected will characterise selection loss. NO FAMILY
  DISCOVERY in this script.

DECISION RULES (fixed now):
  "NBD closes the localisation gap" if boundary F1 >= 0.6 AND R_proposal
     >= 0.8 at the primary tolerance (3 frames = 30 ms);
  "representation is the remaining bottleneck" if oracle Recall@1 < 0.25
     (chance ~ class prevalence, reported alongside);
  "representation adequate" if oracle Recall@1 >= 0.5.

FRAMING/CAVEATS: an NBD proposal is an acoustic-change candidate, NOT a
syllable/motor-primitive/word-like unit (Semantic Ceiling intact). Grid
convention: NBD signal frames (10 ms hop) vs fingerprint window centers
(10 ms hop) may differ by a half-window; tolerance 3 frames absorbs the
offset and is declared primary. One run; no parameter changes after
scores.

Usage:
    python scripts/bengalese_boundary_decomposition.py --self-test
    python scripts/bengalese_boundary_decomposition.py [--birds 0 1 2]

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

from bengalese_natural_benchmark import (
    DATA,
    SR,
    WIN_MS,
    HOP_MS,
    _bout_job,
    _load_wav,
    parse_annotation,
)
from bengalese_score_diagnostics import dtw_full
from natural_unit_reference_benchmark import _boot_rate

TOL_PRIMARY = 3  # frames (30 ms)
SEED = 0
BOOT_B = 1000
N_TOP_CLASSES = 6


def _match_boundaries(true_frames, prop_frames, tol):
    """Greedy nearest one-to-one matching within tol -> (pairs, n_unmatched_true,
    n_unmatched_prop)."""
    used_p = set()
    pairs = []
    for t in sorted(true_frames):
        best = None
        for j, p in enumerate(prop_frames):
            if j in used_p or abs(p - t) > tol:
                continue
            if best is None or abs(p - t) < abs(prop_frames[best] - t):
                best = j
        if best is not None:
            used_p.add(best)
            pairs.append((t, prop_frames[best]))
    return pairs, len(true_frames) - len(pairs), len(prop_frames) - len(pairs)


def _stage_a_bout(b, tol=TOL_PRIMARY):
    """NBD proposals vs annotated boundaries for one bout."""
    import numpy as np

    from bioacoustic_pipeline.segmentation import nbd_proposal as nbd

    audio = _load_wav(Path(b["flac"]))
    if audio is None:
        return None
    sig = nbd.compute_signals(audio, SR, hop_ms=HOP_MS)
    props = nbd.generate_proposals(sig)
    pf = [p.frame for p in props]
    ann = b["ann"]
    if not ann:
        return None
    onset_f = [a["s"] for a in ann]
    offset_f = [a["e"] - 1 for a in ann]
    true_f = sorted(set(onset_f) | set(offset_f))
    out = {"n_proposals": len(pf), "n_true": len(true_f), "n_ann": len(ann)}
    for tol in (1, 2, 3):
        pairs, miss_t, miss_p = _match_boundaries(true_f, pf, tol)
        prec = len(pairs) / len(pf) if pf else None
        rec = len(pairs) / len(true_f) if true_f else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
        out[f"tol{tol}"] = {
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "n_matched": len(pairs),
        }
        if tol == TOL_PRIMARY:
            errs = [abs(p - t) * 10.0 for t, p in pairs]
            out["boundary_abs_error_ms"] = {
                "median": round(float(np.median(errs)), 1) if errs else None,
                "p90": round(float(np.percentile(errs, 90)), 1) if errs else None,
            }
    # segmentation structure: proposals inside vs outside annotated spans
    inside = sum(1 for f in pf if any(a["s"] <= f < a["e"] for a in ann))
    out["gap_pollution"] = round(1 - inside / len(pf), 4) if pf else None
    per_ann = [sum(1 for f in pf if a["s"] <= f < a["e"]) for a in ann]
    out["mean_proposals_per_syllable"] = round(float(np.mean(per_ann)), 3)
    out["syllable_coverage"] = round(float(np.mean([c > 0 for c in per_ann])), 4)
    return out


def _run_bird(bird, args):
    """Stage A over all bouts + Stage B oracle retrieval."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    bdir = DATA / f"Bird{bird}"
    xml = bdir / "Annotation.xml"
    if not xml.exists():
        return None
    bouts_raw = parse_annotation(xml)[: args.bouts]
    jobs = [(bdir / "Wave" / w, notes, hop, win, w) for w, notes in bouts_raw]
    with ProcessPoolExecutor(max_workers=24) as ex:
        loaded = [r for r in ex.map(_bout_job, jobs, chunksize=2) if r]
    uniq = sorted({b["wav"] for b in loaded})
    strata = {
        w: ("discovery" if i % 5 in (0, 1) else "calibration" if i % 5 == 2 else "validation")
        for i, w in enumerate(uniq)
    }
    for b in loaded:
        b["stage"] = strata[b["wav"]]
        b["flac"] = str(bdir / "Wave" / b["wav"])  # wav path for NBD audio
    parts = {
        s: [b for b in loaded if b["stage"] == s]
        for s in ("discovery", "calibration", "validation")
    }
    if min(map(len, parts.values())) < 3:
        return None

    # per-bird discovery scaling (frozen convention)
    disc_frames = np.vstack([b["track"] for b in parts["discovery"]])
    mu, sd = disc_frames.mean(0), disc_frames.std(0) + 1e-9
    for b in loaded:
        b["z"] = (b["track"] - mu) / sd

    # ---- Stage A (all strata; boundaries are label-blind, strata only
    # organise the reporting — dev/validation reported separately)
    stage_a = {"dev": [], "validation": []}
    for b in loaded:
        r = _stage_a_bout(b)
        if r is None:
            continue
        stage_a["dev" if b["stage"] in ("discovery", "calibration") else "validation"].append(r)

    def _agg(rows):
        keys = ["n_proposals", "n_true", "n_ann"]
        out = {k: int(sum(r[k] for r in rows)) for k in keys}
        for tol in (1, 2, 3):
            out[f"tol{tol}"] = {
                m: round(
                    float(
                        np.mean([r[f"tol{tol}"][m] for r in rows if r[f"tol{tol}"][m] is not None])
                    ),
                    4,
                )
                if any(r[f"tol{tol}"][m] is not None for r in rows)
                else None
                for m in ("precision", "recall", "f1")
            }
        # pooled counts for exact per-bird F1 (not mean of bout F1s)
        for tol in (1, 2, 3):
            m = sum(r[f"tol{tol}"]["n_matched"] for r in rows)
            out[f"tol{tol}"]["f1_pooled"] = None
            P = m / out["n_proposals"] if out["n_proposals"] else None
            R = m / out["n_true"] if out["n_true"] else None
            out[f"tol{tol}"]["precision_pooled"] = round(P, 4) if P is not None else None
            out[f"tol{tol}"]["recall_pooled"] = round(R, 4) if R is not None else None
            out[f"tol{tol}"]["f1_pooled"] = round(2 * P * R / (P + R), 4) if P and R else None
        med = [
            r["boundary_abs_error_ms"]["median"]
            for r in rows
            if r["boundary_abs_error_ms"]["median"] is not None
        ]
        out["boundary_abs_error_ms_median_of_bout_medians"] = (
            round(float(np.median(med)), 1) if med else None
        )
        for k in ("gap_pollution", "mean_proposals_per_syllable", "syllable_coverage"):
            vals = [r[k] for r in rows if r[k] is not None]
            out[k] = round(float(np.mean(vals)), 4) if vals else None
        out["n_bouts"] = len(rows)
        return out

    stage_a_agg = {k: _agg(v) for k, v in stage_a.items() if v}

    # ---- Stage B: oracle-boundary same-class retrieval (cross-bout)
    def _segs(bouts):
        out = []
        for b in bouts:
            for a in b["ann"]:
                seg = b["z"][a["s"] : a["e"]]
                if len(seg) >= 3:
                    out.append(
                        {"label": a["label"], "bout": b["wav"], "z": seg, "dur": a["e"] - a["s"]}
                    )
        return out

    q_segs = _segs(parts["discovery"])
    v_segs = _segs(parts["validation"])
    per_tok, pairs_stat = [], []
    for q in q_segs:
        costs = []
        for j, v in enumerate(v_segs):
            if v["bout"] == q["bout"]:
                continue
            c = dtw_full(q["z"], v["z"])
            if np.isfinite(c):
                costs.append((c, j))
        if not costs:
            continue
        costs.sort(key=lambda r: r[0])
        r1 = 1 if v_segs[costs[0][1]]["label"] == q["label"] else 0
        r10 = 1 if any(v_segs[j]["label"] == q["label"] for _, j in costs[:10]) else 0
        per_tok.append((r1, r10))
        # rank statistic: cost of best same-class vs best different-class
        same = [c for c, j in costs if v_segs[j]["label"] == q["label"]]
        diff = [c for c, j in costs if v_segs[j]["label"] != q["label"]]
        if same and diff:
            pairs_stat.append(1 if min(same) < min(diff) else 0)
    prevalence = {}
    for v in v_segs:
        prevalence[v["label"]] = prevalence.get(v["label"], 0) + 1
    chance = max(prevalence.values()) / len(v_segs) if v_segs else None
    stage_b = {
        "n_query_segments": len(per_tok),
        "n_validation_segments": len(v_segs),
        "n_classes": len(prevalence),
        "recall1_oracle": _boot_rate([(a, 1) for a, _ in per_tok], b=BOOT_B),
        "recall10_oracle": _boot_rate([(b, 1) for _, b in per_tok], b=BOOT_B),
        "P_same_lt_diff_oracle": round(float(np.mean(pairs_stat)), 4) if pairs_stat else None,
        "majority_class_chance": round(chance, 4) if chance is not None else None,
    }

    # decision rules
    f1 = stage_a_agg.get("validation", {}).get(f"tol{TOL_PRIMARY}", {}).get("f1_pooled")
    r_prop = stage_a_agg.get("validation", {}).get(f"tol{TOL_PRIMARY}", {}).get("recall_pooled")
    r1 = stage_b["recall1_oracle"]["point"] if stage_b["recall1_oracle"] else None
    verdicts = {
        "nbd_closes_localisation_gap": bool(
            f1 is not None and r_prop is not None and f1 >= 0.6 and r_prop >= 0.8
        ),
        "representation_remaining_bottleneck": bool(r1 is not None and r1 < 0.25),
        "representation_adequate": bool(r1 is not None and r1 >= 0.5),
    }
    return {
        "n_bouts": len(loaded),
        "stage_a": stage_a_agg,
        "stage_b_oracle": stage_b,
        "verdicts": verdicts,
    }


def _self_test() -> None:
    """Synthetic tone syllables with silence gaps: NBD proposals land near
    the true on/offsets (Stage A mechanics) and oracle-boundary retrieval
    separates classes (Stage B mechanics)."""
    import numpy as np

    from bioacoustic_pipeline.segmentation import nbd_proposal as nbd

    rng = np.random.RandomState(0)
    sr = 32000
    parts = []
    bounds = []
    pos = 0
    for i in range(6):
        f0 = 2000 + 800 * (i % 2)  # two classes alternating
        n_s = int(sr * 0.12)
        t = np.arange(n_s) / sr
        syl = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.02 * rng.randn(n_s)
        gap = np.zeros(int(sr * 0.08))
        parts.extend([gap, syl])
        bounds += [pos + len(gap), pos + len(gap) + n_s]
        pos += len(gap) + n_s
    audio = np.concatenate(parts)
    sig = nbd.compute_signals(audio, sr, hop_ms=10.0)
    props = nbd.generate_proposals(sig)
    hop = int(sr * 10 / 1000)
    true_f = sorted(set(round(x / hop) for x in bounds))
    pairs, miss_t, _ = _match_boundaries(true_f, [p.frame for p in props], TOL_PRIMARY)
    assert len(pairs) >= 0.7 * len(true_f), (true_f, [p.frame for p in props])
    # Stage B mechanics: same-class segments cheaper than different-class
    segA = np.stack([np.sin(np.linspace(0, 6 * np.pi, 12))] * 12, axis=1)
    segB = np.stack([np.cos(np.linspace(0, 2 * np.pi, 12))] * 12, axis=1)
    a2 = segA + 0.01 * rng.randn(*segA.shape)
    assert dtw_full(segA, a2) < dtw_full(segA, segB)
    print(
        f"SELF-TEST OK (synthetic syllables: {len(pairs)}/{len(true_f)} boundaries "
        f"matched within {TOL_PRIMARY} frames; oracle retrieval separates classes)",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return

    per_bird = {}
    for bird in args.birds:
        r = _run_bird(bird, args)
        if r is None:
            print(f"[Bird{bird}] skipped (missing xml / insufficient split)", flush=True)
            continue
        per_bird[str(bird)] = r
        print(
            f"[Bird{bird}] A(val): F1 {r['stage_a'].get('validation', {}).get(f'tol{TOL_PRIMARY}', {}).get('f1_pooled')} "
            f"R_prop {r['stage_a'].get('validation', {}).get(f'tol{TOL_PRIMARY}', {}).get('recall_pooled')} | "
            f"B: R@1 {r['stage_b_oracle']['recall1_oracle']['point']}",
            flush=True,
        )

    report = {
        "architecture": "NBD proposes boundaries; fingerprint judges "
        "correspondence; free-start DTW's combined role is the historical "
        "comparator",
        "stage_a": "raw fixed-threshold NBD (floors 0.05/0.03, debounce 30 "
        "ms; NO calibrator — CPC calibrator human-only/firewalled; "
        "annotation-trained animal calibrator deferred as separately "
        "versioned)",
        "stage_b": "oracle-boundary same-class retrieval (GT spans -> "
        "fingerprint -> fixed-endpoint DTW ranking)",
        "historical_comparator": {
            "corrected_dev_D2_localisation": "0.528/0.250/0.500",
            "validation_known_query_iou_localisation": "0.230/0.144/0.118",
            "librispeech_word_recall1": "0.020 (control 0.089)",
        },
        "ceilings": "R_oracle (stage B) - R_proposal (stage A recall) = "
        "segmentation/proposal loss; R_proposal - R_selected = selection "
        "loss (R_selected DEFERRED: typed-NBD/HECB selection, separately "
        "preregistered)",
        "no_family_discovery": True,
        "framing": "an NBD proposal is an acoustic-change candidate, NOT a "
        "syllable/motor primitive/word-like unit; no semantic claims",
        "per_bird": per_bird,
        "stop_rule": "one run; no parameter changes after scores",
    }
    out = ROOT / "outputs" / "bengalese_boundary_decomposition.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({b: r["verdicts"] for b, r in per_bird.items()}, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
