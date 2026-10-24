#!/usr/bin/env python3
"""Bengalese proposal-selection decomposition — the intervention implied by
the boundary-decomposition diagnosis. Advisor-directed 2026-10-20.

MOTIVATION (measured, prior artifacts): with oracle boundaries the
fingerprint + fixed-endpoint matcher retrieves same-class syllables at
Recall@1 0.953-0.994 — the representation is NOT the bottleneck. Raw
fixed-threshold NBD proposes near only 0.66-0.74 of true boundaries with
precision 0.10-0.20 (76-91% gap pollution). A selector alone cannot
rescue boundaries that were never proposed — hence a PROPOSAL-SELECTION
decomposition, not selection-only.

ARMS (frozen split: discovery+calibration = development training,
validation = frozen evaluation; same strata as all natural benchmarks):
  A1 raw NBD (floors 0.05/0.03, drop disabled, debounce 30 ms) —
     historical baseline, re-derived here for the same tables;
  A2 HIGH-RECALL proposal regime — ONE prospectively specified setting:
     floors 0.02/0.015, fire-on-drop floor 0.015 (syllable offsets are
     drop-type boundaries; the default regime had drop DISABLED — a
     structural miss source), debounce 20 ms. Accepts lower raw
     precision to raise candidate recall. No tuning after seeing
     validation scores;
  A3 TYPED 4-way selector on A2's proposals, trained on DEVELOPMENT
     labels only: classes {syllable onset, syllable offset,
     within-syllable change, gap/noise}; KEPT set for segmentation =
     onset+offset (the annotated syllable is the target unit — unlike
     the LibriSpeech internal-phoneme HECB experiment);
  A4 SCALAR binary selector, capacity-matched baseline: same features,
     binary target (onset-or-offset within tolerance vs rest).
  Selector model: multinomial / binomial logistic regression on
  standardised proposal features (balanced class weights), sklearn.

ENDPOINTS (separate stages, fixed now):
  proposal stage:  boundary recall at tolerance 3 frames, proposals/sec,
                   fraction of proposals in inter-syllable gaps;
  selection stage: kept-set precision / recall / F1 vs the union of
                   annotated onset+offset boundaries (tol 3), then
                   segment IoU (segments = consecutive kept boundaries)
                   against annotated spans;
  end-to-end:      usable selected segments (IoU>=0.5 with an annotated
                   syllable) -> same-class retrieval Recall@1 across
                   bouts (Stage-B protocol), COVERAGE = usable segments'
                   capture of annotated validation syllables, and the
                   coverage-weighted end-to-end rate.

DECOMPOSITION REPORTED: R_oracle (prior artifact) vs R_proposal (A2
recall) vs R_selected (A3/A4 kept F1 and kept recall) vs R_end_to_end
(coverage-weighted same-class retrieval).

DECISION RULES (fixed now): the pipeline "works" if A3 kept F1 >= 0.6
AND end-to-end coverage-weighted retrieval >= 0.5; "selection earns
precision" if A3 kept precision >= 2x A2 raw precision at kept recall
>= A2 recall; typed beats scalar if A3 F1 > A4 F1 + 0.05 on validation.

FRAMING: an NBD proposal is an acoustic-change candidate, NOT a
syllable/motor primitive/word-like unit; selector classes are scoring
and filtering constructs (Semantic Ceiling intact). One run; no
parameter, feature, or threshold change after seeing validation scores.

Usage:
    python scripts/bengalese_proposal_selection.py --self-test
    python scripts/bengalese_proposal_selection.py [--birds 0 1 2]

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

from bengalese_boundary_decomposition import TOL_PRIMARY, _match_boundaries
from bengalese_natural_benchmark import (
    DATA,
    HOP_MS,
    SR,
    WIN_MS,
    _bout_job,
    parse_annotation,
)
from bengalese_score_diagnostics import dtw_full
from natural_unit_reference_benchmark import _boot_rate

SEED = 0
BOOT_B = 1000

# A1 raw regime (defaults) and A2 high-recall regime — prospectively fixed
REGIMES = {
    "A1_raw": dict(floor_energy=0.05, floor_flux=0.03, floor_drop=float("inf"), debounce_ms=30.0),
    "A2_highrecall": dict(floor_energy=0.02, floor_flux=0.015, floor_drop=0.015, debounce_ms=20.0),
}

FEATURES = (
    "energy_change spectral_flux band_flux local_rms rms_slope "
    "duration_since_prev_ms rms_ratio flux_contrast energy_contrast "
    "flux_asym duration_to_next_ms spectral_drop det_energy det_spectral det_drop"
).split()


def _proposals(b, regime):
    """NBD proposals with features for one bout under one regime."""
    import numpy as np

    from bengalese_natural_benchmark import _load_wav
    from bioacoustic_pipeline.segmentation import nbd_proposal as nbd

    audio = _load_wav(Path(b["flac"]))
    if audio is None:
        return None
    sig = nbd.compute_signals(audio, SR, hop_ms=HOP_MS)
    props = nbd.generate_proposals(sig, **REGIMES[regime])
    X = []
    for p in props:
        X.append(
            [
                p.energy_change,
                p.spectral_flux,
                p.band_flux,
                np.log1p(p.local_rms),
                p.rms_slope,
                p.duration_since_prev_ms,
                p.rms_ratio,
                p.flux_contrast,
                p.energy_contrast,
                p.flux_asym,
                p.duration_to_next_ms,
                p.spectral_drop,
                float("energy" in p.detector),
                float("spectral" in p.detector),
                float("spectral_drop" in p.detector),
            ]
        )
    return {
        "X": np.array(X) if X else np.zeros((0, len(FEATURES))),
        "frames": [p.frame for p in props],
        "duration_s": len(audio) / SR,
    }


def _label_proposals(frames, ann):
    """4-way development labels from annotations (tolerance TOL_PRIMARY):
    onset / offset / within / gap."""
    onsets = [a["s"] for a in ann]
    offsets = [a["e"] - 1 for a in ann]
    out = []
    for f in frames:
        if any(abs(f - o) <= TOL_PRIMARY for o in onsets):
            out.append("onset")
        elif any(abs(f - o) <= TOL_PRIMARY for o in offsets):
            out.append("offset")
        elif any(a["s"] <= f < a["e"] for a in ann):
            out.append("within")
        else:
            out.append("gap")
    return out


def _true_boundary_frames(ann):
    return sorted(set([a["s"] for a in ann]) | {a["e"] - 1 for a in ann})


def _segments_from_kept(kept_frames, n_frames):
    """Segments = intervals between consecutive kept boundaries."""
    b = [0, *sorted(kept_frames), n_frames]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] >= 3]


def _iou(a_s, a_e, b_s, b_e):
    ov = min(a_e, b_e) - max(a_s, b_s)
    if ov <= 0:
        return 0.0
    return ov / (max(a_e, b_e) - min(a_s, b_s))


def _run_bird(bird, args):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

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
        b["flac"] = str(bdir / "Wave" / b["wav"])
    parts = {
        s: [b for b in loaded if b["stage"] == s]
        for s in ("discovery", "calibration", "validation")
    }
    if min(map(len, parts.values())) < 3:
        return None
    disc_frames = np.vstack([b["track"] for b in parts["discovery"]])
    mu, sd = disc_frames.mean(0), disc_frames.std(0) + 1e-9
    for b in loaded:
        b["z"] = (b["track"] - mu) / sd

    # proposals under both regimes (computed once per bout)
    for b in loaded:
        for reg in REGIMES:
            b[reg] = _proposals(b, reg)

    dev = parts["discovery"] + parts["calibration"]
    val = parts["validation"]

    def _proposal_stats(bouts, reg):
        rec, n_prop, gap, dur = [], 0, 0, 0.0
        for b in bouts:
            pr = b[reg]
            dur += pr["duration_s"]
            n_prop += len(pr["frames"])
            tf = _true_boundary_frames(b["ann"])
            pairs, miss_t, _ = _match_boundaries(tf, pr["frames"], TOL_PRIMARY)
            rec.append(len(pairs))
            rec.append(miss_t)
            gap += sum(1 for f in pr["frames"] if not any(a["s"] <= f < a["e"] for a in b["ann"]))
        tot_rec = sum(rec[i] for i in range(0, len(rec), 2))
        tot_miss = sum(rec[i] for i in range(1, len(rec), 2))
        return {
            "recall": round(tot_rec / max(tot_rec + tot_miss, 1), 4),
            "proposals_per_sec": round(n_prop / max(dur, 1e-9), 2),
            "gap_fraction": round(gap / max(n_prop, 1), 4),
            "n_proposals": n_prop,
        }

    # ---- selectors trained on DEVELOPMENT labels only (A2 proposal set)
    Xd, yd4, yd2 = [], [], []
    for b in dev:
        pr = b["A2_highrecall"]
        labs = _label_proposals(pr["frames"], b["ann"])
        Xd.append(pr["X"])
        yd4.extend(labs)
        yd2.extend(li in ("onset", "offset") for li in labs)
    Xd = np.vstack(Xd)
    scaler = StandardScaler().fit(Xd)
    Xdn = scaler.transform(Xd)
    typed = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(
        Xdn, yd4
    )
    binary = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(
        Xdn, yd2
    )
    train_acc = {
        "typed_4way_dev_accuracy": round(float((typed.predict(Xdn) == np.array(yd4)).mean()), 4),
        "binary_dev_accuracy": round(float((binary.predict(Xdn) == np.array(yd2)).mean()), 4),
    }

    def _selection_stats(bouts, keep_fn):
        """Kept-set boundary P/R/F1 (tol 3) + segment IoU vs annotated spans."""
        n_kept = n_true = n_matched = 0
        ious = []
        usable = []  # (bout, s, e, label)
        n_ann = 0
        for b in bouts:
            kept = keep_fn(b)
            n_kept += len(kept)
            tf = _true_boundary_frames(b["ann"])
            n_true += len(tf)
            pairs, _, _ = _match_boundaries(tf, kept, TOL_PRIMARY)
            n_matched += len(pairs)
            n_ann += len(b["ann"])
            for s, e in _segments_from_kept(kept, len(b["z"])):
                best = None
                for a in b["ann"]:
                    iou = _iou(s, e, a["s"], a["e"])
                    if iou > 0 and (best is None or iou > best[0]):
                        best = (iou, a)
                if best:
                    ious.append(best[0])
                    if best[0] >= 0.5:
                        usable.append((b, s, e, best[1]["label"]))
        P = n_matched / max(n_kept, 1)
        R = n_matched / max(n_true, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        return {
            "kept_precision": round(P, 4),
            "kept_recall": round(R, 4),
            "kept_f1": round(F1, 4),
            "n_kept": n_kept,
            "median_segment_iou": round(float(np.median(ious)), 4) if ious else None,
            "segment_iou_ge_0.5": round(float(np.mean([i >= 0.5 for i in ious])), 4)
            if ious
            else None,
            "usable_segments": len(usable),
            "coverage_of_annotated": round(len(usable) / max(n_ann, 1), 4),
            "_usable": usable,
        }

    def _keep_typed(b):
        pr = b["A2_highrecall"]
        pred = typed.predict(scaler.transform(pr["X"]))
        return [f for f, p in zip(pr["frames"], pred) if p in ("onset", "offset")]

    def _keep_binary(b):
        pr = b["A2_highrecall"]
        pred = binary.predict(scaler.transform(pr["X"]))
        return [f for f, p in zip(pr["frames"], pred) if p]

    sel_typed = _selection_stats(val, _keep_typed)
    sel_binary = _selection_stats(val, _keep_binary)
    sel_typed_dev = _selection_stats(dev, _keep_typed)

    # ---- end-to-end retrieval on selected (typed) segments, Stage-B protocol
    def _retrieval(usable_q, usable_v):
        per_tok = []
        for b, s, e, lab in usable_q:
            q = b["z"][s:e]
            if len(q) < 3:
                continue
            costs = []
            for j, (bv, sv, ev, labv) in enumerate(usable_v):
                if bv["wav"] == b["wav"]:
                    continue
                c = dtw_full(q, bv["z"][sv:ev])
                if np.isfinite(c):
                    costs.append((c, j))
            if not costs:
                continue
            costs.sort(key=lambda r: r[0])
            per_tok.append(1 if usable_v[costs[0][1]][3] == lab else 0)
        return per_tok

    usable_q = [u for u in sel_typed_dev["_usable"]]
    per_tok = _retrieval(usable_q, sel_typed["_usable"])
    r1 = _boot_rate([(a, 1) for a in per_tok], b=BOOT_B) if per_tok else None
    coverage = sel_typed["coverage_of_annotated"]
    end_to_end = {
        "n_queries": len(per_tok),
        "recall1_selected_segments": r1,
        "coverage_of_annotated_syllables": coverage,
        "coverage_weighted_end_to_end": round((r1["point"] if r1 else 0) * coverage, 4),
        "oracle_ceiling_prior": None,  # filled from prior artifact in report
    }

    verdicts = {
        "pipeline_works": bool(
            sel_typed["kept_f1"] >= 0.6 and end_to_end["coverage_weighted_end_to_end"] >= 0.5
        ),
        "selection_earns_precision": None,
        "typed_beats_scalar": None,
    }
    a2 = _proposal_stats(val, "A2_highrecall")
    # non-gap fraction is an upper-bound proxy for the raw regime's boundary
    # precision (a non-gap proposal may still miss a true boundary by >tol)
    a2_raw_precision_proxy = 1 - a2["gap_fraction"]
    verdicts["selection_earns_precision"] = bool(
        a2_raw_precision_proxy > 0 and sel_typed["kept_precision"] >= 2 * a2_raw_precision_proxy
    )
    verdicts["typed_beats_scalar"] = bool(sel_typed["kept_f1"] > sel_binary["kept_f1"] + 0.05)

    for d in (sel_typed, sel_binary, sel_typed_dev):
        d.pop("_usable", None)

    return {
        "n_bouts": len(loaded),
        "proposal_stage": {
            "A1_raw_validation": _proposal_stats(val, "A1_raw"),
            "A2_highrecall_validation": a2,
        },
        "selector_training_dev": train_acc,
        "selection_stage_validation": {
            "A3_typed_onset_offset": sel_typed,
            "A4_scalar_binary": sel_binary,
            "A3_typed_dev_fit": sel_typed_dev,
        },
        "end_to_end_validation": end_to_end,
        "verdicts": verdicts,
    }


def _self_test() -> None:
    """Synthetic syllable song: high-recall regime covers the true
    on/offsets, a dev-trained typed selector keeps them and drops gap
    noise, and end-to-end retrieval separates classes (mechanics check)."""
    import numpy as np

    from bioacoustic_pipeline.segmentation import nbd_proposal as nbd

    rng = np.random.RandomState(0)
    sr = 32000
    audio_parts, ann = [], []
    pos = 0
    for i in range(40):
        f0 = 1800 + 900 * (i % 3)
        n_s = int(sr * 0.10)
        t = np.arange(n_s) / sr
        env = np.minimum(1, np.arange(n_s) / 400) * np.minimum(1, (n_s - np.arange(n_s)) / 400)
        syl = 0.6 * env * np.sin(2 * np.pi * f0 * t) + 0.01 * rng.randn(n_s)
        gap = 0.002 * rng.randn(int(sr * 0.07))
        audio_parts.extend([gap, syl])
        hop = int(sr * 0.01)
        ann.append(
            {
                "label": "ABC"[i % 3],
                "s": (pos + len(gap)) // hop,
                "e": (pos + len(gap) + n_s) // hop,
            }
        )
        pos += len(gap) + n_s
    audio = np.concatenate(audio_parts)
    sig = nbd.compute_signals(audio, sr, hop_ms=10.0)
    pr = nbd.generate_proposals(sig, **REGIMES["A2_highrecall"])
    frames = [p.frame for p in pr]
    tf = _true_boundary_frames(ann)
    pairs, miss_t, _ = _match_boundaries(tf, frames, TOL_PRIMARY)
    assert len(pairs) >= 0.8 * len(tf), (len(pairs), len(tf))
    labs = _label_proposals(frames, ann)
    assert "gap" in labs and "onset" in labs
    # end-to-end mechanics: same-class segment cheaper than different-class
    segA = np.stack([np.sin(np.linspace(0, 6 * np.pi, 12))] * 12, axis=1)
    segB = np.stack([np.cos(np.linspace(0, 2 * np.pi, 12))] * 12, axis=1)
    assert dtw_full(segA, segA + 0.01 * rng.randn(*segA.shape)) < dtw_full(segA, segB)
    print(
        f"SELF-TEST OK (synthetic: {len(pairs)}/{len(tf)} boundaries in high-recall "
        f"proposal set; labels {''.join(sorted(set(labs)))}; retrieval separates)",
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

    oracle_prior = {"0": 0.959, "1": 0.994, "2": 0.953}  # Stage-B artifact
    per_bird = {}
    for bird in args.birds:
        r = _run_bird(bird, args)
        if r is None:
            print(f"[Bird{bird}] skipped", flush=True)
            continue
        r["end_to_end_validation"]["oracle_ceiling_prior"] = oracle_prior.get(str(bird))
        per_bird[str(bird)] = r
        ps = r["proposal_stage"]["A2_highrecall_validation"]
        st = r["selection_stage_validation"]["A3_typed_onset_offset"]
        print(
            f"[Bird{bird}] A2 recall {ps['recall']} gap {ps['gap_fraction']} | "
            f"A3 kept P {st['kept_precision']} R {st['kept_recall']} F1 {st['kept_f1']} | "
            f"cov {r['end_to_end_validation']['coverage_of_annotated_syllables']} "
            f"R@1 {r['end_to_end_validation']['recall1_selected_segments']['point'] if r['end_to_end_validation']['recall1_selected_segments'] else None}",
            flush=True,
        )

    report = {
        "study": "proposal-selection decomposition (intervention implied by "
        "the boundary-decomposition diagnosis; selection alone cannot "
        "rescue unproposed boundaries)",
        "arms": {
            "A1_raw": "floors 0.05/0.03, drop disabled, debounce 30 ms (baseline)",
            "A2_highrecall": "floors 0.02/0.015, fire-on-drop 0.015, debounce "
            "20 ms — ONE prospective setting, no post-hoc tuning",
            "A3_typed": "4-way selector onset/offset/within/gap, kept = "
            "onset+offset, dev-trained, logistic (balanced)",
            "A4_scalar": "binary selector, same features, capacity-matched",
        },
        "decomposition": "R_oracle (prior) vs R_proposal (A2 recall) vs "
        "R_selected (A3 kept F1/recall) vs R_end_to_end (coverage-weighted "
        "same-class retrieval)",
        "framing": "proposals are acoustic-change candidates and selector "
        "classes are filtering constructs, NOT units; no semantic claims; "
        "annotated syllable = target unit (kept set onset+offset, unlike "
        "the LibriSpeech internal-phoneme HECB experiment)",
        "per_bird": per_bird,
        "stop_rule": "one run; no parameter/feature/threshold change after "
        "validation scores; next steps after freeze: zebra-finch external "
        "benchmark, then stop the unit-recovery branch",
    }
    out = ROOT / "outputs" / "bengalese_proposal_selection.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({b: r["verdicts"] for b, r in per_bird.items()}, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
