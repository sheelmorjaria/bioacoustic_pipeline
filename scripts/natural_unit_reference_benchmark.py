#!/usr/bin/env python3
"""Natural-unit reference benchmark — annotated Bengalese song elements.

PREREGISTRATION (frozen before any validation score is computed; review
text quoted verbatim):

  "Song-element boundaries and class labels will remain hidden during
   acoustic discovery. Candidate intervals and families will be generated
   using the frozen representation and discovery procedure. Evaluation
   will quantify boundary overlap, same-class retrieval, family
   purity/completeness, and recovery of annotated units in previously
   unseen recording groups. Failure to recover the reference inventory
   will be interpreted as a limitation of the tested procedure, not
   absence of biological units."

MATERIAL: the on-disk BirdsongRecognition subset (Bird0/1/2; manual
note boundaries + class labels in xwaves Annotation.xml). Labels are
read ONLY inside scoring helpers; the discovery procedure (per-bird
z-scaling on discovery frames, subsequence-DTW seeding, random-fragment
5th-pct radius) never touches them — identical code path to the natural
benchmark, same v2 bout strata (40/20/40 discovery/calibration/validation
per bird over sorted unique wav files).

FROZEN PROCEDURE (no parameter is tuned on validation): representation
12-feature fingerprint contract (schema v2, species bengalese_finch),
discovery-stratum scaling, subseq-DTW radius = 5th percentile of
random-fragment cross-bout costs (re-estimated per bird), seeding and
family construction copied verbatim from bengalese_natural_benchmark.
Known property carried on record: the seeding stage's window distance
saturates (per-window z-normalised rows are not L2-normalised), so seed
ORDER is weakly discriminative and subseq-DTW + radius do the actual
filtering; families are whatever that frozen procedure emits.

FOUR SCORES (decision rules fixed now):
  S1 boundary localization
     (a) BLIND: every discovered family member interval vs its
         best-overlap annotated element — median IoU, IoU>=0.5
         fraction, boundary error (frames), gap fraction.
     (b) KNOWN-QUERY: natural query from discovery, top-1 DTW match
         per validation bout — fraction with IoU>=0.5 against a
         same-class annotated element. PASS: >=0.7 in >=2 of 3 birds.
  S2 identity matching (same-class retrieval)
     (a) RADIUS arm (frozen tau): accepted-and-correct precision and
         class recall on validation, legacy overlap rule — re-derives
         the natural-benchmark Test-1 endpoint.
     (b) RANK arm (radius-free, declared primary given the on-record
         abstention): top-1-per-bout same-class precision and recall.
         PASS (interpretive): precision >=0.7 and recall >=0.25 in
         >=2 of 3 birds.
  S3 grouping purity/completeness on UNSEEN recording groups
     purity: majority-label fraction over each family's validation
     top-1 members. completeness: per annotated class, fraction of
     supported validation occurrences captured (IoU>=0.5) by some
     family whose discovery majority label is that class. PASS:
     mean purity >=0.8 and mean completeness >=0.5 in >=2 of 3 birds.
  S4 contextual reuse (H3 component reuse)
     reference side (testability premise): per annotated class,
     distinct predecessors/successors, non-modal (prev,next)
     fraction, H(next|class)/log K. PREMISE ESTABLISHED if mean
     distinct successors >= 2 — else H3 is not testable here.
     discovery side (rank arm): among correctly recovered same-class
     matches, the fraction whose annotated (prev,next) context
     differs from the query's, vs the class's own context-diversity
     chance level (1 - Simpson). Context-neutral recovery: |obs -
     chance| <= 0.15.

UNCERTAINTY: per-bird point estimates with validation-bout clustered
bootstrap CIs (B=1000, percentile); three birds reported as a
distribution — no pooled CI theatre.

ON-RECORD CONTEXT carried in (not re-derived; CORRECTED 2026-10-20):
score diagnostics on development strata found P_same<diff 0.61-0.88 with
the frozen radius rejecting true renditions, and — after correction of a
cross-bout frame bug that had inflated localization to 0.83-0.97 —
localization 0.53/0.25/0.50 (formal mapping: Bird0 representation/metric;
Birds 1,2 localisation/path scoring); the annotation-calibrated
alternative ABSTAINED on all three birds (no operating point with
F_gap<=0.1, R>=0.25). The rank arm is primary BECAUSE of that abstention
record, decided before this run. This benchmark's own reconciliation
(ca. same criterion, same populations) is what exposed the D2 bug;
corrected D2 and this benchmark now agree on the poor localization.

STOP RULE: one run on the validation strata; no threshold, parameter,
or family-construction change after seeing these scores.

Usage:
    python scripts/natural_unit_reference_benchmark.py --self-test
    python scripts/natural_unit_reference_benchmark.py [--birds 0 1 2] [--bouts 90]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import subcall_term_discovery as std
from bengalese_natural_benchmark import (
    DATA,
    LENGTHS,
    SEED,
    _bout_job,
    parse_annotation,
)

BOOT_B = 1000
MIN_CLASS_OCC = 8  # S4 reference-side minimum validation support


# ---------------------------------------------------------------- helpers


def _iou(a_s, a_e, b_s, b_e):
    ov = min(a_e, b_e) - max(a_s, b_s)
    if ov <= 0:
        return 0.0, 0
    un = max(a_e, b_e) - min(a_s, b_s)
    return ov / un, ov


def _best_overlap(ann, s, e, rule="iou"):
    """Best annotated element for interval [s,e). rule='iou': max IoU;
    rule='legacy': max overlap (natural-benchmark Test-1 convention)."""
    best = None
    for a in ann:
        iou, ov = _iou(s, e, a["s"], a["e"])
        key = iou if rule == "iou" else ov
        if ov > 0 and (best is None or key > best[0]):
            best = (key, a, iou, ov)
    if best is None:
        return None
    _, a, iou, ov = best
    return {
        "label": a["label"],
        "iou": iou,
        "overlap": ov,
        "bnd_err": abs(s - a["s"]) + abs(e - a["e"]),
    }


def _contexts(bout):
    """(element, prev_label, next_label) for each annotated element, in
    bout order — the reference-side context axis for S4."""
    ann = sorted(bout["ann"], key=lambda a: a["s"])
    out = []
    for i, a in enumerate(ann):
        prev = ann[i - 1]["label"] if i > 0 else None
        nxt = ann[i + 1]["label"] if i + 1 < len(ann) else None
        out.append((a, prev, nxt))
    return out


def _boot_rate(per_bout, b=BOOT_B, seed=SEED):
    """percentile CI for sum(hits)/sum(total) over cluster (bout) resamples."""
    import numpy as np

    pairs = [(h, t) for h, t in per_bout if t > 0]
    if not pairs:
        return None
    rng = np.random.RandomState(seed)
    H = np.array([h for h, _ in pairs], float)
    T = np.array([t for _, t in pairs], float)
    rates = []
    for _ in range(b):
        idx = rng.randint(0, len(pairs), len(pairs))
        rates.append(H[idx].sum() / max(T[idx].sum(), 1e-9))
    return {
        "point": round(float(H.sum() / T.sum()), 4),
        "ci": [
            round(float(np.percentile(rates, 2.5)), 4),
            round(float(np.percentile(rates, 97.5)), 4),
        ],
        "n_clusters": len(pairs),
    }


# ---------------------------------------------------------------- scorers


def _score_s1a(fams):
    """S1(a): blind boundary localisation over all family member intervals."""
    import numpy as np

    ious, bnds, gaps, n_mem = [], [], 0, 0
    for f in fams:
        for ob, s, e in f["occ"] + f["val_rank"]:
            n_mem += 1
            hit = _best_overlap(ob["ann"], s, e, rule="iou")
            if hit is None:
                gaps += 1
                continue
            ious.append(hit["iou"])
            if hit["iou"] >= 0.5:
                bnds.append(hit["bnd_err"])
    return {
        "n_members": n_mem,
        "gap_fraction": round(gaps / n_mem, 4) if n_mem else None,
        "median_iou": round(float(np.median(ious)), 4) if ious else None,
        "iou_ge_0.5_fraction": round(float(np.mean([i >= 0.5 for i in ious])), 4) if ious else None,
        "median_boundary_error_frames": round(float(np.median(bnds)), 1) if bnds else None,
    }


def _score_known_query(parts, radius):
    """S1(b) + S2 + S4(discovery side): natural query per top class from
    discovery, top-1 subseq-DTW match per validation bout."""
    import numpy as np

    val_bouts = parts["validation"]
    class_counts: dict = {}
    for b in parts["discovery"]:
        for a in b["ann"]:
            class_counts[a["label"]] = class_counts.get(a["label"], 0) + 1
    top = sorted(class_counts, key=lambda k: -class_counts[k])[:6]
    loc_bout, any_bout, t2_rows, s4_rows = [], [], [], []
    for lab in top:
        qb = next(b for b in parts["discovery"] for a in b["ann"] if a["label"] == lab)
        qa = next(a for a in qb["ann"] if a["label"] == lab)
        qctx = next((p, n) for a, p, n in _contexts(qb) if a is qa)
        q = qb["z"][qa["s"] : qa["e"]]
        acc = corr_r = corr_rad = any_ov = 0
        n_val_occ = 0
        ctx_changes = recovered = 0
        for vb in val_bouts:
            n_val_occ += sum(1 for a in vb["ann"] if a["label"] == lab)
            c, s, e = std.subseq_dtw(q, vb["z"])
            hit = _best_overlap(vb["ann"], s, e, rule="iou")
            same = bool(hit and hit["label"] == lab and hit["iou"] >= 0.5)
            loc_bout.append((1 if same else 0, 1))
            if hit and hit["label"] == lab and hit["overlap"] > 0:
                any_ov += 1
                any_bout.append((1, 1))
            else:
                any_bout.append((0, 1))
            if same:
                corr_r += 1
                best_el = max(vb["ann"], key=lambda a: _iou(s, e, a["s"], a["e"])[0])
                vctx = next((p, n) for a, p, n in _contexts(vb) if a is best_el)
                recovered += 1
                if (vctx[0], vctx[1]) != (qctx[0], qctx[1]):
                    ctx_changes += 1
            if np.isfinite(c) and c <= radius:
                acc += 1
                legacy = _best_overlap(vb["ann"], s, e, rule="legacy")
                if legacy and legacy["label"] == lab and legacy["overlap"] >= 0.5 * (e - s):
                    corr_rad += 1
        t2_rows.append(
            {
                "label": lab,
                "query_bout": qb["wav"],
                "accepted_radius": acc,
                "correct_radius": corr_rad,
                "precision_radius": round(corr_rad / acc, 4) if acc else None,
                "recall_radius": round(corr_rad / n_val_occ, 4) if n_val_occ else None,
                "correct_rank": corr_r,
                "precision_rank": round(corr_r / len(val_bouts), 4) if val_bouts else None,
                "recall_rank": round(corr_r / n_val_occ, 4) if n_val_occ else None,
                "overlap_any_same_class_rank": any_ov,
                "supported_validation_occurrences": n_val_occ,
            }
        )
        s4_rows.append(
            {
                "label": lab,
                "query_context": list(qctx),
                "recovered_rank": recovered,
                "context_change_fraction": round(ctx_changes / recovered, 4) if recovered else None,
            }
        )
    # the pooled per-(class,bout) top-1 same-class rate serves BOTH S1(b)
    # localisation and the S2 rank arm — one IoU>=0.5 hit answers both;
    # any_overlap_rate is a RECONCILIATION readout vs the dev-strata D2
    # diagnostic (argmin-inside-true-interval), added after the first run
    # with no procedure or decision-rule change — same deterministic matches
    return _boot_rate(loc_bout), _boot_rate(any_bout), t2_rows, s4_rows


def _score_s3(fams, val_bouts):
    """S3: family purity and per-class completeness on unseen groups."""
    import numpy as np

    pur_rows = []
    val_occ_by_class: dict = {}
    for vb in val_bouts:
        for a in vb["ann"]:
            val_occ_by_class[a["label"]] = val_occ_by_class.get(a["label"], 0) + 1
    for f in fams:
        if f["majority_label"] is None:
            continue
        labels = []
        for vb, s, e in f["val_rank"]:
            hit = _best_overlap(vb["ann"], s, e, rule="legacy")
            labels.append(hit["label"] if hit else None)
        labeled = [x for x in labels if x is not None]
        if labeled:
            maj = max(set(labeled), key=labeled.count)
            pur_rows.append(labeled.count(maj) / len(labeled))
    # completeness per element identity (bout_id, element start frame)
    captured_by_label: dict = {}
    for f in fams:
        if f["majority_label"] is None:
            continue
        for vb, s, e in f["val_rank"]:
            for a in vb["ann"]:
                if a["label"] != f["majority_label"]:
                    continue
                iou, _ = _iou(s, e, a["s"], a["e"])
                if iou >= 0.5:
                    captured_by_label.setdefault(f["majority_label"], set()).add(
                        (vb["bout_id"], a["s"])
                    )
    comp_by_class = {}
    for lab, n in val_occ_by_class.items():
        comp_by_class[lab] = {
            "supported": n,
            "captured": len(captured_by_label.get(lab, ())),
            "completeness": round(len(captured_by_label.get(lab, ())) / n, 4) if n else None,
        }
    return {
        "n_families": len(fams),
        "n_labeled_families": sum(1 for f in fams if f["majority_label"] is not None),
        "mean_purity_validation": round(float(np.mean(pur_rows)), 4) if pur_rows else None,
        "mean_completeness_supported_classes": round(
            float(
                np.mean([c["completeness"] for c in comp_by_class.values() if c["supported"] > 0])
            ),
            4,
        )
        if comp_by_class
        else None,
        "classes_with_any_capture": sum(1 for c in comp_by_class.values() if c["captured"] > 0),
        "per_class": comp_by_class,
    }


def _score_s4_reference(val_bouts):
    """S4 reference side: does the annotated inventory itself exhibit
    contextual variability (the H3 testability premise)?"""
    import numpy as np

    ctx_dist: dict = {}
    for vb in val_bouts:
        for a, p, n in _contexts(vb):
            ctx_dist.setdefault(a["label"], []).append((p, n))
    ref_rows = []
    for lab, ctxs in sorted(ctx_dist.items()):
        if len(ctxs) < MIN_CLASS_OCC:
            continue
        succ = [n for _, n in ctxs if n is not None]
        K = len(set(succ))
        H = 0.0
        if K > 1:
            for v in set(succ):
                pv = succ.count(v) / len(succ)
                H += -pv * math.log(pv)
            H /= math.log(K)
        modal = max(set(ctxs), key=ctxs.count)
        ref_rows.append(
            {
                "label": lab,
                "n": len(ctxs),
                "distinct_predecessors": len({p for p, _ in ctxs if p is not None}),
                "distinct_successors": K,
                "nonmodal_context_fraction": round(1 - ctxs.count(modal) / len(ctxs), 4),
                "normalized_next_entropy": round(H, 4),
                "context_diversity_chance": round(
                    1 - sum((ctxs.count(c) / len(ctxs)) ** 2 for c in set(ctxs)), 4
                ),
            }
        )
    return {
        "n_classes": len(ref_rows),
        "mean_distinct_successors": round(
            float(np.mean([r["distinct_successors"] for r in ref_rows])), 3
        )
        if ref_rows
        else None,
        "premise_established": bool(
            ref_rows and np.mean([r["distinct_successors"] for r in ref_rows]) >= 2
        ),
        "per_class": ref_rows,
    }


def _attach_s4_chance(s4_rows, ref):
    ref_by_label = {r["label"]: r for r in ref["per_class"]}
    for row in s4_rows:
        r = ref_by_label.get(row["label"])
        row["context_diversity_chance"] = r["context_diversity_chance"] if r else None
        row["deviation_from_chance"] = (
            round(row["context_change_fraction"] - r["context_diversity_chance"], 4)
            if r and row["context_change_fraction"] is not None
            else None
        )


# ------------------------------------------------------- frozen procedure


def _frozen_families(parts, radius):
    """Family construction copied verbatim from bengalese_natural_benchmark
    (deterministic; labels untouched), plus validation-bout top-1 members."""
    import numpy as np

    disc_ev = sorted(parts["discovery"], key=lambda b: b["wav"])
    val_bouts = parts["validation"]
    seeds = []
    for L in LENGTHS:
        for qi, pa in enumerate(disc_ev):
            wq, sq = std._windows(pa["z"], L, L)
            if not len(wq):
                continue
            for ci, pb in enumerate(disc_ev):
                if qi == ci or len(pb["z"]) < L:
                    continue
                wc2, sc2 = std._windows(pb["z"], L, 2)
                if not len(wc2):
                    continue
                dm = np.sqrt(np.maximum(2 - 2 * (wq @ wc2.T), 0))
                best, bj = dm.min(axis=1), dm.argmin(axis=1)
                i_row = int(np.argmin(best))
                if best[i_row] <= radius:
                    seeds.append(
                        {
                            "L": L,
                            "dist": float(best[i_row]),
                            "a": pa,
                            "a_s": sq[i_row],
                            "b": pb,
                            "b_s": sc2[bj[i_row]],
                        }
                    )
    seeds.sort(key=lambda s: s["dist"])
    fams = []
    used: set = set()
    for s in seeds[:150]:
        if s["a"]["wav"] == s["b"]["wav"]:
            continue
        key = (s["a"]["wav"], s["a_s"], s["b"]["wav"])
        if key in used:
            continue
        q = s["a"]["z"][s["a_s"] : s["a_s"] + s["L"]]
        cost, cs, ce = std.subseq_dtw(q, s["b"]["z"])
        if not np.isfinite(cost) or cost > radius:
            continue
        occ = [(s["a"], s["a_s"], s["a_s"] + s["L"]), (s["b"], cs, ce)]
        for p in disc_ev:
            if p["wav"] in {s["a"]["wav"], s["b"]["wav"]}:
                continue
            c2, s2, e2 = std.subseq_dtw(q, p["z"])
            if np.isfinite(c2) and c2 <= radius:
                occ.append((p, s2, e2))
        if len({o[0]["wav"] for o in occ}) < 3:
            continue
        used.add(key)
        labels = []
        for ob, a, b_ in occ:
            hit = _best_overlap(ob["ann"], a, b_, rule="legacy")
            labels.append(hit["label"] if hit else None)
        labeled = [x for x in labels if x is not None]
        maj = max(set(labeled), key=labeled.count) if labeled else None
        fams.append(
            {
                "q": q,
                "occ": occ,
                "majority_label": maj,
                "purity_disc": round(labeled.count(maj) / len(labeled), 4) if labeled else None,
            }
        )
        if len(fams) >= 40:
            break
    # validation members: rank arm (top-1 per bout, any cost) and radius
    # arm recorded alongside — recovery in previously unseen groups
    for f in fams:
        f["val_rank"], f["val_radius"] = [], []
        for vb in val_bouts:
            c, s, e = std.subseq_dtw(f["q"], vb["z"])
            if np.isfinite(c):
                f["val_rank"].append((vb, s, e))
                if c <= radius:
                    f["val_radius"].append((vb, s, e))
    return fams


def _run_bird(parts, radius):
    """All four scores for one bird. Labels touched only in scoring."""
    fams = _frozen_families(parts, radius)
    s1b, any_ov, t2_rows, s4_rows = _score_known_query(parts, radius)
    ref = _score_s4_reference(parts["validation"])
    _attach_s4_chance(s4_rows, ref)
    val_occ = sum(len(b["ann"]) for b in parts["validation"])
    return {
        "radius": round(radius, 4),
        "counts": {
            "validation_bouts": len(parts["validation"]),
            "annotated_validation_elements": val_occ,
        },
        "s1_boundary": {
            "blind_families": _score_s1a(fams),
            "known_query_localization": s1b,
            "known_query_any_overlap_reconciliation": any_ov,
        },
        "s2_identity": {
            "radius_arm": t2_rows,
            "rank_arm_rate": s1b,  # same pooled trials as S1(b) by definition
        },
        "s3_purity_completeness": _score_s3(fams, parts["validation"]),
        "s4_contextual_reuse": {"reference_side": ref, "discovery_side_rank": s4_rows},
        "families_brief": [
            {
                "majority_label": f["majority_label"],
                "purity_discovery": f["purity_disc"],
                "n_disc_events": len({o[0]["wav"] for o in f["occ"]}),
                "n_val_rank_members": len(f["val_rank"]),
                "n_val_radius_members": len(f["val_radius"]),
            }
            for f in fams[:20]
        ],
    }


# ------------------------------------------------------------------ tests


def _self_test() -> None:
    """Synthetic annotated song, two classes in VARIED contexts, families
    constructed directly (seeding is exercised on real data; its window
    distance saturates on synthetic z-normalised rows by construction).

    Verifies the four scorers end-to-end against known ground truth:
    high localisation, high rank precision, non-trivial completeness, an
    ESTABLISHED S4 premise, and mostly context-varying recoveries.
    """
    import numpy as np

    rng = np.random.RandomState(0)
    # 12-D non-degenerate class templates (per-feature phase/amplitude)
    t = np.linspace(0, 1, 14)
    phA = rng.uniform(0, 2 * np.pi, 12)
    phB = rng.uniform(0, 2 * np.pi, 12)
    ampA = rng.uniform(0.5, 1.5, 12)
    ampB = rng.uniform(0.5, 1.5, 12)
    sylA = np.sin(2 * np.pi * (3 * t[:, None] + phA[None, :])) * ampA[None, :]
    sylB = np.cos(2 * np.pi * (5 * t[:, None] + phB[None, :])) * ampB[None, :]

    def _elem(which):
        return (sylA if which == "A" else sylB) + 0.05 * rng.randn(14, 12)

    loaded = []
    for k in range(15):
        chunks, ann = [], []
        pos = 0
        for _ in range(rng.randint(5, 8)):
            which = "A" if rng.rand() < 0.5 else "B"  # varied contexts by design
            chunks.append(0.3 * rng.randn(6, 12))  # inter-element gap
            chunks.append(_elem(which))
            ann.append({"label": which, "s": pos + 6, "e": pos + 20})
            pos += 20
        track = np.vstack(chunks)
        loaded.append(
            {
                "track": track,
                "valid": np.ones(len(track), bool),
                "ann": ann,
                "n_unsupported": 0,
                "wav": f"{k:02d}.wav",
                "bout_id": f"{k:02d}.wav#1",
                "stage": "discovery"
                if k % 5 in (0, 1)
                else "calibration"
                if k % 5 == 2
                else "validation",
            }
        )
    allf = np.vstack([b["track"] for b in loaded])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    for b in loaded:
        b["z"] = (b["track"] - mu) / sd
    parts = {
        s: [b for b in loaded if b["stage"] == s]
        for s in ("discovery", "calibration", "validation")
    }

    # families constructed directly from natural class exemplars (the
    # scoring contract: q, occ, majority_label, val_rank, val_radius).
    # THREE exemplars per class: completeness is structurally capped at
    # (families per class) top-1 captures per bout, and each bout holds
    # several same-class elements — multiplicity is part of the design.
    fams = []
    for lab in ("A", "B"):
        exemplars = []
        for b in parts["discovery"]:
            for a in b["ann"]:
                if a["label"] == lab and all(
                    (a["s"], b["wav"]) != (x[1]["s"], x[0]["wav"]) for x in exemplars
                ):
                    exemplars.append((b, a))
            if len(exemplars) >= 3:
                break
        for qb, qa in exemplars[:3]:
            q = qb["z"][qa["s"] : qa["e"]]
            occ = [(qb, qa["s"], qa["e"])]
            for p in parts["discovery"]:
                if p is qb:
                    continue
                c, s, e = std.subseq_dtw(q, p["z"])
                if np.isfinite(c) and c <= 0.5:
                    occ.append((p, s, e))
            val_rank, val_radius = [], []
            for vb in parts["validation"]:
                c, s, e = std.subseq_dtw(q, vb["z"])
                if np.isfinite(c):
                    val_rank.append((vb, s, e))
                    if c <= 0.5:
                        val_radius.append((vb, s, e))
            fams.append(
                {
                    "q": q,
                    "occ": occ,
                    "majority_label": lab,
                    "purity_disc": 1.0,
                    "val_rank": val_rank,
                    "val_radius": val_radius,
                }
            )

    s1a = _score_s1a(fams)
    assert s1a["iou_ge_0.5_fraction"] is not None and s1a["iou_ge_0.5_fraction"] >= 0.5, s1a
    s3 = _score_s3(fams, parts["validation"])
    assert s3["mean_completeness_supported_classes"] >= 0.5, s3["per_class"]
    ref = _score_s4_reference(parts["validation"])
    assert ref["premise_established"], ref
    s1b, any_ov, t2_rows, s4_rows = _score_known_query(parts, radius=0.5)
    assert s1b["point"] >= 0.7, s1b
    _attach_s4_chance(s4_rows, ref)
    changes = [
        r["context_change_fraction"] for r in s4_rows if r["context_change_fraction"] is not None
    ]
    assert changes and float(np.mean(changes)) >= 0.3, s4_rows
    print(
        "SELF-TEST OK (synthetic varied-context song: "
        f"blind IoU>=0.5 {s1a['iou_ge_0.5_fraction']}, localization {s1b['point']}, "
        f"completeness {s3['mean_completeness_supported_classes']}, "
        f"context change {np.mean(changes):.2f})",
        flush=True,
    )


# ------------------------------------------------------------------- main


def _real_run(args) -> None:
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from bengalese_natural_benchmark import HOP_MS, SR, WIN_MS

    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    rng = np.random.RandomState(SEED)
    per_bird = {}

    for bird in args.birds:
        bdir = DATA / f"Bird{bird}"
        xml = bdir / "Annotation.xml"
        if not xml.exists():
            print(f"[Bird{bird}] missing {xml}", flush=True)
            continue
        bouts_raw = parse_annotation(xml)[: args.bouts]
        jobs = [(bdir / "Wave" / w, notes, hop, win, w) for w, notes in bouts_raw]
        with ProcessPoolExecutor(max_workers=24) as ex:
            loaded = [r for r in ex.map(_bout_job, jobs, chunksize=2) if r]
        uniq_wavs = sorted({b["wav"] for b in loaded})
        strata = {}
        for i, w in enumerate(uniq_wavs):
            strata[w] = (
                "discovery" if i % 5 in (0, 1) else "calibration" if i % 5 == 2 else "validation"
            )
        for b in loaded:
            b["stage"] = strata[b["wav"]]
        seq_count: dict = {}
        for b in loaded:
            seq_count[b["wav"]] = seq_count.get(b["wav"], 0) + 1
            b["bout_id"] = f"{b['wav']}#{seq_count[b['wav']]}"
        parts = {
            s: [b for b in loaded if b["stage"] == s]
            for s in ("discovery", "calibration", "validation")
        }
        if min(len(parts["discovery"]), len(parts["calibration"]), len(parts["validation"])) < 3:
            print(f"[Bird{bird}] insufficient split support — skipped", flush=True)
            continue

        # frozen procedure-portable radius (same draw order as the natural
        # benchmark: 40 draws, 6 cross-bout companions each, 5th percentile)
        disc_frames = np.vstack([b["track"] for b in parts["discovery"]])
        mu, sd = disc_frames.mean(0), disc_frames.std(0) + 1e-9
        for b in loaded:
            b["z"] = (b["track"] - mu) / sd
        cal = parts["calibration"]
        costs = []
        for i in range(40):
            p = cal[rng.randint(0, len(cal))]
            L = LENGTHS[i % len(LENGTHS)]
            if len(p["z"]) < L + 2:
                continue
            s0 = int(rng.randint(0, len(p["z"]) - L))
            for k2 in rng.randint(0, len(cal), size=6):
                p2 = cal[k2]
                if p2["wav"] == p["wav"]:
                    continue
                costs.append(std.subseq_dtw(p["z"][s0 : s0 + L], p2["z"])[0])
        costs = [c for c in costs if np.isfinite(c)]
        radius = float(np.percentile(costs, 5.0))
        print(
            f"[Bird{bird}] bouts {len(loaded)} ({len(parts['discovery'])}/"
            f"{len(parts['calibration'])}/{len(parts['validation'])}) radius {radius:.4f}",
            flush=True,
        )

        per_bird[str(bird)] = _run_bird(parts, radius)
        print(f"[Bird{bird}] scored", flush=True)

    report = {
        "preregistration": __doc__.split("PREREGISTRATION")[1].split("MATERIAL")[0].strip(),
        "labels_hidden_during_discovery": True,
        "interpretation_clause": "Failure to recover the reference inventory "
        "will be interpreted as a limitation of the tested procedure, not "
        "absence of biological units.",
        "on_record_context": {
            "score_diagnostics_dev_strata": "CORRECTED 2026-10-20: "
            "P_same<diff 0.61-0.88, frozen radius rejects true renditions, "
            "localization 0.53/0.25/0.50 after the cross-bout frame-bug "
            "correction (original inflated values 0.83-0.97 preserved in "
            "outputs/bengalese_score_diagnostics.json; formal mapping "
            "Bird0 representation/metric, Birds 1/2 localisation/path "
            "scoring)",
            "annotation_calibrated_alternative": "ABSTENTION on all three "
            "birds (no operating point with F_gap<=0.1, R>=0.25)",
            "seeding_note": "seed ORDER is weakly discriminative (window "
            "rows z-normalised but not L2-normalised); subseq-DTW + radius "
            "do the actual filtering — families are whatever the frozen "
            "procedure emits",
            "correction_provenance": "this benchmark's reconciliation "
            "(outputs/nurb_reconciliation.json + instrumented replay) "
            "exposed the D2 bug; corrected D2 and this benchmark agree on "
            "poor localization",
        },
        "per_bird": per_bird,
        "stop_rule": "one run on validation strata; no parameter changes after seeing these scores",
    }
    out = ROOT / "outputs" / "natural_unit_reference_benchmark.json"
    out.write_text(json.dumps(report, indent=1))
    summary = {}
    for b, v in per_bird.items():
        summary[b] = {
            "s1a": v["s1_boundary"]["blind_families"],
            "s1b_localization": v["s1_boundary"]["known_query_localization"],
            "s1b_any_overlap": v["s1_boundary"]["known_query_any_overlap_reconciliation"],
            "s2_radius_accepted_total": sum(
                r["accepted_radius"] for r in v["s2_identity"]["radius_arm"]
            ),
            "s3": {
                k: v["s3_purity_completeness"][k]
                for k in (
                    "n_families",
                    "mean_purity_validation",
                    "mean_completeness_supported_classes",
                )
            },
            "s4_premise": v["s4_contextual_reuse"]["reference_side"]["premise_established"],
            "s4_context_change": [
                r["context_change_fraction"]
                for r in v["s4_contextual_reuse"]["discovery_side_rank"]
            ],
        }
    print(json.dumps(summary, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run(args)


if __name__ == "__main__":
    main()
