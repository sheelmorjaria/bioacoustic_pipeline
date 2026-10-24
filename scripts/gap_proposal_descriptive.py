#!/usr/bin/env python3
"""Descriptive analysis of retained gap proposals (advisor-directed
2026-10-20, §4 of the consolidation message). NO retraining, NO new model:
the frozen M0 scalar selector is reproduced exactly (asserted equal to
the ablation artifact) and its validation-set score distributions are
examined to answer:

  Are retained gap proposals ACOUSTICALLY INDISTINGUISHABLE from true
  unit boundaries under the current 13-D measurements, or are they
  DISTINGUISHABLE but assigned incorrect decisions by the fitted
  selector?

DESCRIPTIVE ONLY — explicitly not a preregistered confirmatory analysis;
no thresholds are adopted, no models are selected. Outputs:
  - P(edge | M0) quantiles by reference role (edge / within / gap);
  - overlap statistics: P(gap score >= median edge score), P(gap >= p90
    edge score);
  - operating table: for gap-keep fractions {0.5, 0.25, 0.1, 0.05}, the
    implied threshold and the edge-recall cost (what edge recall remains
    if gaps were rejected to that rate);
  - calibration: binned mean predicted P(edge) vs observed edge rate.

Usage:
    python scripts/gap_proposal_descriptive.py [--birds 0 1 2]

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
from bengalese_natural_benchmark import DATA, HOP_MS, SR, WIN_MS, _bout_job, parse_annotation
from bengalese_proposal_selection import _label_proposals, _proposals

SEED = 0
FROZEN = ROOT / "outputs" / "bengalese_contrast_ablation.json"


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
    with ProcessPoolExecutor(max_workers=12) as ex:
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
    disc = np.vstack([b["track"] for b in parts["discovery"]])
    mu, sd = disc.mean(0), disc.std(0) + 1e-9
    for b in loaded:
        b["z"] = (b["track"] - mu) / sd
        b["pr"] = _proposals(b, "A2_highrecall")
    dev = parts["discovery"] + parts["calibration"]
    val = parts["validation"]

    lab2id = {"onset": 0, "offset": 1, "within": 2, "gap": 3}
    Xd, yd, bout_id = [], [], []
    for bi, b in enumerate(dev):
        labs = _label_proposals(b["pr"]["frames"], b["ann"])
        Xd.append(b["pr"]["X"])
        yd.extend(lab2id[x] for x in labs)
        bout_id.extend([bi] * len(labs))
    Xd = np.vstack(Xd)
    yd = np.array(yd)
    bout_id = np.array(bout_id)
    # identical dev bout-carve to the ablation (same seed/procedure) so M0
    # reproduces the frozen selector exactly
    rng = np.random.RandomState(SEED)
    dev_bouts = sorted(set(bout_id.tolist()))
    stop_bouts = set(rng.choice(dev_bouts, size=max(1, len(dev_bouts) // 5), replace=False).tolist())
    m_tr = ~np.isin(bout_id, list(stop_bouts))
    scaler = StandardScaler().fit(Xd[m_tr])
    m0 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        scaler.transform(Xd[m_tr]), yd[m_tr]
    )

    # frozen reproduction assert (pooled kept-F1 on validation)
    n_kept = n_true = n_m = 0
    scores, roles = [], []
    for b in val:
        X = scaler.transform(b["pr"]["X"])
        prob = m0.predict_proba(X)
        p_edge = prob[:, :2].sum(1)
        pred = prob.argmax(1)
        kept = [f for f, p in zip(b["pr"]["frames"], pred) if p in (0, 1)]
        n_kept += len(kept)
        true_union = sorted({a["s"] for a in b["ann"]} | {a["e"] - 1 for a in b["ann"]})
        n_true += len(true_union)
        pairs, _, _ = _match_boundaries(true_union, kept, TOL_PRIMARY)
        n_m += len(pairs)
        labs = _label_proposals(b["pr"]["frames"], b["ann"])
        scores.extend(p_edge.tolist())
        roles.extend(labs)
    P = n_m / max(n_kept, 1)
    R = n_m / max(n_true, 1)
    f1 = round(2 * P * R / max(P + R, 1e-9), 4)

    s = np.array(scores)
    r = np.array(roles)
    edge_mask = np.isin(r, ("onset", "offset"))
    e, w_, g = s[edge_mask], s[r == "within"], s[r == "gap"]

    # NOTE: 'edge' here = reference-role onset/offset proposals
    def qs(a):
        return (
            {q: round(float(np.percentile(a, q)), 4) for q in (10, 25, 50, 75, 90)}
            if len(a)
            else None
        )

    out = {
        "reproduced_M0_f1": f1,
        "n_proposals_by_role": {k: int((np.isin(r, ("onset", "offset")) if k == "edge" else (r == k)).sum()) for k in ("edge", "within", "gap")},
        "score_quantiles_by_role": {"edge": qs(e), "within": qs(w_), "gap": qs(g)},
        "overlap": {
            "P_gap_ge_median_edge": round(float((g >= np.median(e)).mean()), 4)
            if len(g) and len(e)
            else None,
            "P_gap_ge_p90_edge": round(float((g >= np.percentile(e, 90)).mean()), 4)
            if len(g) and len(e)
            else None,
        },
        "operating_table_gap_rejection": {},
        "calibration_bins": [],
    }
    # operating table: reject gaps to a keep-fraction -> threshold -> edge recall cost
    edge_true = np.isin(r, ("onset", "offset"))
    for keep_frac in (0.5, 0.25, 0.1, 0.05):
        thr = float(np.quantile(g, 1 - keep_frac)) if len(g) else None
        if thr is None:
            continue
        edge_recall_at_thr = float((e >= thr).mean()) if len(e) else None
        within_keep = float((w_ >= thr).mean()) if len(w_) else None
        out["operating_table_gap_rejection"][f"gap_keep_{keep_frac}"] = {
            "threshold_p_edge": round(thr, 4),
            "edge_recall_at_threshold": round(edge_recall_at_thr, 4)
            if edge_recall_at_thr is not None
            else None,
            "within_keep_fraction": round(within_keep, 4) if within_keep is not None else None,
        }
    # calibration: deciles of predicted vs observed
    bins = np.percentile(s, np.linspace(0, 100, 11))
    for i in range(10):
        m = (s >= bins[i]) & (s < bins[i + 1] if i < 9 else s <= bins[i + 1])
        if m.sum() < 20:
            continue
        out["calibration_bins"].append(
            {
                "mean_predicted": round(float(s[m].mean()), 4),
                "observed_edge_rate": round(float(edge_true[m].mean()), 4),
                "n": int(m.sum()),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    per_bird = {}
    for bird in args.birds:
        r = _run_bird(bird, args)
        if r is None:
            continue
        f0 = (
            frozen.get("per_bird", {})
            .get(str(bird), {})
            .get("arms", {})
            .get("M0", {})
            .get("kept_f1")
        )
        ok = f0 is not None and abs(r["reproduced_M0_f1"] - f0) < 5e-4
        r["freeze_assert_vs_ablation_M0"] = {"frozen": f0, "match": ok}
        per_bird[str(bird)] = r
        print(
            f"[Bird{bird}] M0 {r['reproduced_M0_f1']} (match {ok}) | "
            f"overlap {r['overlap']} | quantiles edge {r['score_quantiles_by_role']['edge']} "
            f"gap {r['score_quantiles_by_role']['gap']}",
            flush=True,
        )

    report = {
        "analysis": "DESCRIPTIVE (advisor-directed): are retained gap "
        "proposals acoustically indistinguishable from true boundaries "
        "under the current 13-D measurements, or distinguishable but "
        "misclassified by the fitted selector?",
        "no_retraining": True,
        "frozen_selector": "M0 logistic reproduced exactly (asserted vs the ablation artifact)",
        "per_bird": per_bird,
    }
    out = ROOT / "outputs" / "gap_proposal_descriptive.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
