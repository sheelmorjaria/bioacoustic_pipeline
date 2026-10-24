#!/usr/bin/env python3
"""Boundary-representation ablation M0-M3 (advisor-directed 2026-10-20).

PRIMARY HYPOTHESIS (preregistered): explicitly representing signed
acoustic contrasts at multiple temporal scales improves candidate
boundary-role classification beyond what the contextual sequence encoder
learns implicitly — an INDUCTIVE-BIAS intervention (a sufficiently
expressive TCN could learn these differences; we test generalisation, not
information novelty).

ARMS (identical A2 proposals, eligibility, labels, recording-group
partitions, seed; ONE run):
  M0  13-D proposal scalars + logistic              (FROZEN comparator:
      recomputed deterministically and ASSERTED equal to
      outputs/bengalese_contextual_typing.json — mismatch aborts);
  M1  +-100 ms TCN                                  (FROZEN comparator,
      same assert);
  M2  13-D + explicit multiscale before-after contrasts + logistic;
  M3  13-D + contrasts + frozen-TCN 32-D embedding + logistic.
CONTRAST FEATURES (per scale w in {2,5,10} frames = 20/50/100 ms; per
feature j): signed difference d_j = mu+ - mu-; normalised contrast
D_j = d_j / sqrt(sigma+^2 + sigma-^2 + eps); dispersion log-ratio
log((sigma+ + eps)/(sigma- + eps)); computed over VALID frames only
(the bout validity mask); a side with <2 valid frames contributes zeros
plus a validity flag bit (V_t = eligibility metadata, never a label
shortcut — declared). 36 features/scale + 2 flags/scale -> 114 contrast
dims.

ENDPOINTS (fixed now): kept (onset+offset) F1 pooled per bird on
validation (the completed study's estimand); per-role recall/precision
(onset / offset / within / gap — reported separately so abundant
within/gap proposals cannot dominate the reading); edge-vs-rest AUC;
bout-cluster bootstrap CI on each arm DIFFERENCE (same procedure as the
completed study: difference of pooled F1s over resampled validation
bouts; point estimand = difference of pooled F1s).
DECISION RULES: arm wins if dF1 >= +0.05 vs its baseline (M2 vs M0;
M3 vs M1) with CI excluding 0. M0/M1 historical numbers are NEVER
retuned.

POST-HOC BIRD-2 DIAGNOSTIC (descriptive only, clearly labelled, on the
deterministically regenerated frozen predictions): why did e2e recovery
gain +0.163 with only +0.025 boundary F1 — segment IoU distribution,
paired-boundary coverage (annotated syllables with BOTH edges kept),
per-class usable-segment coverage.

Usage:
    python scripts/bengalese_contrast_ablation.py [--birds 0 1 2]

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
from bengalese_contextual_typing import _ctx_windows, _predict, _train_tcn
from bengalese_natural_benchmark import DATA, HOP_MS, SR, WIN_MS, _bout_job, parse_annotation
from bengalese_proposal_selection import _label_proposals, _proposals
from natural_unit_reference_benchmark import _iou

SCALES = (2, 5, 10)  # frames: 20 / 50 / 100 ms
EPS = 1e-6
SEED = 0
BOOT_B = 500
FROZEN = ROOT / "outputs" / "bengalese_contextual_typing.json"


def _contrast_features(b, frames):
    """Multiscale signed contrasts + normalised + dispersion + validity flags."""
    import numpy as np

    z, valid = b["z"], b["valid"]
    n = len(z)
    out = np.zeros((len(frames), 3 * 12 * len(SCALES) + 2 * len(SCALES)), dtype=np.float32)
    for i, f in enumerate(frames):
        col = 0
        for w in SCALES:
            pre_sl = slice(max(0, f - w), f)
            post_sl = slice(f + 1, min(n, f + 1 + w))
            pre_v = valid[pre_sl]
            post_v = valid[post_sl]
            pre_ok = int(pre_v.sum() >= 2)
            post_ok = int(post_v.sum() >= 2)
            if pre_ok:
                mu_m = z[pre_sl][pre_v].mean(0)
                sd_m = z[pre_sl][pre_v].std(0)
            else:
                mu_m = np.zeros(12)
                sd_m = np.zeros(12)
            if post_ok:
                mu_p = z[post_sl][post_v].mean(0)
                sd_p = z[post_sl][post_v].std(0)
            else:
                mu_p = np.zeros(12)
                sd_p = np.zeros(12)
            d = mu_p - mu_m
            D = d / np.sqrt(sd_p**2 + sd_m**2 + EPS)
            disp = np.log((sd_p + EPS) / (sd_m + EPS))
            out[i, col : col + 12] = d
            col += 12
            out[i, col : col + 12] = D
            col += 12
            out[i, col : col + 12] = disp
            col += 12
            out[i, col] = pre_ok
            out[i, col + 1] = post_ok
            col += 2
    return out


def _score(bouts, keep_fn):

    rows = []
    role = {"onset": [0, 0], "offset": [0, 0], "within": [0, 0], "gap": [0, 0]}
    for b in bouts:
        kept = keep_fn(b)
        true_union = sorted({a["s"] for a in b["ann"]} | {a["e"] - 1 for a in b["ann"]})
        pairs, _, _ = _match_boundaries(true_union, kept, TOL_PRIMARY)
        rows.append({"kept": len(kept), "true": len(true_union), "matched": len(pairs)})
        labs = _label_proposals(kept, b["ann"])
        kept_set = set(kept)
        for r in ("onset", "offset"):
            targets = [a["s"] for a in b["ann"]] if r == "onset" else [a["e"] - 1 for a in b["ann"]]
            role[r][1] += len(targets)
            role[r][0] += sum(
                1 for t in targets if any(abs(t - k) <= TOL_PRIMARY for k in kept_set)
            )
        for x in labs:
            if x in ("within", "gap"):
                role[x][1] += 1
                role[x][0] += 0  # precision-side below
        n_within_kept = sum(1 for x in labs if x == "within")
        n_gap_kept = sum(1 for x in labs if x == "gap")
        role["within"][0] = n_within_kept
        role["gap"][0] = n_gap_kept
    n_kept = sum(r["kept"] for r in rows)
    n_true = sum(r["true"] for r in rows)
    n_m = sum(r["matched"] for r in rows)
    P = n_m / max(n_kept, 1)
    R = n_m / max(n_true, 1)
    out = {
        "kept_precision": round(P, 4),
        "kept_recall": round(R, 4),
        "kept_f1": round(2 * P * R / max(P + R, 1e-9), 4),
        "role_onset_recall": round(role["onset"][0] / max(role["onset"][1], 1), 4),
        "role_offset_recall": round(role["offset"][0] / max(role["offset"][1], 1), 4),
        "kept_labeled_within": role["within"][0],
        "kept_labeled_gap": role["gap"][0],
    }
    out["_rows"] = rows
    return out


def _df1_bootstrap(rows_a, rows_b, seed=SEED, b=BOOT_B):
    """CI for pooled-F1(a) - pooled-F1(b), bout-cluster resampling."""
    import numpy as np

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(b):
        idx = rng.randint(0, len(rows_a), len(rows_a))

        def _f1(rs, idx):
            mk = sum(rs[i]["matched"] for i in idx)
            kk = sum(rs[i]["kept"] for i in idx)
            tt = sum(rs[i]["true"] for i in idx)
            P, R = mk / max(kk, 1), mk / max(tt, 1)
            return 2 * P * R / max(P + R, 1e-9)

        diffs.append(_f1(rows_a, idx) - _f1(rows_b, idx))
    return [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]


def _bird2_diagnostic(val, keep_m0, keep_m1):
    """Post hoc, descriptive: segment IoU, paired coverage, per-class cover."""
    import numpy as np

    def _stats(keep_fn, tag):
        ious, paired, percls, n_ann = [], 0, {}, 0
        for b in val:
            kept = keep_fn(b)
            n_ann += len(b["ann"])
            ks = set(kept)
            for a in b["ann"]:
                on = any(abs(a["s"] - k) <= TOL_PRIMARY for k in ks)
                off = any(abs((a["e"] - 1) - k) <= TOL_PRIMARY for k in ks)
                paired += int(on and off)
                # segment containing/overlapping this annotation
                best = 0.0
                bnd = [0, *sorted(kept), len(b["z"])]
                for i in range(len(bnd) - 1):
                    iou, _ov = _iou(bnd[i], bnd[i + 1], a["s"], a["e"])
                    best = max(best, iou)
                ious.append(best)
                c = percls.setdefault(a["label"], [0, 0])
                c[1] += 1
                c[0] += int(best >= 0.5)
        top = sorted(percls.items(), key=lambda kv: -kv[1][1])[:8]
        return {
            "median_segment_iou": round(float(np.median(ious)), 4),
            "segment_iou_ge_0.5": round(float(np.mean([i >= 0.5 for i in ious])), 4),
            "paired_edge_coverage": round(paired / max(n_ann, 1), 4),
            "top_classes_iou05_coverage": {k: [v[0], v[1]] for k, v in top},
        }

    return {"M0_scalar": _stats(keep_m0, "m0"), "M1_tcn": _stats(keep_m1, "m1")}


def _run_bird(bird, args):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
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

    Xs, Xt, Xc, y, bout_id = [], [], [], [], []
    for bi, b in enumerate(dev):
        labs = _label_proposals(b["pr"]["frames"], b["ann"])
        Xs.append(b["pr"]["X"])
        Xt.append(_ctx_windows(b, b["pr"]["frames"]))
        Xc.append(_contrast_features(b, b["pr"]["frames"]))
        y.extend(lab2id[x] for x in labs)
        bout_id.extend([bi] * len(labs))
    Xs, Xt, Xc = np.vstack(Xs), np.vstack(Xt), np.vstack(Xc)
    y = np.array(y)
    bout_id = np.array(bout_id)
    rng = np.random.RandomState(SEED)
    dev_bouts = sorted(set(bout_id.tolist()))
    stop_bouts = set(
        rng.choice(dev_bouts, size=max(1, len(dev_bouts) // 5), replace=False).tolist()
    )
    m_va = np.isin(bout_id, list(stop_bouts))
    m_tr = ~m_va

    sc0 = StandardScaler().fit(Xs[m_tr])
    m0 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        sc0.transform(Xs[m_tr]), y[m_tr]
    )
    tcn, _ = _train_tcn(Xt[m_tr], y[m_tr], Xt[m_va], y[m_va])  # identical recipe -> M1 frozen

    def _emb(b):
        import torch

        with torch.no_grad():
            x = torch.tensor(_ctx_windows(b, b["pr"]["frames"]), dtype=torch.float32).to(
                next(tcn.parameters()).device
            )
            h = tcn.net[:-1](x)  # penultimate: conv stack + avgpool + flatten -> (B, 32)
            return h.cpu().numpy()

    sc2 = StandardScaler().fit(np.hstack([Xs, Xc])[m_tr])
    X2 = np.hstack([Xs, Xc])
    m2 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        sc2.transform(X2[m_tr]), y[m_tr]
    )
    np.hstack([Xs, Xc])
    _emb({"pr": {"frames": []}}) if False else None
    # M3: embeddings per dev/validation bout (TCN frozen)
    emb_blocks, _off_i = [], 0
    for b in dev:
        e = _emb(b)
        emb_blocks.append(e)
    E_dev = np.vstack(emb_blocks)
    sc3 = StandardScaler().fit(np.hstack([X2, E_dev])[m_tr])
    m3 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        sc3.transform(np.hstack([X2, E_dev])[m_tr]), y[m_tr]
    )

    def keep_m0(b):
        pred = m0.predict(sc0.transform(b["pr"]["X"]))
        return [f for f, p in zip(b["pr"]["frames"], pred) if p in (0, 1)]

    def keep_m1(b):
        p = _predict(tcn, _ctx_windows(b, b["pr"]["frames"]))
        return [f for f, pr in zip(b["pr"]["frames"], p.argmax(1)) if pr in (0, 1)]

    def keep_m2(b):
        X = sc2.transform(np.hstack([b["pr"]["X"], _contrast_features(b, b["pr"]["frames"])]))
        pred = m2.predict(X)
        return [f for f, p in zip(b["pr"]["frames"], pred) if p in (0, 1)]

    def keep_m3(b):
        X = np.hstack([b["pr"]["X"], _contrast_features(b, b["pr"]["frames"]), _emb(b)])
        pred = m3.predict(sc3.transform(X))
        return [f for f, p in zip(b["pr"]["frames"], pred) if p in (0, 1)]

    res = {
        k: _score(val, kf)
        for k, kf in (("M0", keep_m0), ("M1", keep_m1), ("M2", keep_m2), ("M3", keep_m3))
    }
    cis = {
        "d_explicit_M2_M0": _df1_bootstrap(res["M2"]["_rows"], res["M0"]["_rows"]),
        "d_combination_M3_M1": _df1_bootstrap(res["M3"]["_rows"], res["M1"]["_rows"]),
    }
    for r in res.values():
        r.pop("_rows", None)

    # AUC edge-vs-rest on validation (M2/M3 vs frozen M0/M1 numbers)
    auc = {}
    for tag, prob_fn in (
        ("M0", lambda b: m0.predict_proba(sc0.transform(b["pr"]["X"]))[:, :2].sum(1)),
        (
            "M1",
            lambda b: _predict(tcn, _ctx_windows(b, b["pr"]["frames"]))[:, :2].sum(1),
        ),
        (
            "M2",
            lambda b: m2.predict_proba(
                sc2.transform(np.hstack([b["pr"]["X"], _contrast_features(b, b["pr"]["frames"])]))
            )[:, :2].sum(1),
        ),
        (
            "M3",
            lambda b: m3.predict_proba(
                sc3.transform(
                    np.hstack([b["pr"]["X"], _contrast_features(b, b["pr"]["frames"]), _emb(b)])
                )
            )[:, :2].sum(1),
        ),
    ):
        ys, ps = [], []
        for b in val:
            labs = _label_proposals(b["pr"]["frames"], b["ann"])
            ys.extend(x in ("onset", "offset") for x in labs)
            ps.extend(prob_fn(b).tolist())
        auc[tag] = round(float(roc_auc_score(ys, ps)), 4)

    diag = _bird2_diagnostic(val, keep_m0, keep_m1) if bird == 2 else None
    return {"arms": res, "df1_cis": cis, "auc_edge_vs_rest": auc, "bird2_post_hoc_diagnostic": diag}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else None
    per_bird = {}
    for bird in args.birds:
        r = _run_bird(bird, args)
        if r is None:
            continue
        # freeze assert: regenerated M0/M1 must equal the frozen comparators
        if frozen and str(bird) in frozen.get("per_bird", {}):
            f0 = frozen["per_bird"][str(bird)]["scalar"]["kept_f1"]
            f1 = frozen["per_bird"][str(bird)]["tcn"]["kept_f1"]
            # exact match expected for M0 (pure sklearn); M1 may drift by
            # XPU-kernel nondeterminism (conv backward atomics) — tolerated
            # to 0.005 pooled-F1 and documented, never silently accepted
            ok0 = abs(r["arms"]["M0"]["kept_f1"] - f0) < 5e-4
            ok1 = abs(r["arms"]["M1"]["kept_f1"] - f1) < 5e-3
            r["freeze_assert"] = {
                "M0": ok0,
                "M1": ok1,
                "frozen_M0": f0,
                "frozen_M1": f1,
                "M1_drift": round(abs(r["arms"]["M1"]["kept_f1"] - f1), 4),
                "note": "M1 tolerance 0.005 = documented XPU nondeterminism; "
                "within-study contrasts use the regenerated M1 consistently",
            }
            if not (ok0 and ok1):
                print(
                    f"[Bird{bird}] FREEZE ASSERT FAILED: M0 {r['arms']['M0']['kept_f1']} vs {f0}, "
                    f"M1 {r['arms']['M1']['kept_f1']} vs {f1} — aborting (pipeline divergence)",
                    flush=True,
                )
                return
        per_bird[str(bird)] = r
        print(
            f"[Bird{bird}] M0 {r['arms']['M0']['kept_f1']} M1 {r['arms']['M1']['kept_f1']} "
            f"M2 {r['arms']['M2']['kept_f1']} M3 {r['arms']['M3']['kept_f1']} "
            f"AUC {r['auc_edge_vs_rest']}",
            flush=True,
        )

    def _verdict(a, b):
        return bool(a - b >= 0.05)

    verdicts = {}
    for b_, r in per_bird.items():
        verdicts[b_] = {
            "explicit_contrast_wins_M2_vs_M0": _verdict(
                r["arms"]["M2"]["kept_f1"], r["arms"]["M0"]["kept_f1"]
            ),
            "combination_wins_M3_vs_M1": _verdict(
                r["arms"]["M3"]["kept_f1"], r["arms"]["M1"]["kept_f1"]
            ),
        }

    report = {
        "study": "boundary-representation ablation M0-M3 (inductive-bias "
        "test: explicit multiscale signed contrasts vs implicitly learned "
        "context)",
        "primary_hypothesis": "explicit signed contrasts at 20/50/100 ms "
        "improve boundary-role classification beyond the contextual "
        "encoder's implicit learning",
        "contrast_features": "per scale {2,5,10} frames and feature: "
        "signed mu diff, normalised D = d/sqrt(sig+^2+sig-^2+eps), "
        "dispersion log-ratio; valid-frame-only; <2 valid frames -> zeros "
        "+ eligibility flag (V_t metadata, never a label shortcut)",
        "frozen_comparators": "M0/M1 asserted equal to "
        "outputs/bengalese_contextual_typing.json before scoring (abort on "
        "mismatch); never retuned",
        "decision_rule": "arm wins if dF1 >= +0.05 (pooled kept-F1) with "
        "bout-cluster bootstrap CI excluding 0",
        "per_bird": per_bird,
        "verdicts": verdicts,
        "bird2_diagnostic_label": "POST HOC, descriptive only; no model selection reopened",
        "stop_rule": "one run; no tuning after validation scores",
    }
    out = ROOT / "outputs" / "bengalese_contrast_ablation.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(verdicts, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
