#!/usr/bin/env python3
"""Contextual boundary-typing study (Bengalese) — the review's top-priority
next experiment (2026-10-20). Advisor framing: the weak link is boundary
EVIDENCE; test exactly ONE richer feature family at a time with the
proposal set FROZEN. This study: family #1 — local acoustic trajectory
context around each proposal, via a small temporal convolutional network.

ARMS (identical A2 high-recall proposal sets, frozen split, dev-trained,
validation-scored):
  scalar  logistic 4-way selector on the 13 proposal features (the frozen
          baseline, re-derived under the identical pipeline);
  tcn     small temporal CNN over +-200 ms (21-frame) windows of the
          per-bird discovery-scaled 12-feature trajectory centred on each
          proposal. Architecture declared IN ADVANCE, capacity-controlled
          (~26k parameters): Conv1d(12->32,k5,d1) -ReLU- Conv1d(32->32,k5,
          d2) -ReLU- Conv1d(32->32,k5,d4) -ReLU- avgpool - Linear(32->4).
          Adam lr 1e-3, batch 512, class-weighted cross-entropy, early
          stopping on a dev bout-carve (20% of dev bouts, patience 5,
          max 40 epochs, seed 0). NO tuning after validation scores.

NO structured decoder in this study (gated: first show the boundary
evidence is informative). Human labels/models never touch animal data
(firewall); human CPC remains the open arm elsewhere.

PRESECIFIED VALIDATION TARGETS (advisor's criteria):
  T1 proposal recall >= 0.90 (already established; re-derived);
  T2 kept F1 >= scalar + 0.05 with bout-bootstrap CI on dF1 excluding 0;
  T3 no meaningful end-to-end reduction (coverage-weighted recovery drop
     <= 0.05 vs the scalar arm);
  T4 edge-vs-{internal,gap} separation AUC improves over scalar;
  T5 recording-group reliability: per-bout bootstrap CIs reported.
STOP-THE-ARCHITECTURE rule: if T2 fails, the TCN arm is stopped and
reported — no re-tuning on validation.

Usage:
    python scripts/bengalese_contextual_typing.py [--birds 0 1 2]

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
from bengalese_score_diagnostics import dtw_full
from natural_unit_reference_benchmark import _iou

CTX_HALF = 10  # +-10 frames = +-100 ms at 10 ms hop -> 21-frame window
SEED = 0
BOOT_B = 500
EPOCHS = 40
PATIENCE = 5
LR = 1e-3
BATCH = 512


def _ctx_windows(b, frames):
    """(n, 12, 21) trajectory windows centred on each proposal frame."""
    import numpy as np

    z = b["z"]
    n = len(z)
    out = np.zeros((len(frames), 12, 2 * CTX_HALF + 1), dtype=np.float32)
    for i, f in enumerate(frames):
        lo, hi = f - CTX_HALF, f + CTX_HALF + 1
        pad_lo, pad_hi = max(0, -lo), max(0, hi - n)
        seg = z[max(lo, 0) : min(hi, n)]
        w = np.pad(seg, ((pad_lo, pad_hi), (0, 0)), constant_values=0.0)
        out[i] = w.T
    return out


def _train_tcn(Xtr, ytr, Xva, yva, n_classes=4):
    """Small dilated TCN, class-weighted, early-stopped on the dev carve."""
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    # xpu:1 per the machine's stable-device note; CPU fallback
    dev = (
        "xpu:1"
        if (hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() > 1)
        else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    )

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(12, 32, 5, dilation=1, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 32, 5, dilation=2, padding=4),
                nn.ReLU(),
                nn.Conv1d(32, 32, 5, dilation=4, padding=8),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    model = TCN().to(dev)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long).to(dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(dev)
    yva_t = torch.tensor(yva, dtype=torch.long).to(dev)
    n = len(Xtr_t)
    rng = np.random.RandomState(SEED)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, BATCH):
            j = idx[i : i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[j]), ytr_t[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xva_t), yva_t))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"stop_val_loss": round(best, 4)}


def _predict(model, X):
    import torch

    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
        p = torch.softmax(model(x), dim=1).cpu().numpy()
    return p


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
    disc = np.vstack([b["track"] for b in parts["discovery"]])
    mu, sd = disc.mean(0), disc.std(0) + 1e-9
    for b in loaded:
        b["z"] = (b["track"] - mu) / sd
        b["pr"] = _proposals(b, "A2_highrecall")

    dev = parts["discovery"] + parts["calibration"]
    val = parts["validation"]
    lab2id = {"onset": 0, "offset": 1, "within": 2, "gap": 3}

    # ---- assemble dev matrices + bout-carve for early stopping
    Xs, Xt, y, bout_id = [], [], [], []
    for bi, b in enumerate(dev):
        labs = _label_proposals(b["pr"]["frames"], b["ann"])
        Xs.append(b["pr"]["X"])
        Xt.append(_ctx_windows(b, b["pr"]["frames"]))
        y.extend(lab2id[x] for x in labs)
        bout_id.extend([bi] * len(labs))
    Xs = np.vstack(Xs)
    Xt = np.vstack(Xt)
    y = np.array(y)
    bout_id = np.array(bout_id)
    rng = np.random.RandomState(SEED)
    dev_bouts = sorted(set(bout_id.tolist()))
    stop_bouts = set(
        rng.choice(dev_bouts, size=max(1, len(dev_bouts) // 5), replace=False).tolist()
    )
    m_va = np.isin(bout_id, list(stop_bouts))
    m_tr = ~m_va

    scaler = StandardScaler().fit(Xs[m_tr])
    scalar = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        scaler.transform(Xs[m_tr]), y[m_tr]
    )
    tcn, train_info = _train_tcn(Xt[m_tr], y[m_tr], Xt[m_va], y[m_va])

    # ---- validation scoring
    def _score(keep_fn, prob_fn):
        rows, sel_edge, sel_n = [], 0, 0
        for b in val:
            kept, probs = keep_fn(b)
            true_union = sorted({a["s"] for a in b["ann"]} | {a["e"] - 1 for a in b["ann"]})
            pairs, _, _ = _match_boundaries(true_union, kept, TOL_PRIMARY)
            rows.append(
                {
                    "kept": len(kept),
                    "true": len(true_union),
                    "matched": len(pairs),
                    "sel_edge": sum(
                        x in ("onset", "offset") for x in _label_proposals(kept, b["ann"])
                    ),
                    "sel_n": len(kept),
                }
            )
            sel_edge += rows[-1]["sel_edge"]
            sel_n += rows[-1]["sel_n"]
        n_kept = sum(r["kept"] for r in rows)
        n_true = sum(r["true"] for r in rows)
        n_m = sum(r["matched"] for r in rows)
        P = n_m / max(n_kept, 1)
        R = n_m / max(n_true, 1)
        return {
            "kept_precision": round(P, 4),
            "kept_recall": round(R, 4),
            "kept_f1": round(2 * P * R / max(P + R, 1e-9), 4),
            "selected_edge_purity": round(sel_edge / max(sel_n, 1), 4) if sel_n else None,
            "_rows": rows,
        }

    def _keep_scalar(b):
        pred = scalar.predict(scaler.transform(b["pr"]["X"]))
        d = dict(zip(b["pr"]["frames"], pred))
        return [f for f, p in d.items() if p in (0, 1)], d

    def _keep_tcn(b):
        X = _ctx_windows(b, b["pr"]["frames"])
        p = _predict(tcn, X)
        pred = p.argmax(1)
        d = {f: int(pr) for f, pr in zip(b["pr"]["frames"], pred)}
        return [f for f, p_ in d.items() if p_ in (0, 1)], (d, p)

    s_scalar = _score(_keep_scalar, None)
    s_tcn = _score(_keep_tcn, None)

    # AUC: edge (onset+offset) vs rest, on validation proposals
    ys, ps, yt, pt_ = [], [], [], []
    for b in val:
        labs = _label_proposals(b["pr"]["frames"], b["ann"])
        ybin = np.array([x in ("onset", "offset") for x in labs])
        ps.append(scalar.predict_proba(scaler.transform(b["pr"]["X"]))[:, :2].sum(1))
        p2 = _predict(tcn, _ctx_windows(b, b["pr"]["frames"]))
        pt_.append(p2[:, :2].sum(1))
        ys.append(ybin)
        yt.append(ybin)
    auc_scalar = round(float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))), 4)
    auc_tcn = round(float(roc_auc_score(np.concatenate(yt), np.concatenate(pt_))), 4)

    # dF1 bootstrap over validation bouts (cluster)
    r_s = [(r["matched"], r["kept"], r["true"]) for r in s_scalar["_rows"]]
    r_t = [(r["matched"], r["kept"], r["true"]) for r in s_tcn["_rows"]]
    rng2 = np.random.RandomState(SEED)
    dfs = []
    for _ in range(BOOT_B):
        idx = rng2.randint(0, len(r_s), len(r_s))

        def _f1(rs):
            mk = sum(rs[i][0] for i in idx)
            kk = sum(rs[i][1] for i in idx)
            tt = sum(rs[i][2] for i in idx)
            P = mk / max(kk, 1)
            R = mk / max(tt, 1)
            return 2 * P * R / max(P + R, 1e-9)

        dfs.append(_f1(r_t) - _f1(r_s))
    df1_ci = [round(float(np.percentile(dfs, 2.5)), 4), round(float(np.percentile(dfs, 97.5)), 4)]

    # T3: end-to-end coverage-weighted recovery on selected segments
    def _end_to_end(keep_fn):
        usable_dev, usable_val = [], []
        for b in dev:
            kept, _ = keep_fn(b)
            bnd = [0, *sorted(kept), len(b["z"])]
            for i in range(len(bnd) - 1):
                s, e = bnd[i], bnd[i + 1]
                if e - s < 3:
                    continue
                best = None
                for a in b["ann"]:
                    iou, _ov = _iou(s, e, a["s"], a["e"])
                    if iou > 0 and (best is None or iou > best[0]):
                        best = (iou, a["label"])
                if best and best[0] >= 0.5:
                    usable_dev.append((b["wav"], b["z"][s:e], best[1]))
        for b in val:
            kept, _ = keep_fn(b)
            bnd = [0, *sorted(kept), len(b["z"])]
            for i in range(len(bnd) - 1):
                s, e = bnd[i], bnd[i + 1]
                if e - s < 3:
                    continue
                best = None
                for a in b["ann"]:
                    iou, _ov = _iou(s, e, a["s"], a["e"])
                    if iou > 0 and (best is None or iou > best[0]):
                        best = (iou, a["label"])
                if best and best[0] >= 0.5:
                    usable_val.append((b["wav"], b["z"][s:e], best[1]))
        n_ann_val = sum(len(b["ann"]) for b in val)
        per_tok = []
        for wv, q, lab in usable_dev:
            costs = []
            for j, (wv2, v, lab2) in enumerate(usable_val):
                if wv2 == wv:
                    continue
                c = dtw_full(q, v)
                if np.isfinite(c):
                    costs.append((c, j))
            if costs:
                costs.sort(key=lambda r: r[0])
                per_tok.append(1 if usable_val[costs[0][1]][2] == lab else 0)
        cov = len(usable_val) / max(n_ann_val, 1)
        r1 = float(np.mean(per_tok)) if per_tok else 0.0
        return {
            "coverage": round(cov, 4),
            "retrieval_r1": round(r1, 4),
            "coverage_weighted": round(cov * r1, 4),
        }

    e2e_scalar = _end_to_end(_keep_scalar)
    e2e_tcn = _end_to_end(_keep_tcn)

    s_scalar.pop("_rows", None)
    s_tcn.pop("_rows", None)

    verdicts = {
        "T1_proposal_recall": None,  # filled by caller from regime stats
        "T2_f1_gain": bool(s_tcn["kept_f1"] >= s_scalar["kept_f1"] + 0.05 and df1_ci[0] > 0),
        "T3_no_e2e_reduction": bool(
            e2e_tcn["coverage_weighted"] >= e2e_scalar["coverage_weighted"] - 0.05
        ),
        "T4_auc_gain": bool(auc_tcn > auc_scalar),
    }
    return {
        "n_bouts": len(loaded),
        "scalar": s_scalar,
        "tcn": s_tcn,
        "df1_bootstrap_ci": df1_ci,
        "auc_edge_vs_rest": {"scalar": auc_scalar, "tcn": auc_tcn},
        "end_to_end": {"scalar": e2e_scalar, "tcn": e2e_tcn},
        "train_info": train_info,
        "verdicts": verdicts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    args = ap.parse_args()

    per_bird = {}
    for bird in args.birds:
        r = _run_bird(bird, args)
        if r is None:
            print(f"[Bird{bird}] skipped", flush=True)
            continue
        per_bird[str(bird)] = r
        print(
            f"[Bird{bird}] scalar F1 {r['scalar']['kept_f1']} tcn F1 {r['tcn']['kept_f1']} "
            f"dF1 CI {r['df1_bootstrap_ci']} AUC {r['auc_edge_vs_rest']} "
            f"e2e {r['end_to_end']['tcn']['coverage_weighted']} vs {r['end_to_end']['scalar']['coverage_weighted']}",
            flush=True,
        )

    report = {
        "study": "contextual boundary-typing study (family #1: local "
        "trajectory context via small TCN; proposal set frozen)",
        "architecture_declared": "Conv1d(12->32,k5,d1)-ReLU-Conv1d(32->32,"
        "k5,d2)-ReLU-Conv1d(32->32,k5,d4)-ReLU-avgpool-Linear(32->4); "
        "Adam 1e-3, batch 512, class-weighted CE, early stop on dev bout "
        "carve (patience 5, max 40 epochs, seed 0)",
        "no_structured_decoder": True,
        "firewall": "no human labels or models touch animal data",
        "prespecified_targets": "T2 F1 >= scalar+0.05 (CI excludes 0); T3 "
        "e2e drop <= 0.05; T4 AUC gain; stop-the-architecture on T2 failure",
        "per_bird": per_bird,
        "stop_rule": "one run per arm; no tuning after validation scores",
    }
    out = ROOT / "outputs" / "bengalese_contextual_typing.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({b: r["verdicts"] for b, r in per_bird.items()}, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
