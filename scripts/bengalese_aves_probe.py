#!/usr/bin/env python3
"""M4 boundary-typing arm — frozen AVES v2 embeddings + linear probe
(advisor ladder rung 4: 'frozen pretrained acoustic embeddings | linear
probe | external representation comparator').

PREREGISTRATION (frozen before eval scores): identical A2 high-recall
proposals, strata, labels and dev bout-carve as the M0-M3 ablation; M0
reproduced deterministically and asserted (abort on mismatch). M4 input:
for each proposal, the esp-aves2-eat-bio patch embeddings of a FIXED
+-1 s audio window centred on the proposal (16 kHz linear resample,
checkpoint's mel normalisation, zero-padded short flanks — extraction
contract declared per the advisor's AVES padding caveat), summarised as
[mean-pre || mean-post || diff] (3x768 dims) split at the proposal.
Probe: logistic (balanced) on standardised features. NOTHING learned
here transfers as human/animal boundary weights; the firewall registry
gates AVES to embedding/probe use. Endpoint: kept (onset+offset) pooled
F1 on validation vs frozen M0 (+0.05 rule, bout-cluster bootstrap CI on
the pooled-F1 difference); edge-vs-rest AUC; per-role recall. One run.

Usage:
    python scripts/bengalese_aves_probe.py [--birds 0 1 2]

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

from aves2_loader import embed_audio
from bengalese_boundary_decomposition import TOL_PRIMARY, _match_boundaries
from bengalese_natural_benchmark import (
    DATA,
    HOP_MS,
    SR,
    WIN_MS,
    _bout_job,
    _load_wav,
    parse_annotation,
)
from bengalese_proposal_selection import _label_proposals, _proposals

WIN_S = 1.0  # +-1 s extraction window
SEED = 0
BOOT_B = 500
FROZEN = ROOT / "outputs" / "bengalese_contrast_ablation.json"


def _proposal_aves(b, audio):
    """(n, 3*768) AVES context per proposal: mean-pre || mean-post || diff."""
    import numpy as np

    int(SR * WIN_S)
    out = np.zeros((len(b["pr"]["frames"]), 3 * 768), dtype=np.float32)
    # 32 kHz -> 16 kHz linear resample once per bout
    x = np.linspace(0, 1, len(audio))
    xi = np.linspace(0, 1, max(len(audio) // 2, 1))
    a16 = np.interp(xi, x, audio).astype(np.float32)
    sr16 = 16000
    for i, f in enumerate(b["pr"]["frames"]):
        c16 = int(round(f * (SR / 1000.0) / 2))  # proposal centre in 16k samples
        lo, hi = max(0, c16 - sr16), min(len(a16), c16 + sr16)
        seg = np.zeros(2 * sr16, dtype=np.float32)
        seg[max(0, sr16 - (c16 - lo)) : sr16 + max(0, hi - c16)] = a16[lo:hi]
        emb = embed_audio(seg, target_frames=200)  # ~2 s -> ~100 patches
        n = len(emb)
        pre, post = emb[: n // 2].mean(0), emb[n // 2 :].mean(0)
        out[i] = np.concatenate([pre, post, post - pre])
    return out


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

    # AVES contexts per bout (sequential — single XPU model instance)
    for b in loaded:
        audio = _load_wav(Path(b["flac"]))
        b["aves"] = _proposal_aves(b, audio)
        print(f"  [Bird{bird}] {b['wav']} aves {b['aves'].shape}", flush=True)

    Xs, Xa, y, bout_id = [], [], [], []
    for bi, b in enumerate(dev):
        labs = _label_proposals(b["pr"]["frames"], b["ann"])
        Xs.append(b["pr"]["X"])
        Xa.append(b["aves"])
        y.extend(lab2id[x] for x in labs)
        bout_id.extend([bi] * len(labs))
    Xs, Xa = np.vstack(Xs), np.vstack(Xa)
    y = np.array(y)
    bout_id = np.array(bout_id)
    rng = np.random.RandomState(SEED)
    dev_bouts = sorted(set(bout_id.tolist()))
    stop_bouts = set(
        rng.choice(dev_bouts, size=max(1, len(dev_bouts) // 5), replace=False).tolist()
    )
    m_tr = ~np.isin(bout_id, list(stop_bouts))

    sc0 = StandardScaler().fit(Xs[m_tr])
    m0 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        sc0.transform(Xs[m_tr]), y[m_tr]
    )
    sca = StandardScaler().fit(Xa[m_tr])
    m4 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        sca.transform(Xa[m_tr]), y[m_tr]
    )

    def _score(keep_fn, prob_fn):
        rows, on_r, off_r, on_t, off_t = [], 0, 0, 0, 0
        ys, ps = [], []
        for b in val:
            kept = keep_fn(b)
            true_union = sorted({a["s"] for a in b["ann"]} | {a["e"] - 1 for a in b["ann"]})
            pairs, _, _ = _match_boundaries(true_union, kept, TOL_PRIMARY)
            rows.append({"kept": len(kept), "true": len(true_union), "matched": len(pairs)})
            ks = set(kept)
            on = [a["s"] for a in b["ann"]]
            off = [a["e"] - 1 for a in b["ann"]]
            on_t += len(on)
            off_t += len(off)
            on_r += sum(1 for t in on if any(abs(t - k) <= TOL_PRIMARY for k in ks))
            off_r += sum(1 for t in off if any(abs(t - k) <= TOL_PRIMARY for k in ks))
            labs = _label_proposals(b["pr"]["frames"], b["ann"])
            ys.extend(x in ("onset", "offset") for x in labs)
            ps.extend(prob_fn(b).tolist())
        n_kept = sum(r["kept"] for r in rows)
        n_true = sum(r["true"] for r in rows)
        n_m = sum(r["matched"] for r in rows)
        P, R = n_m / max(n_kept, 1), n_m / max(n_true, 1)
        return {
            "kept_f1": round(2 * P * R / max(P + R, 1e-9), 4),
            "kept_precision": round(P, 4),
            "kept_recall": round(R, 4),
            "role_onset_recall": round(on_r / max(on_t, 1), 4),
            "role_offset_recall": round(off_r / max(off_t, 1), 4),
            "auc_edge_vs_rest": round(float(roc_auc_score(ys, ps)), 4),
            "_rows": rows,
        }

    s_m0 = _score(
        lambda b: [
            f
            for f, p in zip(b["pr"]["frames"], m0.predict(sc0.transform(b["pr"]["X"])))
            if p in (0, 1)
        ],
        lambda b: m0.predict_proba(sc0.transform(b["pr"]["X"]))[:, :2].sum(1),
    )
    s_m4 = _score(
        lambda b: [
            f
            for f, p in zip(b["pr"]["frames"], m4.predict(sca.transform(b["aves"])))
            if p in (0, 1)
        ],
        lambda b: m4.predict_proba(sca.transform(b["aves"]))[:, :2].sum(1),
    )

    rng2 = np.random.RandomState(SEED)
    diffs = []
    for _ in range(BOOT_B):
        idx = rng2.randint(0, len(s_m0["_rows"]), len(s_m0["_rows"]))

        def _f1(rs):
            mk = sum(rs[i]["matched"] for i in idx)
            kk = sum(rs[i]["kept"] for i in idx)
            tt = sum(rs[i]["true"] for i in idx)
            P, R = mk / max(kk, 1), mk / max(tt, 1)
            return 2 * P * R / max(P + R, 1e-9)

        diffs.append(_f1(s_m4["_rows"]) - _f1(s_m0["_rows"]))
    ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    s_m0.pop("_rows", None)
    s_m4.pop("_rows", None)
    return {"M0": s_m0, "M4_aves": s_m4, "df1_ci": ci}


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
        ok = f0 is not None and abs(r["M0"]["kept_f1"] - f0) < 5e-4
        r["freeze_assert_M0"] = {"frozen": f0, "match": ok}
        if not ok:
            print(f"[Bird{bird}] FREEZE ASSERT FAILED — aborting", flush=True)
            return
        per_bird[str(bird)] = r
        print(
            f"[Bird{bird}] M0 {r['M0']['kept_f1']} M4 {r['M4_aves']['kept_f1']} "
            f"CI {r['df1_ci']} AUC {r['M0']['auc_edge_vs_rest']}/{r['M4_aves']['auc_edge_vs_rest']}",
            flush=True,
        )

    report = {
        "study": "M4: frozen AVES v2 (esp-aves2-eat-bio) embeddings + linear "
        "probe — advisor ladder rung 4 (external representation comparator)",
        "extraction_contract": "+-1 s zero-padded window per proposal, 16 "
        "kHz linear resample, checkpoint mel normalisation, [pre||post||"
        "diff] pooled patch embeddings (3x768)",
        "firewall": "AVES registry entry allows embedding/probe use only; "
        "no boundary weights transfer; no semantic claims",
        "decision_rule": "M4 wins if pooled kept-F1 >= M0 + 0.05 with "
        "bout-cluster bootstrap CI excluding 0",
        "per_bird": per_bird,
        "verdicts": {
            b: bool(r["M4_aves"]["kept_f1"] >= r["M0"]["kept_f1"] + 0.05 and r["df1_ci"][0] > 0)
            for b, r in per_bird.items()
        },
        "stop_rule": "one run; loader validated separately (centered "
        "within/between cosine 0.736/-0.254 on real syllables)",
    }
    out = ROOT / "outputs" / "bengalese_aves_probe.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdicts"], indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
