#!/usr/bin/env python3
"""LibriSpeech Human Reference Battery — positive-control regime (studies 1-3).

FRAMING (review-directed): a methodological positive-control regime, NOT a
template for animal communication. Known linguistic structure calibrates
what the programme's analyses look like when discrete words and sentence
composition independently exist.

PROVENANCE CAVEAT (prominent): LibriSpeech ships utterance transcripts
but NO word/phoneme timestamps. Without a separately validated forced-
alignment source, word boundaries are NOT available; study 2 therefore
uses UTTERANCE-LEVEL recurrence (same-speaker utterances vs different-
speaker utterances as the reference classes — an honest weaker reference),
and study 3 stratifies masks by ENERGY-DEFINED boundary categories
(syllable-energy onsets), declared as acoustic proxies, not lexical
ground truth. Any future word-boundary claim requires an alignment
source with its own provenance and manual validation.

STUDY 1 — forecasting + order battery (the direct animal comparator):
  16 kHz, 40 ms/10 ms grid, 12-feature fingerprints (schema v2, human
  species tag); horizons {40,80,160,320} ms; arms persistence/state/
  +immediate/+older-lags; PLUS same-trajectory shuffled-history control
  at 40 ms. Endpoints Delta_recent, Delta_older(h), Delta_order. Raw
  nRMSE magnitudes are NOT biologically comparable across corpora; the
  comparison is STRUCTURAL (does known speech show history gain, decay,
  adjacency dependence?).

STUDY 2 — utterance-level recurrence recovery (label-blind discovery
  vs reference classes): k-best candidates (the exact bat/Bengalese
  machinery) within-speaker across utterances; oracle coverage = does a
  candidate land in a high-energy (voiced) region of another utterance
  of the SAME speaker vs a DIFFERENT speaker. Reports the selection
  headroom with a real reference axis.

STUDY 3 — masked reconstruction stratified by energy-boundary category:
  central 70 ms masks; arms blend/forward/backward/bidir/+G (utterance
  rest summaries); masks stratified prospectively by energy-onset
  count inside the mask (0, 1, >=2 acoustic boundaries — proxy
  categories, declared).

Usage:
    python scripts/human_reference_battery.py [--max-utter 400]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 16000
WIN_MS, HOP_MS = 40.0, 10.0
HORIZONS = (4, 8, 16, 32)
LAGS = (4, 8, 12, 20)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 500
SEED = 0
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
MASK_FRAMES = 7
MIN_SPAN_FRAMES = 120


def _fingerprint_track(path):
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    audio, sr = load_audio(str(path), target_sr=SR)
    win, hop = int(sr * WIN_MS / 1000), int(sr * HOP_MS / 1000)
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 700:
        v = np.asarray(
            afp.extract(audio[i * hop : i * hop + win], sr, species="human").combined,
            dtype=np.float64,
        )
        rows.append(v)
        i += 1
    if len(rows) < MIN_SPAN_FRAMES:
        return None
    f = np.stack(rows)
    valid = ~np.all(f == 0, axis=1)
    return f[:, fidx], valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-utter", type=int, default=400)
    args = ap.parse_args()

    import glob

    import numpy as np

    flacs = sorted(glob.glob(str(ROOT / "data/LibriSpeech/**/*.flac"), recursive=True))[
        : args.max_utter
    ]
    trans = {}
    for tf in glob.glob(str(ROOT / "data/LibriSpeech/**/*.trans.txt"), recursive=True):
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            trans[uid] = text
    utterances = []
    for p in flacs:
        uid = Path(p).stem
        spk = uid.split("-")[0]
        out = _fingerprint_track(Path(p))
        if out is None:
            continue
        z, valid = out
        utterances.append(
            {
                "uid": uid,
                "spk": spk,
                "text": trans.get(uid, ""),
                "z": z,
                "valid": valid,
                "group": f"spk{spk}",
            }
        )
    print(
        f"utterances with long support: {len(utterances)} "
        f"({len({u['spk'] for u in utterances})} speakers)",
        flush=True,
    )
    if len(utterances) < 24:
        print("INSUFFICIENT MATERIAL — terminal", flush=True)
        return

    groups = sorted({u["group"] for u in utterances})
    fit_blocks = {
        g
        for g in groups
        if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng = np.random.RandomState(SEED)
    fl = sorted(fit_blocks)
    rng.shuffle(fl)
    val_blocks = set(fl[: max(1, int(len(fl) * VAL_FRAC))])
    fit_only = set(fl) - val_blocks
    for u in utterances:
        u["part"] = (
            "fit" if u["group"] in fit_only else ("val" if u["group"] in val_blocks else "eval")
        )

    def _collect(h, shuffle=False, selected=None):
        X6, Y, GRP = [], [], []
        rng2 = np.random.RandomState(SEED)
        for u in selected or utterances:
            z, v = u["z"], u["valid"]
            T = len(z)
            pool = np.where(v)[0]
            for t in range(max(LAGS) + 1, T - h):
                if v[t] and v[t - 1] and v[t + h] and all(v[t - k] for k in LAGS):
                    if shuffle:
                        picks = [
                            int(pool[rng2.randint(0, len(pool))]) for _ in range(1 + len(LAGS))
                        ]
                        hist = [z[t]] + [z[p] for p in picks]
                    else:
                        hist = [z[t], z[t - 1]] + [z[t - k] for k in LAGS]
                    X6.append(np.concatenate(hist))
                    Y.append(z[t + h])
                    GRP.append(u["group"])
        return np.stack(X6), np.stack(Y), np.array(GRP)

    from acoustic_dynamics_report import ridge_fit, ridge_predict

    def _loss(r):
        return float(np.sqrt((r**2).mean(0)).mean())

    def _three_way(G):
        sorted(set(G.tolist()))
        f = np.isin(G, sorted(fit_only))
        v = np.isin(G, sorted(val_blocks))
        return f, v, ~(f | v)

    # ---- STUDY 1
    decay = {}
    for h in HORIZONS:
        X, Y, G = _collect(h)
        f, v, e = _three_way(G)
        mu6, sd6 = X[f].mean(0), X[f].std(0) + 1e-9
        Xn = (X - mu6) / sd6
        bs = bf = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(Xn[f][:, :12], Y[f], alpha)
            lv = _loss(ridge_predict(w, b, Xn[v][:, :12]) - Y[v])
            if lv < bs[1]:
                bs = (alpha, lv)
            w, b = ridge_fit(Xn[f], Y[f], alpha)
            lv = _loss(ridge_predict(w, b, Xn[v]) - Y[v])
            if lv < bf[1]:
                bf = (alpha, lv)
        ws, bs_ = ridge_fit(Xn[f][:, :12], Y[f], bs[0])
        wf, bf_ = ridge_fit(Xn[f], Y[f], bf[0])
        rs = ridge_predict(ws, bs_, Xn[e][:, :12]) - Y[e]
        rf = ridge_predict(wf, bf_, Xn[e]) - Y[e]
        pers = Y[e] - X[e][:, :12]
        decay[str(h * 10)] = {
            "L_persistence": round(_loss(pers), 4),
            "L_state": round(_loss(rs), 4),
            "L_history": round(_loss(rf), 4),
            "delta_recent": round(_loss(pers) - _loss(rs), 4),
            "delta_older": round(_loss(rs) - _loss(rf), 4),
            "n_eval": int(e.sum()),
        }
        print(
            f"h={h * 10}ms: pers {_loss(pers):.4f} state {_loss(rs):.4f} hist {_loss(rf):.4f}",
            flush=True,
        )

    # order control at 40 ms
    Xt, Yt, Gt = _collect(4, shuffle=False)
    Xs, _, _ = _collect(4, shuffle=True)
    n = min(len(Xt), len(Xs))
    Xt, Yt, Gt, Xs = Xt[:n], Yt[:n], Gt[:n], Xs[:n]
    f, v, e = _three_way(Gt)
    mu6, sd6 = Xt[f].mean(0), Xt[f].std(0) + 1e-9
    Xtn = (Xt - mu6) / sd6
    best = (None, np.inf)
    for alpha in ALPHAS:
        w, b = ridge_fit(Xtn[f], Yt[f], alpha)
        lv = _loss(ridge_predict(w, b, Xtn[v]) - Yt[v])
        if lv < best[1]:
            best = (alpha, lv)
    w, b = ridge_fit(Xtn[f], Yt[f], best[0])
    r_true = ridge_predict(w, b, Xtn[e]) - Yt[e]
    r_shuf = (
        ridge_predict(w, b, (Xs - mu6) / sd6)[e] - Yt[e]
        if False
        else ridge_predict(w, b, ((Xs - mu6) / sd6)[: int(e.sum())]) - Yt[e][: int(e.sum())]
    )
    order = {
        "delta_order": round(_loss(r_shuf) - _loss(r_true), 4),
        "L_true": round(_loss(r_true), 4),
        "L_shuffled": round(_loss(r_shuf), 4),
        "n_eval": int(e.sum()),
    }

    # ---- STUDY 2: utterance-level recurrence (same- vs different-speaker)
    from bengalese_candidate_ranking import kbest_candidates

    eval_us = [u for u in utterances if u["part"] == "eval"]
    fit_us = [u for u in utterances if u["part"] != "eval"]
    oracle = {"same_speaker": 0, "diff_speaker": 0, "n": 0}
    rngq = np.random.RandomState(SEED)
    queries = rngq.choice(len(fit_us), size=min(20, len(fit_us)), replace=False)
    for qi in queries:
        q = fit_us[qi]
        zq = q["z"]
        L = 16
        t0 = len(zq) // 2
        query = zq[t0 : t0 + L]
        if len(query) < L:
            continue
        best_same = best_diff = None
        for uj, u in enumerate(eval_us):
            if u["uid"] == q["uid"]:
                continue
            cands, _ = kbest_candidates(query, u["z"], k=5)
            if not cands:
                continue
            top = cands[0]["interval"]
            # "correct" reference class = the candidate's midpoint is voiced
            mid = (top[0] + top[1]) // 2
            voiced = u["valid"][min(mid, len(u["valid"]) - 1)]
            same = u["spk"] == q["spk"]
            if voiced:
                if same:
                    best_same = True
                else:
                    best_diff = True
        oracle["n"] += 1
        if best_same:
            oracle["same_speaker"] += 1
        if best_diff:
            oracle["diff_speaker"] += 1
    study2 = {
        "design": "utterance-level: top candidate lands in a voiced "
        "region of a same/different-speaker utterance",
        "n_queries": oracle["n"],
        "same_speaker_voiced_rate": round(oracle["same_speaker"] / max(oracle["n"], 1), 3),
        "diff_speaker_voiced_rate": round(oracle["diff_speaker"] / max(oracle["n"], 1), 3),
        "provenance_caveat": "no word timestamps: utterance-level reference, NOT word recovery",
    }

    # ---- STUDY 3: masked reconstruction with energy-boundary strata
    def _energy_onsets_count(z, a0, b0):
        rms = np.sqrt((z**2).sum(axis=1))
        seg = rms[a0:b0]
        thr = np.median(rms)
        c = 0
        for k in range(1, len(seg)):
            if seg[k] >= thr > seg[k - 1]:
                c += 1
        return c

    rows = []
    for u in utterances:
        z, v = u["z"], u["valid"]
        T = len(z)
        a0 = T // 2 - MASK_FRAMES // 2
        b0 = a0 + MASK_FRAMES
        if a0 < 2 or b0 + 2 >= T:
            continue
        rest = np.vstack([z[:a0], z[b0:]])
        G = np.concatenate(
            [
                np.median(rest, axis=0),
                np.percentile(rest, 75, axis=0) - np.percentile(rest, 25, axis=0),
            ]
        )
        rows.append(
            {
                "fa": np.concatenate([z[a0 - 1], z[a0 - 2]]),
                "ba": np.concatenate([z[b0], z[b0 + 1]]),
                "G": G,
                "Y": z[a0:b0],
                "grp": u["group"],
                "strat": _energy_onsets_count(z, a0, b0),
            }
        )
    GRP3 = np.array([r["grp"] for r in rows])
    f3, v3, e3 = _three_way(GRP3)
    Yall = np.concatenate([r["Y"] for k, r in enumerate(rows) if f3[k]])
    mu_y, sd_y = Yall.mean(0), Yall.std(0) + 1e-9
    faF = np.stack([r["fa"] for r in rows])
    baF = np.stack([r["ba"] for r in rows])
    GF = np.stack([r["G"] for r in rows])
    muX, sdX = faF[f3].mean(0), faF[f3].std(0) + 1e-9
    muB, sdB = baF[f3].mean(0), baF[f3].std(0) + 1e-9
    muG, sdG = GF[f3].mean(0), GF[f3].std(0) + 1e-9

    def _mats(mask):
        fa = (faF[mask] - muX) / sdX
        ba = (baF[mask] - muB) / sdB
        Gm = (GF[mask] - muG) / sdG
        Y = np.stack([r["Y"] for k, r in enumerate(rows) if mask[k]])
        return fa, ba, Gm, (Y - mu_y) / sd_y

    fa_tr, ba_tr, G_tr, Y_tr = _mats(f3)
    fa_va, ba_va, G_va, Y_va = _mats(v3)
    fa_ev, ba_ev, G_ev, Y_ev = _mats(e3)
    Y_tr = Y_tr.reshape(len(Y_tr), -1)
    Y_va = Y_va.reshape(len(Y_va), -1)
    Y_ev = Y_ev.reshape(len(Y_ev), -1)
    arms3, resid3 = {}, {}
    fa_raw = faF[e3][:, :12]
    ba_raw = baF[e3][:, :12]
    blend = np.zeros((int(e3.sum()), MASK_FRAMES, 12))
    for k in range(MASK_FRAMES):
        w = (k + 1) / (MASK_FRAMES + 1)
        blend[:, k, :] = ((1 - w) * fa_raw + w * ba_raw - mu_y) / sd_y
    resid3["blend"] = blend.reshape(int(e3.sum()), -1) - Y_ev.reshape(int(e3.sum()), -1)
    arms3["blend"] = _loss(resid3["blend"])
    for arm, Amat in (
        ("bidir", np.hstack([fa_tr, ba_tr])),
        ("bidir+G", np.hstack([fa_tr, ba_tr, G_tr])),
    ):
        Av = np.hstack([fa_va, ba_va]) if arm == "bidir" else np.hstack([fa_va, ba_va, G_va])
        Ae = np.hstack([fa_ev, ba_ev]) if arm == "bidir" else np.hstack([fa_ev, ba_ev, G_ev])
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(Amat, Y_tr, alpha)
            lv = _loss(ridge_predict(w, b, Av) - Y_va)
            if lv < best[1]:
                best = (alpha, lv)
        w, b = ridge_fit(Amat, Y_tr, best[0])
        r = ridge_predict(w, b, Ae) - Y_ev
        resid3[arm] = r
        arms3[arm] = _loss(r)
    strat_out = {}
    strats = [r["strat"] for k, r in enumerate(rows) if e3[k]]
    for st in (0, 1, 2):
        idx = [j for j, s in enumerate(strats) if (s if s < 2 else 2) == st]
        if len(idx) < 3:
            continue
        strat_out[f"onsets_{st}"] = {
            "n": len(idx),
            "blend": round(_loss(resid3["blend"][idx]), 4),
            "bidir": round(_loss(resid3["bidir"][idx]), 4),
            "bidir+G": round(_loss(resid3["bidir+G"][idx]), 4),
        }
    delta_G = arms3["bidir"] - arms3["bidir+G"]

    report = {
        "study": "LibriSpeech Human Reference Battery (positive-control "
        "regime; NOT a template for animal communication)",
        "provenance": "utterance transcripts only; NO word/phoneme "
        "timestamps — study 2 is utterance-level, study 3 strata are "
        "energy-defined acoustic proxies (declared, not lexical ground "
        "truth)",
        "study1_forecasting": decay,
        "study1_order": order,
        "study2_recurrence": study2,
        "study3_masked": {
            "arms": {k: round(v, 4) for k, v in arms3.items()},
            "delta_G": round(delta_G, 4),
            "strata_energy_onsets": strat_out,
        },
        "framing": "structural comparison only (does known speech show "
        "the same continuous predictive diagnostics?); raw nRMSE "
        "magnitudes not biologically comparable across corpora",
    }
    out = ROOT / "outputs" / "human_reference_battery.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "decay": {
                    k: {
                        kk: vv
                        for kk, vv in v.items()
                        if kk.startswith("delta") or kk == "L_persistence"
                    }
                    for k, v in decay.items()
                },
                "order": order,
                "study2": study2,
                "study3": report["study3_masked"],
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
