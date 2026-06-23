#!/usr/bin/env python3
"""Soundwel Track C ablation: does acoustic structure predict valence/context beyond call type?

Soundwel = 6887 labelled pig vocalizations (44.1 kHz mono, ~0.3–0.5 s) with
Valence / production Context / Call Type / Sex / Recording Team (farm). The clean Track C
test of the project's central question: *does acoustic structure predict emotional valence
or context beyond vocalisation type?*

Critical confound handled head-on: valence is tied to context + call type BY DESIGN
(Castration/Crushing/NegativeConditioning→Neg; Enriched/PositiveConditioning→Pos; HF screams
vs LF grunts). So a naive fp→valence probe mostly recovers that mapping. The honest tests are
call-type-controlled: (a) call-type→valence baseline, (b) fp→valence, (c) fp+call-type→valence
(Δ = the answer), (d) within-call-type fp→valence, (e) sex-controlled, (f) context-balanced.

Six feature sets (whole-call fp, HECB-pooled, relative-sequence, DUSTED motif histogram,
call-type one-hot, whole+call-type) × 3 targets (vocalisation_type / context / valence),
Recording-Team-held-out LogReg macro-F1.

Honest framing: valence is a SEMANTIC/ethological label — prediction here is acoustic
ASSOCIATION, not validated emotion. `semantic_validation` stays `none`. Pig is a new species
using only the all-species fingerprint + raw NBD (no learned tokens, no cross-species weights)
— firewall-clean.

Run:  python scripts/soundwel_trackc.py --max-calls 6887

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/mnt/c/Users/sheel/Desktop/data/Soundwel Dataset - Audio and Spectrograms")
KEY = Path("/mnt/c/Users/sheel/Desktop/data/SoundwelDatasetKey.xlsx")
SR = 44100
K_SUB = 24  # sub-call unit k-means
TOP_MOTIFS = 50  # motif-histogram dim cap
CONTEXT_MIN = 100  # min class size to keep a Context for the context target
_PROPOSAL = dict(floor_energy=0.05, floor_flux=0.03, debounce_ms=15.0)


def load_key(max_calls):
    df = pd.read_excel(KEY)
    df = df[df["Audio Filename"].notna()].copy()
    if max_calls and len(df) > max_calls:
        df = df.sample(n=max_calls, random_state=0)
    return df.reset_index(drop=True)


def _val(row, col):
    v = row.get(col)
    return "NaN" if pd.isna(v) else str(v).strip()


def build_recs(df):
    """Per call: whole-call fp + NBD sub-call unit fingerprints + metadata."""
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="").combined  # noqa: E731
    recs = []
    for i, row in df.iterrows():
        path = DATA / row["Audio Filename"]
        if not path.exists():
            continue
        try:
            audio, _ = load_audio(str(path), target_sr=SR)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.03 * SR):
            continue
        whole = np.asarray(fp_fn(audio, SR), dtype=np.float32)
        if not np.any(whole):
            continue
        sig = compute_signals(audio, SR, hop_ms=5.0)
        props = generate_proposals(sig, **_PROPOSAL)
        sub_fps = []
        for p in props:
            si = max(0, int(p.time_ms / 1000 * SR) - int(0.02 * SR))
            sj = si + int(0.04 * SR)
            seg = audio[si:sj]
            if len(seg) >= int(0.012 * SR):
                v = np.asarray(fp_fn(seg, SR), dtype=np.float32)
                if np.any(v):
                    sub_fps.append(v)
        recs.append(
            dict(
                whole=whole,
                sub_fps=sub_fps,
                valence=_val(row, "Valence"),
                context=_val(row, "Context"),
                calltype=_val(row, "Call Type (Freq)"),
                sex=_val(row, "Sex"),
                team=_val(row, "Recording Team"),
            )
        )
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)} → {len(recs)} calls", flush=True)
    return recs


def kmeans_clusters(list_of_vecs, k):
    """Global k-means over all sub-call fps → per-call cluster-id list (or None if too few)."""
    all_v = [v for sub in list_of_vecs for v in sub]
    if len(all_v) < k * 5:
        return None
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(np.asarray(all_v, dtype=np.float32))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init="auto", batch_size=4096).fit(
        scaler.transform(np.asarray(all_v, dtype=np.float32))
    )
    cents = km.cluster_centers_
    out = []
    for sub in list_of_vecs:
        if not sub:
            out.append([])
            continue
        sv = scaler.transform(np.asarray(sub, dtype=np.float32))
        out.append(
            np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1).tolist()
        )
    return out


def _contains(clusters, sig):
    n, m = len(clusters), len(sig)
    if m == 0 or m > n:
        return False
    for i in range(n - m + 1):
        if tuple(clusters[i : i + m]) == sig:
            return True
    return False


def discover_motifs(recs, clusters_by_call, args):
    """DTW discovery over ≥2-sub-unit calls (capped sample) → top motifs (id, signature)."""
    cand = [i for i, c in enumerate(clusters_by_call) if len(c) >= 2]
    if len(cand) < 200:
        return []
    sample = cand[: args.n_motif_sample] if len(cand) > args.n_motif_sample else cand
    sm_calls = [
        sm.CallUnits(
            fingerprints=recs[i]["sub_fps"],
            clusters=clusters_by_call[i],
            calltype=recs[i]["calltype"],
            caller=recs[i]["team"],
            file_id=str(i),
            call_idx=j,
        )
        for j, i in enumerate(sample)
    ]
    common = dict(min_n=2, max_n=4, min_support=3, min_calls=2)
    dtw = sm.discover_recurrent_subsequences(
        sm_calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    s = sm.summarize_discovery(dtw, len(sm_calls))
    print(
        f"  DUSTED DTW: fams={s['n_motifs']} cov={s['coverage']:.3f} H={s['family_entropy_norm']:.2f} "
        f"top={s['top_family_support_fraction']:.2f} | discovery n={len(sm_calls)}",
        flush=True,
    )
    ranked = sorted(
        [(i, tuple(m.signature_pattern)) for i, m in enumerate(dtw) if m.signature_pattern],
        key=lambda kv: -dtw[kv[0]].support,
    )
    return ranked[:TOP_MOTIFS]


def motif_histogram(clusters, top_motifs):
    """Per-call multi-hot over top motif families by signature-presence."""
    vec = np.zeros(len(top_motifs), dtype=np.float32)
    for j, (_, sig) in enumerate(top_motifs):
        if _contains(clusters, sig):
            vec[j] = 1.0
    return vec


def seq_vector(whole, clusters, k):
    """relative-sequence: whole ‖ sub-call cluster histogram (k) ‖ [n_sub, internal_frac]."""
    hist = np.zeros(k, dtype=np.float32)
    for c in clusters:
        if 0 <= c < k:
            hist[c] += 1
    s = hist.sum()
    if s > 0:
        hist /= s
    n_sub = min(len(clusters) / 10.0, 1.0)
    internal_frac = max(0.0, (len(clusters) - 1)) / 10.0 if len(clusters) > 1 else 0.0
    return np.concatenate([whole, hist, [n_sub, internal_frac]])


def probe_team_holdout(X, y, teams, n_holdout=2, seed=0):
    """LogReg macro-F1, Recording-Team-held-out; returns dict or None."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    uniq = sorted(set(teams))
    if len(uniq) < 3:
        return None
    rng = random.Random(seed)
    rng.shuffle(uniq)
    test_teams = set(uniq[-n_holdout:])
    tr = [i for i, t in enumerate(teams) if t not in test_teams]
    te = [i for i, t in enumerate(teams) if t in test_teams]
    common = set(y[i] for i in tr) & set(y[i] for i in te)
    tr = [i for i in tr if y[i] in common]
    te = [i for i in te if y[i] in common]
    if len(tr) < 20 or len(te) < 10 or len(common) < 2:
        return None
    Xtr = np.asarray([X[i] for i in tr], dtype=np.float32)
    Xte = np.asarray([X[i] for i in te], dtype=np.float32)
    ytr = [y[i] for i in tr]
    yte = [y[i] for i in te]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0).fit(
        sc.transform(Xtr), ytr
    )
    pred = clf.predict(sc.transform(Xte))
    return dict(
        f1=f1_score(yte, pred, average="macro"),
        chance=1.0 / len(common),
        n_te=len(te),
        n_cls=len(common),
    )


def _fmt(r):
    if r is None:
        return "    n/a"
    return f"{r['f1']:.3f} (ch {r['chance']:.2f}, n={r['n_te']}, k={r['n_cls']})"


def main():
    ap = argparse.ArgumentParser(description="Soundwel Track C ablation")
    ap.add_argument("--max-calls", type=int, default=6887)
    ap.add_argument("--k-sub", type=int, default=K_SUB)
    ap.add_argument("--n-motif-sample", type=int, default=2000)
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading Soundwel key...", flush=True)
    df = load_key(args.max_calls)
    print(
        f"  {len(df)} calls | valence {dict(Counter(df['Valence'].astype(str)))} | "
        f"{df['Context'].nunique()} contexts | {df['Call Type (Freq)'].nunique()} call types | "
        f"{df['Recording Team'].nunique()} teams",
        flush=True,
    )

    print("building whole-call fp + NBD sub-call units...", flush=True)
    recs = build_recs(df)
    n_sub2 = sum(1 for r in recs if len(r["sub_fps"]) >= 2)
    print(
        f"  {len(recs)} calls with fp | {n_sub2} ({100 * n_sub2 / max(len(recs), 1):.0f}%) have ≥2 sub-units",
        flush=True,
    )

    # sub-call cluster ids + DTW motif discovery (on ≥2-sub subset)
    clusters_by_call = kmeans_clusters([r["sub_fps"] for r in recs], args.k_sub) or [
        [] for _ in recs
    ]
    print("DUSTED motif discovery on ≥2-sub-unit calls...", flush=True)
    top_motifs = discover_motifs(recs, clusters_by_call, args)

    # calltype codebook
    ct_types = sorted({r["calltype"] for r in recs if r["calltype"] != "NaN"})
    ct_idx = {c: i for i, c in enumerate(ct_types)}

    def ct_onehot(r):
        v = np.zeros(len(ct_types), dtype=np.float32)
        if r["calltype"] != "NaN":
            v[ct_idx[r["calltype"]]] = 1.0
        return v

    # assemble feature vectors per rec
    feats = {"whole": [], "hecb": [], "seq": [], "motif": [], "ctype": [], "whole+ctype": []}
    for r, cl in zip(recs, clusters_by_call):
        oh = ct_onehot(r)
        feats["whole"].append(r["whole"])
        feats["whole+ctype"].append(np.concatenate([r["whole"], oh]))
        feats["ctype"].append(oh)
        if len(r["sub_fps"]) >= 2:
            feats["hecb"].append(np.mean(np.asarray(r["sub_fps"], dtype=np.float32), axis=0))
            feats["seq"].append(seq_vector(r["whole"], cl, args.k_sub))
            feats["motif"].append(motif_histogram(cl, top_motifs))
        else:
            feats["hecb"].append(None)
            feats["seq"].append(None)
            feats["motif"].append(None)

    # ── coverage ──
    print(f"\n{'=' * 92}\nSOUNDWEL Track C — feature coverage + 6×3 probe table\n{'=' * 92}")
    print(
        f"  whole-call fp: {len(feats['whole'])}/{len(recs)} | calltype one-hot: "
        f"{sum(1 for r in recs if r['calltype'] != 'NaN')}/{len(recs)}"
    )
    print(f"  ≥2-sub-unit features (HECB-pooled / sequence / motif-hist): {n_sub2}/{len(recs)}")

    teams = [r["team"] for r in recs]
    # restrict context target to classes with ≥ CONTEXT_MIN
    ctx_valid = {c for c, n in Counter(r["context"] for r in recs).items() if n >= CONTEXT_MIN}

    def target_vec(target_name, filter_fn=None):
        idxs = [i for i, r in enumerate(recs) if (filter_fn(r) if filter_fn else True)]
        y = [recs[i][target_name] for i in idxs]
        return idxs, y

    def run_cell(feat_key, idxs, y):
        X = [feats[feat_key][i] for i in idxs]
        if any(x is None for x in X):
            keep = [j for j, x in enumerate(X) if x is not None]
            X = [X[j] for j in keep]
            yy = [y[j] for j in keep]
            sub_teams = [teams[idxs[j]] for j in keep]
        else:
            yy = y
            sub_teams = [teams[i] for i in idxs]
        return probe_team_holdout(X, yy, sub_teams, seed=args.seed)

    ct_idx_list, ct_y = target_vec("calltype", lambda r: r["calltype"] != "NaN")

    def _ctx_valid(r):
        return r["context"] in ctx_valid

    cx_idx_list, cx_y = target_vec("context", _ctx_valid)
    va_idx_list, va_y = target_vec("valence")

    targets = [
        ("vocalisation_type", ct_idx_list, ct_y, ["whole", "hecb", "seq", "motif"]),
        ("production_context", cx_idx_list, cx_y, ["whole", "ctype", "whole+ctype"]),
        ("emotional_valence", va_idx_list, va_y, ["whole", "ctype", "whole+ctype"]),
    ]

    for tname, idxs, y, fkeys in targets:
        print(f"\n  target = {tname} (n={len(idxs)}, {len(set(y))} classes)")
        for fk in fkeys:
            print(f"    {fk:14} {_fmt(run_cell(fk, idxs, y))}")

    # ── decisive valence block ──
    print(
        f"\n{'=' * 92}\nDECISIVE: does acoustic structure predict VALENCE beyond call type?\n{'=' * 92}"
    )
    va_ct = [i for i in va_idx_list if recs[i]["calltype"] != "NaN"]
    # call-type → valence (baseline) on non-NaN calltype
    print(
        f"  call-type → valence (baseline)      {_fmt(run_cell('ctype', va_ct, [recs[i]['valence'] for i in va_ct]))}"
    )
    print(f"  fp → valence                        {_fmt(run_cell('whole', va_idx_list, va_y))}")
    print(
        f"  fp + call-type → valence            {_fmt(run_cell('whole+ctype', va_idx_list, va_y))}"
    )
    # within-call-type: pool calltypes that have BOTH valences
    both = {
        c
        for c in ct_types
        if any(recs[i]["calltype"] == c and recs[i]["valence"] == "Pos" for i in va_idx_list)
        and any(recs[i]["calltype"] == c and recs[i]["valence"] == "Neg" for i in va_idx_list)
    }
    wct = [i for i in va_idx_list if recs[i]["calltype"] in both]
    print(
        f"  within-call-type fp → valence       {_fmt(run_cell('whole', wct, [recs[i]['valence'] for i in wct]))}"
        f"  [{len(both)} calltypes, n={len(wct)}]"
    )
    # sex-controlled
    for sx in ("male", "female"):
        sx_idx = [i for i in va_idx_list if recs[i]["sex"] == sx]
        print(
            f"  fp → valence ({sx:>6})              {_fmt(run_cell('whole', sx_idx, [recs[i]['valence'] for i in sx_idx]))}"
        )
    # context-balanced: within each context, downsample to min(pos,neg)
    rng = random.Random(args.seed)
    keep_idx = []
    for ctx in ctx_valid:
        c_idx = [i for i in va_idx_list if recs[i]["context"] == ctx]
        pos = [i for i in c_idx if recs[i]["valence"] == "Pos"]
        neg = [i for i in c_idx if recs[i]["valence"] == "Neg"]
        if pos and neg:
            n = min(len(pos), len(neg))
            rng.shuffle(pos)
            rng.shuffle(neg)
            keep_idx += pos[:n] + neg[:n]
    if keep_idx:
        print(
            f"  context-balanced fp → valence       "
            f"{_fmt(run_cell('whole', keep_idx, [recs[i]['valence'] for i in keep_idx]))}"
            f"  [n={len(keep_idx)}]"
        )

    print("\n  Read: if fp+call-type ≈ call-type (and within-call-type fp→valence ≈ chance),")
    print("  acoustic structure does NOT predict valence beyond call form → strengthens the")
    print("  'context/valence needs more than acoustics' thesis (semantic_validation: none).")
    print("  Valence prediction here = acoustic ASSOCIATION, not validated emotion.")
    print("\nSOUNDWEL TRACK C DONE")


if __name__ == "__main__":
    main()
