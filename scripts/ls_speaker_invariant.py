#!/usr/bin/env python3
"""LibriSpeech speaker-invariant / universal acoustic-structure test.

The §14 unified run showed NN_xsp ≈ NN (same-word nearest-neighbour survives forcing
cross-speaker matches). This makes it explicit and decisive: does the ordered acoustic
sub-unit representation preserve WORD-associated structure independently of SPEAKER identity?

Key test — the inequality:
    dist(same-word, different-speaker)  <  dist(different-word, same-speaker)
( + duration- / phoneme-count-matched controls so it isn't a length artifact), and
speaker-held-out cross-speaker same-word retrieval (top-1/5, MRR).

NO re-extraction: loads outputs/ls_unified/recs.pkl (2903 word instances, 178 speakers, 325
multi-speaker word types) — every rec carries fp_whole (44-D), fp/hecb sub-unit fps, cpc
frames (256-D), hubert frames (768-D) + word/speaker/dur/nphon. K-means + distances + probes
only (CPU, fast).

Representations: fp_whole (cosine); fingerprint/HECB/CPC/HuBERT-km cluster sequences (normalized
Levenshtein); HuBERT-DTW (raw-frame DTW, capped subset — the §16 point that HuBERT needs DTW).
Plus word-vs-speaker probes and a speaker-centering intervention on fp_whole.

Manuscript-safe: "universal" = cross-speaker word-associated acoustic structure, NOT validated
meaning (semantic ceiling holds; tokens ≠ words). semantic_validation stays none. Human-speech
word GT is honest here. CPC/HuBERT are human-only (firewall unchanged).

Run:  python scripts/ls_speaker_invariant.py [--recs outputs/ls_unified/recs.pkl]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def dtw_frames(a, b):
    """Length-normalised DTW over two (T, D) frame arrays with cosine substitution cost.

    (sm.dtw_distance expects SubSequence objects, so this is a standalone frame DTW —
    the zero-resource 'DTW over raw HuBERT frames' metric.)
    """
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    m, n = len(an), len(bn)
    if m == 0 or n == 0:
        return 1.0
    inf = float("inf")
    prev = [inf] * (n + 1)
    prev[0] = 0.0
    for i in range(1, m + 1):
        cur = [inf] * (n + 1)
        ai = an[i - 1]
        for j in range(1, n + 1):
            cost = max(0.0, 1.0 - float(np.dot(ai, bn[j - 1])))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n] / max(m, n)


RECS = ROOT / "outputs" / "ls_unified" / "recs.pkl"
K_FP = 32
K_SSL = 50
N_PAIRS = 2000  # sampled pairs per 4-way condition
EVAL_CAP = 400  # max eval instances for sequence retrieval (Levenshtein is pure-Python)
DTW_CAP = 300  # max instances for HuBERT-DTW


def edit_dist_norm(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0 if m == n else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[n] / max(m, n)


def collapse_repeats(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def kmeans_tokens(list_of_vecs, k, seed=0):
    """Global k-means → per-element collapsed cluster-id sequence (aligned to input order)."""
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    all_v = [np.asarray(v, np.float32) for sub in list_of_vecs for v in sub]
    if len(all_v) < k * 5:
        return None
    X = np.asarray(all_v, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096).fit(
        scaler.transform(X)
    )
    cents = km.cluster_centers_
    out = []
    for sub in list_of_vecs:
        if len(sub) == 0:  # works for python lists AND numpy 2-D arrays
            out.append([])
            continue
        sv = scaler.transform(np.asarray(sub, dtype=np.float32))
        out.append(
            collapse_repeats(
                np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1).tolist()
            )
        )
    return out


def sample_pairs(recs, predicate, n, rng):
    idx = list(range(len(recs)))
    out, tries = [], 0
    while len(out) < n and tries < n * 40:
        i, j = rng.choice(idx), rng.choice(idx)
        tries += 1
        if i != j and predicate(i, j, recs):
            out.append((i, j))
    return out


def four_way(recs, dist, rng, n=N_PAIRS):
    def sw_ss(a, b, r):
        return r[a]["word"] == r[b]["word"] and r[a]["speaker"] == r[b]["speaker"]

    def sw_ds(a, b, r):
        return r[a]["word"] == r[b]["word"] and r[a]["speaker"] != r[b]["speaker"]

    def dw_ss(a, b, r):
        return r[a]["word"] != r[b]["word"] and r[a]["speaker"] == r[b]["speaker"]

    def dw_ds(a, b, r):
        return r[a]["word"] != r[b]["word"] and r[a]["speaker"] != r[b]["speaker"]

    def dw_dur(a, b, r):
        return (
            r[a]["word"] != r[b]["word"]
            and r[a]["speaker"] == r[b]["speaker"]
            and abs(r[a]["dur"] - r[b]["dur"]) < 0.25 * max(r[a]["dur"], r[b]["dur"])
        )

    def dw_phn(a, b, r):
        return (
            r[a]["word"] != r[b]["word"]
            and r[a]["speaker"] == r[b]["speaker"]
            and abs(r[a]["nphon"] - r[b]["nphon"]) <= 1
        )

    def md(pred):
        pairs = sample_pairs(recs, pred, n, rng)
        return float(np.mean([dist(a, b) for a, b in pairs])) if pairs else None

    return {
        k: md(p)
        for k, p in [
            ("sw_ss", sw_ss),
            ("sw_ds", sw_ds),
            ("dw_ss", dw_ss),
            ("dw_ds", dw_ds),
            ("dw_dur", dw_dur),
            ("dw_phn", dw_phn),
        ]
    }


def retrieval(recs, dist, eval_idx, gallery_idx, rng):
    """Cross-speaker same-word retrieval: gallery excludes the query's speaker."""
    gwords = [recs[i]["word"] for i in gallery_idx]
    gspk = [recs[i]["speaker"] for i in gallery_idx]
    chance = sum((gwords.count(w) / len(gwords)) ** 2 for w in set(gwords))
    top1 = top5 = mrr = 0.0
    neval = 0
    for q in eval_idx:
        qs, qw = recs[q]["speaker"], recs[q]["word"]
        gal = [g for k, g in enumerate(gallery_idx) if gspk[k] != qs]
        if len(gal) < 5:
            continue
        ds = sorted(((dist(q, g), recs[g]["word"]) for g in gal), key=lambda x: x[0])
        ranks = [k for k, (_, w) in enumerate(ds) if w == qw]
        neval += 1
        if ranks:
            r0 = ranks[0]
            top1 += r0 == 0
            top5 += r0 < 5
            mrr += 1.0 / (r0 + 1)
    neval = max(neval, 1)
    return dict(top1=top1 / neval, top5=top5 / neval, mrr=mrr / neval, chance=chance, n=neval)


def probe_disentangle(X, y_word, y_spk, seed=0):
    """Word probe (speaker-held-out) + speaker probe + word-after-speaker-regression."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float32)
    rng = random.Random(seed)
    n = len(X)

    def wp(Xin):
        spks = sorted(set(y_spk))
        rng.shuffle(spks)
        tr_spk = set(spks[: int(0.7 * len(spks))])
        tr = [i for i in range(n) if y_spk[i] in tr_spk]
        te = [i for i in range(n) if y_spk[i] not in tr_spk]
        common = set(y_word[i] for i in tr) & set(y_word[i] for i in te)
        tr = [i for i in tr if y_word[i] in common]
        te = [i for i in te if y_word[i] in common]
        if len(tr) < 20 or len(te) < 10 or len(common) < 2:
            return None
        sc = StandardScaler().fit(Xin[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0).fit(
            sc.transform(Xin[tr]), [y_word[i] for i in tr]
        )
        return f1_score(
            [y_word[i] for i in te], clf.predict(sc.transform(Xin[te])), average="macro"
        )

    def sp(Xin):
        idx = list(range(n))
        rng.shuffle(idx)
        tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n) :]
        common = set(y_spk[i] for i in tr) & set(y_spk[i] for i in te)
        tr = [i for i in tr if y_spk[i] in common]
        te = [i for i in te if y_spk[i] in common]
        if len(tr) < 20 or len(te) < 10 or len(common) < 2:
            return None, None
        sc = StandardScaler().fit(Xin[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0).fit(
            sc.transform(Xin[tr]), [y_spk[i] for i in tr]
        )
        return f1_score(
            [y_spk[i] for i in te], clf.predict(sc.transform(Xin[te])), average="macro"
        ), 1.0 / len(common)

    word = wp(X)
    spk, spk_ch = sp(X)
    spk_ids = {s: i for i, s in enumerate(sorted(set(y_spk)))}
    Oh = np.zeros((n, len(spk_ids)), dtype=np.float32)
    for i, s in enumerate(y_spk):
        Oh[i, spk_ids[s]] = 1.0
    beta, *_ = np.linalg.lstsq(Oh, X, rcond=None)
    word_resid = wp(X - Oh @ beta)
    return dict(
        word=word, speaker=spk, speaker_chance=spk_ch, word_after_speaker_regression=word_resid
    )


def main():
    ap = argparse.ArgumentParser(description="LibriSpeech speaker-invariant structure test")
    ap.add_argument("--recs", type=str, default=str(RECS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.recs, "rb") as f:
        recs = pickle.load(f)  # noqa: S301
    by_word = defaultdict(list)
    for i, r in enumerate(recs):
        by_word[r["word"]].append(i)
    multi = {w for w, idxs in by_word.items() if len({recs[i]["speaker"] for i in idxs}) >= 2}
    mrecs_idx = [i for i, r in enumerate(recs) if r["word"] in multi]
    print(
        f"loaded {len(recs)} instances | {len(multi)} multi-speaker word types | "
        f"{len(mrecs_idx)} cross-speaker-eval instances",
        flush=True,
    )

    print("k-means cluster sequences...", flush=True)
    fp_seq = kmeans_tokens([r["fp"] for r in recs], K_FP, args.seed)
    hecb_seq = kmeans_tokens([r["hecb"] for r in recs], K_FP, args.seed)
    cpc_seq = kmeans_tokens(
        [r["cpc"] if r["cpc"] is not None else [] for r in recs], K_SSL, args.seed
    )
    hub_seq = kmeans_tokens([r["hubert"] for r in recs], K_SSL, args.seed)
    print("  done", flush=True)

    fpw = np.asarray([r["fp_whole"] for r in recs], dtype=np.float32)
    fpw_n = fpw / (np.linalg.norm(fpw, axis=1, keepdims=True) + 1e-12)

    def d_cos(i, j):
        return float(1.0 - np.dot(fpw_n[i], fpw_n[j]))

    def mk_seq_dist(seq):
        def _d(i, j):
            return edit_dist_norm(seq[i], seq[j]) if seq[i] and seq[j] else 1.0

        return _d

    reps = [
        ("fp_whole (cos)", d_cos),
        ("fingerprint-km", mk_seq_dist(fp_seq)),
        ("HECB", mk_seq_dist(hecb_seq)),
        ("CPC-km", mk_seq_dist(cpc_seq)),
        ("HuBERT-km", mk_seq_dist(hub_seq)),
    ]

    # ── 4-way distances ──
    print(
        f"\n{'=' * 92}\nSPEAKER-INVARIANT TEST — 4-way distances (multi-speaker words)\n{'=' * 92}"
    )
    print(
        f"  {'representation':20} {'sw/ss':>6} {'sw/ds':>6} {'dw/ss':>6} {'dw/ds':>6} "
        f"{'dw/dur':>7} {'dw/phn':>7} {'ratio':>6}"
    )
    print("  " + "-" * 78)
    for name, d in reps:
        fw = four_way(recs, d, rng)
        ratio = (fw["dw_ss"] / fw["sw_ds"]) if (fw["dw_ss"] and fw["sw_ds"]) else float("nan")
        print(
            f"  {name:20} {fw['sw_ss']:6.3f} {fw['sw_ds']:6.3f} {fw['dw_ss']:6.3f} "
            f"{fw['dw_ds']:6.3f} {fw['dw_dur']:7.3f} {fw['dw_phn']:7.3f} {ratio:6.2f}"
        )
    print(
        "  sw/ds = same-word diff-speaker; dw/ss = diff-word same-speaker; "
        "ratio = dw/ss ÷ sw/ds (>1 ⇒ word > speaker)."
    )

    # ── cross-speaker retrieval ──
    rng.shuffle(mrecs_idx)
    eval_idx = mrecs_idx[:EVAL_CAP]
    print(
        f"\n{'=' * 92}\nCROSS-SPEAKER SAME-WORD RETRIEVAL (eval={len(eval_idx)}, "
        f"gallery={len(mrecs_idx)}; gallery excludes query speaker)\n{'=' * 92}"
    )
    print(f"  {'representation':20} {'top1':>6} {'top5':>6} {'MRR':>6} {'chance':>7}")
    print("  " + "-" * 50)
    ret_raw = None
    for name, d in reps:
        r = retrieval(recs, d, eval_idx, mrecs_idx, rng)
        if name == "fp_whole (cos)":
            ret_raw = r
        print(f"  {name:20} {r['top1']:6.3f} {r['top5']:6.3f} {r['mrr']:6.3f} {r['chance']:7.3f}")

    # HuBERT-DTW (capped subset) — the §16 point
    dtw_idx = mrecs_idx[:DTW_CAP]
    hub = [np.asarray(recs[i]["hubert"], dtype=np.float32) for i in dtw_idx]
    hnorm = [h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12) for h in hub]
    pos = {gi: li for li, gi in enumerate(dtw_idx)}

    def d_hdtw(i, j):
        return dtw_frames(hnorm[pos[i]], hnorm[pos[j]])

    print(f"\n  HuBERT-DTW (raw-frame DTW, capped n={len(dtw_idx)}):")
    r = retrieval(recs, d_hdtw, dtw_idx[:120], dtw_idx, rng)
    print(
        f"  {'HuBERT-DTW':20} {r['top1']:6.3f} {r['top5']:6.3f} {r['mrr']:6.3f} {r['chance']:7.3f}"
    )
    sub_recs = [recs[i] for i in dtw_idx]
    fwd = four_way(sub_recs, lambda a, b: d_hdtw(dtw_idx[a], dtw_idx[b]), rng, n=600)
    ratio = (fwd["dw_ss"] / fwd["sw_ds"]) if (fwd["dw_ss"] and fwd["sw_ds"]) else float("nan")
    print(
        f"  HuBERT-DTW 4-way: sw/ds={fwd['sw_ds']:.3f} dw/ss={fwd['dw_ss']:.3f} ratio={ratio:.2f}"
    )

    # ── word vs speaker probes (fp_whole) ──
    print(f"\n{'=' * 92}\nWORD-vs-SPEAKER DISENTANGLEMENT (fp_whole probe)\n{'=' * 92}")
    pr = probe_disentangle(fpw, [r["word"] for r in recs], [r["speaker"] for r in recs], args.seed)
    print(f"  word probe (speaker-held-out macro-F1)       : {pr['word']}")
    print(
        f"  speaker probe (macro-F1, chance {pr['speaker_chance']:.3f})          : {pr['speaker']}"
    )
    print(f"  word probe after speaker-regression residual : {pr['word_after_speaker_regression']}")
    print("  'more universal' = high word / low speaker; residual = word signal left after")
    print("  removing the linear speaker component.")

    # ── speaker-centering on fp_whole ──
    print(f"\n{'=' * 92}\nSPEAKER-CENTERING intervention (fp_whole − per-speaker mean)\n{'=' * 92}")
    byspk = defaultdict(list)
    for i, r in enumerate(recs):
        byspk[r["speaker"]].append(i)
    spk_mean = {s: fpw[idxs].mean(axis=0) for s, idxs in byspk.items()}
    fpw_c = fpw - np.asarray([spk_mean[r["speaker"]] for r in recs], dtype=np.float32)
    fpw_c_n = fpw_c / (np.linalg.norm(fpw_c, axis=1, keepdims=True) + 1e-12)

    def d_cos_c(i, j):
        return float(1.0 - np.dot(fpw_c_n[i], fpw_c_n[j]))

    r_ctr = retrieval(recs, d_cos_c, eval_idx, mrecs_idx, rng)
    print(f"  fp_whole raw      top1={ret_raw['top1']:.3f}  MRR={ret_raw['mrr']:.3f}")
    print(
        f"  fp_whole centered top1={r_ctr['top1']:.3f}  MRR={r_ctr['mrr']:.3f}  "
        f"(Δtop1 {r_ctr['top1'] - ret_raw['top1']:+.3f})"
    )

    print("\n  Verdict: word structure dominates speaker voice where ratio>1 and cross-speaker")
    print("  top-1 ≫ chance. 'Universal' = cross-speaker word-associated acoustic structure, NOT")
    print("  validated meaning (semantic ceiling; tokens ≠ words). semantic_validation: none.")
    print("\nLS SPEAKER-INVARIANT DONE")


if __name__ == "__main__":
    main()
