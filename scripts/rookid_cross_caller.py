#!/usr/bin/env python3
"""RookID cross-CALLER same-call retrieval — the cross-species analogue of §25.

§25 (LibriSpeech) asked: does word structure survive across speakers
(`dist(same-word, diff-speaker) < dist(diff-word, same-speaker)` + cross-speaker retrieval)?
This asks the same of RookID: does CALL-TYPE structure survive across CALLERS (rooks)?

  query  = a call by rook A
  gallery = calls by all OTHER rooks (gallery excludes A)
  metric  = top-1/5/MRR same-call-type retrieval accuracy

Targets (call-type labels): activity (sing/nest_building/call), sing/non-sing, song_variant
(sing-prefix), Event. Caller = Source (the rook). Representations: whole-call fingerprint
(cosine — the §25 fp_whole row analogue) + sub-call cluster sequence (Levenshtein).

Hypothesis (from §18/§23: individual-ID strong, context near-chance): cross-caller retrieval
of acoustic CATEGORY (sing vs call) should be high (it's an acoustic distinction); song_variant
and within-song structure weaker; and — mirroring §25 — the raw distance ratio will sit ≈1
because the caller signature is deeply entangled with call form (the mechanistic link to the
high bat Emitter V).

Reuses rookid_event_comment (parse_enriched / build_recs / kmeans_tokens) + the §25 four-way /
retrieval logic. Whole-call fp + raw NBD only (all-species, firewall-clean; no learned tokens).
Retrieval is acoustic/structural association, NOT validated meaning — `semantic_validation`
stays `none`.

Run:  python scripts/rookid_cross_caller.py --max-calls 3000

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
sys.path.insert(0, str(ROOT / "scripts"))

import rookid_event_comment as RE  # noqa: E402

K_SUB = 24
N_PAIRS = 2000
EVAL_CAP = 400


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


def sample_pairs(recs, label_key, caller_key, predicate, n, rng):
    idx = list(range(len(recs)))
    out, tries = [], 0
    while len(out) < n and tries < n * 40:
        i, j = rng.choice(idx), rng.choice(idx)
        tries += 1
        if i != j and predicate(i, j, recs, label_key, caller_key):
            out.append((i, j))
    return out


def four_way(recs, dist, label_key, caller_key, rng, n=N_PAIRS):
    def same_label_diff_caller(a, b, r, lk, ck):
        return r[a][lk] == r[b][lk] and r[a][ck] != r[b][ck]

    def diff_label_same_caller(a, b, r, lk, ck):
        return r[a][lk] != r[b][lk] and r[a][ck] == r[b][ck]

    def diff_diff(a, b, r, lk, ck):
        return r[a][lk] != r[b][lk] and r[a][ck] != r[b][ck]

    def same_same(a, b, r, lk, ck):
        return r[a][lk] == r[b][lk] and r[a][ck] == r[b][ck]

    def md(pred):
        pairs = sample_pairs(recs, label_key, caller_key, pred, n, rng)
        return float(np.mean([dist(a, b) for a, b in pairs])) if pairs else None

    return {
        "same_diff": md(same_label_diff_caller),
        "diff_same": md(diff_label_same_caller),
        "diff_diff": md(diff_diff),
        "same_same": md(same_same),
    }


def retrieval(recs, dist, label_key, caller_key, eval_idx, gallery_idx, rng):
    """Cross-caller same-label retrieval: gallery excludes the query's caller."""
    glabels = [recs[i][label_key] for i in gallery_idx]
    gcallers = [recs[i][caller_key] for i in gallery_idx]
    chance = sum((glabels.count(lab) / len(glabels)) ** 2 for lab in set(glabels))
    top1 = top5 = mrr = 0.0
    neval = 0
    for q in eval_idx:
        qc, ql = recs[q][caller_key], recs[q][label_key]
        gal = [g for k, g in enumerate(gallery_idx) if gcallers[k] != qc]
        if len(gal) < 5:
            continue
        ds = sorted(((dist(q, g), recs[g][label_key]) for g in gal), key=lambda x: x[0])
        ranks = [k for k, (_, lab) in enumerate(ds) if lab == ql]
        neval += 1
        if ranks:
            r0 = ranks[0]
            top1 += r0 == 0
            top5 += r0 < 5
            mrr += 1.0 / (r0 + 1)
    neval = max(neval, 1)
    return dict(top1=top1 / neval, top5=top5 / neval, mrr=mrr / neval, chance=chance, n=neval)


def main():
    ap = argparse.ArgumentParser(
        description="RookID cross-caller same-call retrieval (§25 analogue)"
    )
    ap.add_argument("--max-calls", type=int, default=3000)
    ap.add_argument("--k-sub", type=int, default=K_SUB)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cache", type=str, default="", help="reuse a pickled recs file instead of re-extracting"
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.cache and Path(args.cache).exists():
        with open(args.cache, "rb") as f:
            recs = pickle.load(f)  # noqa: S301
        print(f"loaded {len(recs)} recs from cache {args.cache}", flush=True)
    else:
        print("parsing RookID TSVs + building whole-call fp + NBD sub-call units...", flush=True)
        calls = RE.parse_enriched(args.max_calls)
        recs = RE.build_recs(calls)
        recs = [r for r in recs if len(r["sub_fps"]) >= 2]  # need sub-units for the motif/seq rep
        with open(ROOT / "outputs" / "rookid_crosscaller_recs.pkl", "wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {len(recs)} calls with fp + ≥2 sub-units (cached for re-scoring)", flush=True)

    # multi-caller label sets per target (≥2 callers spoken the label)
    def multicaller(label_key):
        by = defaultdict(set)
        for r in recs:
            by[r[label_key]].add(r["source"])
        return {lab for lab, cs in by.items() if len(cs) >= 2}

    targets = [
        ("activity", "activity"),
        ("sing/non-sing", "_sing_lbl"),
        ("song_variant", "song_variant"),
        ("Event", "event"),
    ]
    # add a convenience binary sing label
    for r in recs:
        r["_sing_lbl"] = "sing" if r["is_sing"] else "call"

    # ── representations ──
    fpw = np.asarray([r["whole"] for r in recs], dtype=np.float32)
    fpw_n = fpw / (np.linalg.norm(fpw, axis=1, keepdims=True) + 1e-12)

    def d_cos(i, j):
        return float(1.0 - np.dot(fpw_n[i], fpw_n[j]))

    # sub-call cluster sequences: flatten all sub-unit fps → k-means → re-chunk per rec
    all_sub = [np.asarray(fp, np.float32) for r in recs for fp in r["sub_fps"]]
    sub_tok = RE.kmeans_tokens(all_sub, args.k_sub, args.seed)
    idx = 0
    for r in recs:
        n = len(r["sub_fps"])
        if sub_tok is None:
            r["_sub_seq"] = []
        else:
            r["_sub_seq"] = collapse_repeats(sub_tok[idx : idx + n].tolist())
        idx += n

    def d_seq(i, j):
        a, b = recs[i]["_sub_seq"], recs[j]["_sub_seq"]
        return edit_dist_norm(a, b) if a and b else 1.0

    reps = [("fp_whole (cos)", d_cos), ("sub-call-seq (Levenshtein)", d_seq)]

    for tname, tkey in targets:
        valid = multicaller(tkey)
        if len(valid) < 2:
            print(f"\n[{tname}] fewer than 2 multi-caller labels — skipping")
            continue
        sub = [i for i, r in enumerate(recs) if r[tkey] in valid]
        if len(sub) < 50:
            print(f"\n[{tname}] too few multi-caller instances ({len(sub)}) — skipping")
            continue
        sub_recs = [recs[i] for i in sub]
        print(
            f"\n{'=' * 88}\n[{tname}] n={len(sub)} multi-caller instances, "
            f"{len(valid)} labels, {len({recs[i]['source'] for i in sub})} callers"
        )
        print(
            f"  {'representation':26} {'s-lbl/d-caller':>14} {'d-lbl/s-caller':>14} {'ratio':>6} "
            f"| {'top1':>6} {'top5':>6} {'MRR':>6} {'chance':>7}"
        )
        print("  " + "-" * 92)
        rng.shuffle(sub)
        eval_idx = sub[: min(EVAL_CAP, len(sub))]
        for rname, d in reps:

            def dd(a, b, _d=d):
                return _d(sub[a], sub[b])

            fw = four_way(sub_recs, dd, tkey, "source", rng)
            ratio = (
                (fw["diff_same"] / fw["same_diff"])
                if (fw["diff_same"] and fw["same_diff"])
                else float("nan")
            )
            ret = retrieval(recs, d, tkey, "source", eval_idx, sub, rng)
            print(
                f"  {rname:26} {fw['same_diff']:14.3f} {fw['diff_same']:14.3f} {ratio:6.2f} "
                f"| {ret['top1']:6.3f} {ret['top5']:6.3f} {ret['mrr']:6.3f} {ret['chance']:7.3f}"
            )

    print(
        "\n  Read: ratio = d(diff-label,same-caller) ÷ d(same-label,diff-caller) "
        "(>1 ⇒ call-type > caller; §25 sat at ~0.86–1.00)."
    )
    print("  Cross-caller top-1 ≫ chance ⇒ species-level call structure survives across rooks.")
    print("  Acoustic/structural association, NOT meaning — semantic_validation stays none.")
    print("\nROOKID CROSS-CALLER DONE")


if __name__ == "__main__":
    main()
