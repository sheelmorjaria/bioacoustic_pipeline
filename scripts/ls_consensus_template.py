#!/usr/bin/env python3
"""LibriSpeech DTW-barycenter consensus word template (step #3 of the §25 follow-on).

§25 found linear speaker-centering FAILS (Δ −0.003) — the speaker/word entanglement is
non-additive. The principled successor is a SEQUENCE-space average: a DTW Barycenter
Average (DBA; Petitjean et al.) per word across speakers, then nearest-template retrieval.

Each word instance = its fp-window trajectory (list of 43-D vectors). For each frequent
multi-speaker word, build a DBA consensus template (medoid-init, iterated DTW averaging).
Then LEAVE-ONE-INSTANCE-OUT nearest-template retrieval: query instance → distance to each
word template (its own word's template rebuilt without it); top-1 = correct word?

Compare to raw cross-speaker NN (§25-style, same word set). If the template beats raw NN,
the consensus structure is a better speaker-invariant code than pointwise distances; if not,
the entanglement defeats even sequence-space averaging.

NO re-extraction: loads outputs/ls_unified/recs.pkl. Pure CPU. Human-speech word GT honest
here; "universal" = cross-speaker word-associated structure, NOT meaning (semantic ceiling;
tokens ≠ words). semantic_validation stays none. fp is human-only (all-species view but
computed on human audio).

Run:  python scripts/ls_consensus_template.py

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

RECS = ROOT / "outputs" / "ls_unified" / "recs.pkl"
TOP_WORDS = 25
EVAL_CAP = 250
MIN_SPEAKERS = 3
DBA_ITER = 5


def _norm_rows(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def traj(r, rep):
    """Word trajectory as a sequence of vectors: rep='cpc' → CPC frames (Tc,256);
    rep='fp' → fp-window vectors (list of 43-D). [] if missing/empty."""
    v = r.get(rep)
    if v is None or len(v) == 0:
        return []
    return v


def dtw_full(a, b):
    """Length-normalised DTW (cosine cost) + alignment path. a,b: (T,D) arrays."""
    an = _norm_rows(np.asarray(a, dtype=np.float32))
    bn = _norm_rows(np.asarray(b, dtype=np.float32))
    m, n = len(an), len(bn)
    if m == 0 or n == 0:
        return 1.0, []
    inf = float("inf")
    C = [[inf] * (n + 1) for _ in range(m + 1)]
    C[0][0] = 0.0
    for i in range(1, m + 1):
        ai = an[i - 1]
        Ci = C[i]
        Ci1 = C[i - 1]
        for j in range(1, n + 1):
            cost = max(0.0, 1.0 - float(np.dot(ai, bn[j - 1])))
            Ci[j] = cost + min(Ci1[j], Ci[j - 1], Ci1[j - 1])
    # backtrace
    path = []
    i, j = m, n
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        dd, uu, le = C[i - 1][j - 1], C[i - 1][j], C[i][j - 1]
        if dd <= uu and dd <= le:
            i, j = i - 1, j - 1
        elif uu <= le:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return C[m][n] / max(m, n), path


def dtw_cost(a, b):
    return dtw_full(a, b)[0]


def dba(seqs, n_iter=DBA_ITER):
    """Petitjean DTW barycenter average: medoid-init, iterated DTW-aligned averaging."""
    seqs = [s for s in seqs if len(s) >= 2]
    if not seqs:
        return None
    if len(seqs) == 1:
        return [np.asarray(v, dtype=np.float32) for v in seqs[0]]
    # medoid init = seq with min total DTW cost to others
    sums = [sum(dtw_cost(s, t) for t in seqs) for s in seqs]
    ref = [np.asarray(v, dtype=np.float32) for v in seqs[int(np.argmin(sums))]]
    for _ in range(n_iter):
        accum = [[] for _ in range(len(ref))]
        for s in seqs:
            _, path = dtw_full(ref, s)
            sa = [np.asarray(v, dtype=np.float32) for v in s]
            for i, j in path:
                accum[i].append(sa[j])
        new_ref = [np.mean(vs, axis=0) for vs in accum if vs]
        ref = new_ref if new_ref else ref
    return ref


def retrieval_nn(query_idx, words, by_word, fp_seq, qs):
    """Raw cross-speaker NN: nearest instance by OTHER speakers; top-1 same word?"""
    gal = [(gi, w) for w in words for gi in by_word[w] if fp_seq["speaker"][gi] != qs]
    if len(gal) < 5:
        return None
    q = fp_seq["fp"][query_idx]
    gal.sort(key=lambda gw: dtw_cost(q, fp_seq["fp"][gw[0]]))
    return 1 if gal[0][1] == fp_seq["word"][query_idx] else 0


def retrieval_template(query_idx, words, by_word, fp_seq, templates, qs):
    """Leave-one-out nearest-template: own word template rebuilt without the query."""
    q = fp_seq["fp"][query_idx]
    qw = fp_seq["word"][query_idx]
    best_d, best_w = 2.0, None
    for w in words:
        if w == qw:
            tmpl = dba([fp_seq["fp"][gi] for gi in by_word[w] if gi != query_idx])
        else:
            tmpl = templates.get(w)
        if tmpl is None:
            continue
        d = dtw_cost(q, tmpl)
        if d < best_d:
            best_d, best_w = d, w
    return 1 if best_w == qw else 0


def main():
    ap = argparse.ArgumentParser(description="LibriSpeech DTW-barycenter consensus word template")
    ap.add_argument("--recs", type=str, default=str(RECS))
    ap.add_argument(
        "--rep",
        choices=("fp", "cpc", "hubert", "aves"),
        default="fp",
        help="trajectory representation: 'fp', 'cpc', 'hubert', or 'aves' "
        "(768-D AVES frames — multi-taxa SSL, firewall-legal for animals).",
    )
    ap.add_argument(
        "--aves-recs",
        type=str,
        default="",
        help="path to aves_recs.pkl (required for --rep aves; injected into recs as r['aves'])",
    )
    ap.add_argument("--top-words", type=int, default=TOP_WORDS)
    ap.add_argument("--min-speakers", type=int, default=MIN_SPEAKERS)
    ap.add_argument("--eval-cap", type=int, default=EVAL_CAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.recs, "rb") as f:
        recs = pickle.load(f)  # noqa: S301
    if args.aves_recs:
        with open(args.aves_recs, "rb") as f:
            aves_recs = pickle.load(f)  # noqa: S301
        for i, r in enumerate(recs):
            r["aves"] = aves_recs[i] if i < len(aves_recs) else None
        print(f"  injected AVES frames from {args.aves_recs}", flush=True)
    by_word = defaultdict(list)
    by_word_spk = defaultdict(set)
    for i, r in enumerate(recs):
        if len(traj(r, args.rep)) >= 2:  # need a trajectory in the chosen rep
            by_word[r["word"]].append(i)
            by_word_spk[r["word"]].add(r["speaker"])
    cand = sorted(
        (w for w in by_word if len(by_word_spk[w]) >= args.min_speakers),
        key=lambda w: -len(by_word[w]),
    )
    words = cand[: args.top_words]
    print(
        f"loaded {len(recs)} instances | rep={args.rep} | {len(words)} frequent multi-speaker "
        f"words (≥{args.min_speakers} speakers) | "
        f"instances in scope: {sum(len(by_word[w]) for w in words)}",
        flush=True,
    )

    fp_seq = {
        "fp": [traj(recs[i], args.rep) for i in range(len(recs))],  # rep-keyed trajectory
        "word": [recs[i]["word"] for i in range(len(recs))],
        "speaker": [recs[i]["speaker"] for i in range(len(recs))],
    }

    # build consensus templates (all instances per word)
    print(f"building DBA consensus templates ({DBA_ITER} iters)...", flush=True)
    templates = {}
    for w in words:
        templates[w] = dba([fp_seq["fp"][gi] for gi in by_word[w]])
    print("  done", flush=True)

    # eval instances (capped), restricted to the chosen words
    eval_pool = [gi for w in words for gi in by_word[w]]
    rng.shuffle(eval_pool)
    eval_idx = eval_pool[: args.eval_cap]

    print(f"\nretrieval (n_eval={len(eval_idx)}, {len(words)} word templates)...", flush=True)
    nn_hits = nn_n = 0
    tpl_hits = tpl_n = 0
    for k, q in enumerate(eval_idx):
        qs = fp_seq["speaker"][q]
        r_nn = retrieval_nn(q, words, by_word, fp_seq, qs)
        if r_nn is not None:
            nn_hits += r_nn
            nn_n += 1
        r_tpl = retrieval_template(q, words, by_word, fp_seq, templates, qs)
        tpl_hits += r_tpl
        tpl_n += 1
        if (k + 1) % 50 == 0:
            print(
                f"  {k + 1}/{len(eval_idx)}  NN top1={nn_hits / max(nn_n, 1):.3f}  "
                f"template top1={tpl_hits / max(tpl_n, 1):.3f}",
                flush=True,
            )

    chance = 1.0 / len(words)
    print(
        f"\n{'=' * 80}\nCONSENSUS TEMPLATE vs RAW CROSS-SPEAKER NN (n_eval={tpl_n}, "
        f"{len(words)} words, chance={chance:.3f})\n{'=' * 80}"
    )
    print(f"  raw cross-speaker NN top-1   : {nn_hits / max(nn_n, 1):.3f}  (n={nn_n})")
    print(f"  DBA consensus template top-1 : {tpl_hits / max(tpl_n, 1):.3f}  (n={tpl_n})")
    print(f"  chance (1/#words)            : {chance:.3f}")
    delta = tpl_hits / max(tpl_n, 1) - nn_hits / max(nn_n, 1)
    print(f"\n  Δ(template − NN) = {delta:+.3f}")
    if delta > 0.01:
        print("  → DBA consensus template BEATS raw NN: sequence-space averaging extracts a")
        print("    speaker-invariant word structure that pointwise cross-speaker distances miss.")
    elif delta < -0.01:
        print("  → DBA template does NOT beat raw NN: the speaker entanglement (§25) defeats")
        print("    even DTW-barycenter averaging — there is no simple consensus 'universal word'.")
    else:
        print("  → template ≈ raw NN: consensus averaging neither helps nor hurts.")
    print("\n  Template = Petitjean DTW barycenter (medoid init, 5 iters) over fp-window")
    print("  trajectories; leave-one-instance-out nearest-template. 'Universal' = cross-speaker")
    print("  word-associated structure, NOT meaning (semantic ceiling; semantic_validation none).")
    print("\nLS CONSENSUS TEMPLATE DONE")


if __name__ == "__main__":
    main()
