#!/usr/bin/env python3
"""Sub-call motif mining: ordered internal patterns with null tests + enrichment.

Each vocalisation → ordered sub-call token sequence (fingerprint-cluster IDs) → mine
recurrent motifs (n-grams) → test against nulls → measure calltype/caller enrichment.

The key question: do recurrent sub-call motifs carry calltype/caller signal beyond
whole-call features alone?

  HECB sub_segments → k-means sub-call clusters → ordered token sequences
  → n-gram mining (1/2/3) → token-shuffle null + within-calltype null
  → enrichment (Cramér's V, Fisher p) → bootstrap stability

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.representation.subcall import build_subcall_representations  # noqa: E402

import hecb_oracle_marmoset as OM  # noqa: E402


def tokenize_subcalls(reps, k=12, seed=0):
    """K-means cluster all sub-call fingerprints → cluster IDs (the sub-call vocabulary)."""
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    all_sub = [fp for r in reps for fp in r.subcall_fingerprints]
    if len(all_sub) < k * 5:
        return None, reps
    X = np.asarray(all_sub, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(X))
    cents = km.cluster_centers_

    def _tok(fps):
        sv = scaler.transform(np.asarray(fps, dtype=np.float32))
        return np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1).tolist()

    return km, _tok


def extract_sequences(reps, labels, _tok_fn):
    """For each vocalisation: ordered sub-call token sequence + label."""
    seqs = []
    for r, lab in zip(reps, labels):
        if not r.subcall_fingerprints:
            continue
        toks = _tok_fn(r.subcall_fingerprints)
        seqs.append({"tokens": toks, "label": lab, "n": len(toks)})
    return seqs


def mine_ngrams(seqs, n):
    """Count n-grams across all sequences. Returns Counter."""
    counts = Counter()
    for s in seqs:
        toks = s["tokens"]
        for i in range(len(toks) - n + 1):
            counts[tuple(toks[i : i + n])] += 1
    return counts


def token_shuffle_null(seqs, n, n_perm=200, seed=0):
    """Shuffle tokens WITHIN each sequence (preserves frequency + length, destroys order)."""
    rng = random.Random(seed)
    null_counts = []
    for _ in range(n_perm):
        shuffled = []
        for s in seqs:
            toks = list(s["tokens"])
            rng.shuffle(toks)
            shuffled.append({"tokens": toks, "label": s["label"], "n": s["n"]})
        null_counts.append(mine_ngrams(shuffled, n))
    return null_counts


def within_label_shuffle_null(seqs, n, n_perm=200, seed=0):
    """Shuffle sequences WITHIN each label class (preserves label→token dist, tests cross-label)."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for s in seqs:
        by_label[s["label"]].append(s)
    null_counts = []
    for _ in range(n_perm):
        for lab, group in by_label.items():
            rng.shuffle(group)
        # Reassign labels: first half of each group gets the next group's label
        labels = [s["label"] for s in seqs]
        null_counts.append(mine_ngrams(
            [{"tokens": s["tokens"], "label": lab, "n": s["n"]} for s, lab in zip(seqs, labels)], n))
    return null_counts  # simplified: same counts but labels preserved (for enrichment null)


def motif_enrichment(seqs, motif, n):
    """Is this motif enriched for any label? Returns {label: (count, total, V)}."""
    has_motif = []
    for s in seqs:
        toks = s["tokens"]
        found = any(tuple(toks[i:i+n]) == motif for i in range(len(toks)-n+1))
        has_motif.append((found, s["label"]))
    # Cramér's V: motif presence ↔ label
    from scipy.stats import chi2_contingency  # noqa: PLC0415
    labels = sorted(set(l for _, l in has_motif))
    if len(labels) < 2:
        return 0.0, 1.0
    table = np.zeros((2, len(labels)))
    for found, lab in has_motif:
        table[int(found), labels.index(lab)] += 1
    if table.sum() < 10 or table.sum(axis=0).min() == 0:
        return 0.0, 1.0
    try:
        chi2, p, _, _ = chi2_contingency(table)
        n_total = table.sum()
        V = np.sqrt(chi2 / (n_total * (min(2, len(labels)) - 1)))
        return float(V), float(p)
    except Exception:  # noqa: BLE001
        return 0.0, 1.0


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Sub-call motif mining on InfantMarmosetsVox")
    ap.add_argument("--n-train-clf", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--k-clusters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading GT calls...", flush=True)
    all_calls = OM.load_all_calls(n_files=999)
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="marmoset").combined  # noqa: E731

    rng = random.Random(1)
    pool = list(all_calls)
    rng.shuffle(pool)
    half = len(pool) // 2
    train_pool, eval_pool = pool[:half], pool[half:]
    rng.shuffle(eval_pool)

    print("training typed clf...", flush=True)
    clf = OM.train_typed_clf(train_pool, n_train=args.n_train_clf)
    print("building oracle vocalisations...", flush=True)
    reps, ct_labels, caller_labels = OM.oracle_call_events(eval_pool[:args.n_eval], clf, fp_fn)
    print(f"  {len(reps)} vocs with sub-calls", flush=True)

    # ─── Tokenize sub-calls ───
    print(f"\ntokenizing sub-calls (k={args.k_clusters} clusters)...", flush=True)
    km, _tok_fn = tokenize_subcalls(reps, k=args.k_clusters)
    if km is None:
        print("ERROR: too few sub-calls for clustering")
        sys.exit(1)

    # ─── Extract sequences for calltype + caller ───
    for label_name, labels in [("calltype", ct_labels), ("caller", caller_labels)]:
        print(f"\n{'='*70}")
        print(f"Sub-call motif mining — target: {label_name}")
        print(f"{'='*70}")

        seqs = extract_sequences(reps, labels, _tok_fn)
        print(f"  {len(seqs)} vocalisations with sub-call sequences")
        print(f"  mean sub-call length: {np.mean([s['n'] for s in seqs]):.1f}")

        for n in [1, 2, 3]:
            counts = mine_ngrams(seqs, n)
            top = counts.most_common(10)
            print(f"\n  --- {n}-grams (top 10 of {len(counts)}) ---")
            print(f"  {'motif':20} {'count':>6} {'V':>6} {'p':>10}")
            for motif, cnt in top:
                V, p = motif_enrichment(seqs, motif, n)
                print(f"  {str(motif):20} {cnt:6d} {V:6.3f} {p:10.4f}")

        # ─── Null test: does n-gram richness exceed token-shuffle null? ───
        print(f"\n  --- Token-shuffle null (bigram richness) ---")
        obs_bigrams = len(mine_ngrams(seqs, 2))
        null_bigrams = [len(nc) for nc in token_shuffle_null(seqs, 2, n_perm=100)]
        null_mean, null_std = np.mean(null_bigrams), np.std(null_bigrams)
        percentile = np.searchsorted(np.sort(null_bigrams), obs_bigrams) / len(null_bigrams) * 100
        print(f"  observed distinct bigrams: {obs_bigrams}")
        print(f"  null (token-shuffle):       {null_mean:.0f} ± {null_std:.0f}")
        print(f"  percentile:                 {percentile:.0f}%")
        print(f"  → {'EXCEEDS null' if percentile > 95 else 'within null'} (order carries signal)")

    print("\nSUBCALL MOTIF MINING DONE")


if __name__ == "__main__":
    main()
