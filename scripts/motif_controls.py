#!/usr/bin/env python3
"""Motif stability + specificity controls for sub-call sequences.

Five controls on the InfantMarmosetsVox sub-call motif result:
  A. Bootstrap stability: do top motifs recur across resampled subsets?
  B. Position-specific: are some clusters preferentially onset/body/offset?
  C. Caller-balanced null: are calltype motifs still significant after balancing callers?
  D. Within-calltype caller: are caller motifs significant within a single calltype?
  E. Bigram-preserving null: are trigrams meaningful beyond local transitions?

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
import hecb_oracle_marmoset as OM  # noqa: E402
import subcall_motifs as SM  # noqa: E402


def position_label(idx, seq_len):
    """Label a sub-call's position: first / middle / last."""
    if seq_len <= 1:
        return "only"
    if idx == 0:
        return "first"
    if idx == seq_len - 1:
        return "last"
    return "middle"


def position_enrichment(seqs, k_clusters):
    """For each cluster: is it enriched for a specific position (first/middle/last)?"""
    from scipy.stats import chi2_contingency  # noqa: PLC0415

    print(f"\n  {'cluster':>8} {'first%':>7} {'middle%':>8} {'last%':>7} {'V':>6} {'p':>10} {'role':>8}")
    for k in range(k_clusters):
        positions = []
        for s in seqs:
            for i, tok in enumerate(s["tokens"]):
                if tok == k:
                    positions.append(position_label(i, len(s["tokens"])))
        if len(positions) < 10:
            continue
        pos_counts = Counter(positions)
        total = len(positions)
        pct_first = pos_counts.get("first", 0) / total * 100
        pct_mid = pos_counts.get("middle", 0) / total * 100
        pct_last = pos_counts.get("last", 0) / total * 100
        # Expected: uniform across first/middle/last (rough)
        expected = total / 3
        table = np.array([
            [pos_counts.get("first", 0), pos_counts.get("middle", 0), pos_counts.get("last", 0)],
            [max(expected - pos_counts.get("first", 0), 0),
             max(expected - pos_counts.get("middle", 0), 0),
             max(expected - pos_counts.get("last", 0), 0)],
        ])
        try:
            chi2, p, _, _ = chi2_contingency(table)
            V = np.sqrt(chi2 / (total * 1))  # min(2,3)-1 = 1
        except Exception:  # noqa: BLE001
            V, p = 0.0, 1.0
        role = "onset" if pct_first > 40 else ("offset" if pct_last > 40 else "body" if pct_mid > 50 else "—")
        print(f"  {k:>8} {pct_first:7.1f} {pct_mid:8.1f} {pct_last:7.1f} {V:6.3f} {p:10.4f} {role:>8}")


def bootstrap_stability(seqs, n=2, top_k=10, n_boot=20, seed=0):
    """Resample vocalisations, mine top-n motifs each time, measure Jaccard overlap."""
    rng = random.Random(seed)
    top_sets = []
    for _ in range(n_boot):
        sample = rng.choices(seqs, k=len(seqs))
        counts = SM.mine_ngrams(sample, n)
        top = set(m for m, _ in counts.most_common(top_k))
        top_sets.append(top)
    # Mean pairwise Jaccard
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = top_sets[i] | top_sets[j]
            if union:
                jaccards.append(len(top_sets[i] & top_sets[j]) / len(union))
    return float(np.mean(jaccards)) if jaccards else 0.0


def caller_balanced_null(seqs, n=2, seed=0):
    """Subsample to balance callers per calltype, recompute motif enrichment."""
    rng = random.Random(seed)
    # Group by (calltype, caller)
    by_cc = defaultdict(list)
    for s in seqs:
        by_cc[(s.get("ct"), s.get("caller"))].append(s)
    # For each calltype, subsample equal n per caller
    by_ct = defaultdict(list)
    for (ct, caller), group in by_cc.items():
        by_ct[ct].extend(group)
    min_per_caller = min(len(g) for g in by_cc.values()) if by_cc else 1
    balanced = []
    for ct, group in by_ct.items():
        callers = defaultdict(list)
        for s in group:
            callers[s.get("caller")].append(s)
        for caller, cg in callers.items():
            balanced.extend(rng.sample(cg, min(len(cg), min_per_caller)))
    counts = SM.mine_ngrams(balanced, n)
    return counts


def bigram_preserving_null(seqs, n=3, n_perm=100, seed=0):
    """Shuffle trigrams preserving bigram transitions: swap non-overlapping bigram pairs."""
    rng = random.Random(seed)
    null_counts = []
    for _ in range(n_perm):
        shuffled = []
        for s in seqs:
            toks = list(s["tokens"])
            # Swap adjacent bigrams (preserve local transitions)
            if len(toks) > 3:
                for i in range(0, len(toks) - 3, 2):
                    if rng.random() < 0.5:
                        toks[i + 1], toks[i + 2] = toks[i + 2], toks[i + 1]
            shuffled.append({"tokens": toks, "label": s["label"], "n": len(toks),
                             "ct": s.get("ct"), "caller": s.get("caller")})
        null_counts.append(len(SM.mine_ngrams(shuffled, n)))
    return null_counts


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Motif stability + specificity controls")
    ap.add_argument("--n-train-clf", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--k-clusters", type=int, default=12)
    args = ap.parse_args()

    print("loading GT calls + building oracle vocs...", flush=True)
    all_calls = OM.load_all_calls(n_files=999)
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="marmoset").combined  # noqa: E731
    rng = random.Random(1)
    pool = list(all_calls)
    rng.shuffle(pool)
    half = len(pool) // 2
    clf = OM.train_typed_clf(pool[:half], n_train=args.n_train_clf)
    reps, ct_labels, caller_labels = OM.oracle_call_events(pool[half:][:args.n_eval], clf, fp_fn)
    print(f"  {len(reps)} vocs", flush=True)

    # Tokenize
    km, _tok_fn = SM.tokenize_subcalls(reps, k=args.k_clusters)
    if km is None:
        print("ERROR: too few sub-calls")
        sys.exit(1)

    # Build sequences with both labels
    seqs = []
    for r, ct, caller in zip(reps, ct_labels, caller_labels):
        if not r.subcall_fingerprints:
            continue
        toks = _tok_fn(r.subcall_fingerprints)
        seqs.append({"tokens": toks, "ct": ct, "caller": caller, "n": len(toks),
                     "label": ct})  # label=ct for enrichment

    print(f"  {len(seqs)} sequences, mean length {np.mean([s['n'] for s in seqs]):.1f}", flush=True)

    # ─── A. Bootstrap stability ───
    print(f"\n{'='*70}")
    print("CONTROL A: Bootstrap stability (bigrams)")
    print(f"{'='*70}")
    for n in [2, 3]:
        jaccard = bootstrap_stability(seqs, n=n, top_k=10, n_boot=20)
        print(f"  {n}-gram top-10 Jaccard: {jaccard:.3f} ({'STABLE' if jaccard > 0.5 else 'UNSTABLE' if jaccard < 0.3 else 'MODERATE'})")

    # ─── B. Position-specific enrichment ───
    print(f"\n{'='*70}")
    print("CONTROL B: Position-specific enrichment (onset/body/offset roles)")
    print(f"{'='*70}")
    position_enrichment(seqs, args.k_clusters)

    # ─── C. Caller-balanced calltype null ───
    print(f"\n{'='*70}")
    print("CONTROL C: Caller-balanced calltype motifs (bigrams)")
    print(f"{'='*70}")
    balanced_counts = caller_balanced_null(seqs, n=2)
    print(f"  Top 5 bigrams after caller-balancing:")
    for motif, cnt in balanced_counts.most_common(5):
        print(f"    {str(motif):20} count={cnt}")

    # ─── D. Within-calltype caller test ───
    print(f"\n{'='*70}")
    print("CONTROL D: Within-calltype caller motifs (Phee bigrams)")
    print(f"{'='*70}")
    phee_seqs = [s for s in seqs if s["ct"] == 1]  # Phee
    if len(phee_seqs) >= 100:
        phee_caller_seqs = [{"tokens": s["tokens"], "label": s["caller"], "n": s["n"]} for s in phee_seqs]
        counts = SM.mine_ngrams(phee_caller_seqs, 2)
        n_callers = len(set(s["caller"] for s in phee_seqs))
        print(f"  Phee: {len(phee_seqs)} vocs, {n_callers} callers")
        print(f"  Top 5 bigrams + caller enrichment:")
        for motif, cnt in counts.most_common(5):
            V, p = SM.motif_enrichment(phee_caller_seqs, motif, 2)
            print(f"    {str(motif):20} count={cnt:4d}  V={V:.3f}  p={p:.4f}")

    # ─── E. Bigram-preserving trigram null ───
    print(f"\n{'='*70}")
    print("CONTROL E: Bigram-preserving trigram null")
    print(f"{'='*70}")
    obs_tri = len(SM.mine_ngrams(seqs, 3))
    null_tri = bigram_preserving_null(seqs, n=3, n_perm=100)
    null_mean = np.mean(null_tri)
    pct = np.searchsorted(np.sort(null_tri), obs_tri) / len(null_tri) * 100
    print(f"  observed distinct trigrams: {obs_tri}")
    print(f"  null (bigram-preserving):    {null_mean:.0f} ± {np.std(null_tri):.0f}")
    print(f"  percentile:                  {pct:.0f}%")
    print(f"  → {'trigrams carry signal beyond bigrams' if pct < 5 or pct > 95 else 'within null'}")

    print("\nMOTIF CONTROLS DONE")


if __name__ == "__main__":
    main()
