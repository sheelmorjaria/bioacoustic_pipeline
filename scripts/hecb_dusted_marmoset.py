#!/usr/bin/env python3
"""HECB-DUSTED on InfantMarmosetsVox: 3-way recurrent-acoustic-term discovery.

Compares three subsequence-matching primitives on the SAME HECB-typed sub-call unit
stream (oracle GT spans → HECB typed internal segments → sub-call fingerprints):

    exact n-grams  |  fixed-length cosine  |  DTW with indels (DUSTED)

The discovery target is RECURRENT ACOUSTIC TERMS / MOTIFS — NOT words. Calltype and
caller enrichment are acoustic/structural association, not semantics (semantic_validation
stays `none` until Track-C playback/context evidence).

Expected trend (the decisive question): exact n-grams fragment every substituted/dropped/
inserted variant; fixed-length cosine catches substitutions but misses indels; DTW recovers
variant families across lengths → richest motif inventory and best coverage.

Metrics per method: #motifs, total support, coverage, mean calltype Cramér's V (+ fraction
significant), mean caller V, plus (DTW only) order-shuffle null percentile + bootstrap Jaccard.

Run:  python scripts/hecb_dusted_marmoset.py --n-eval 1500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402

import hecb_oracle_marmoset as OM  # noqa: E402
import subcall_motifs as SM  # noqa: E402


def build_call_units(reps, ct_labels, caller_labels, tok_fn):
    """SubcallRepresentations + labels → CallUnits (fingerprints + cluster IDs)."""
    calls = []
    for i, (r, ct, caller) in enumerate(zip(reps, ct_labels, caller_labels)):
        if not r.subcall_fingerprints:
            continue
        clusters = tok_fn(r.subcall_fingerprints)
        calls.append(
            sm.CallUnits(
                fingerprints=[np.asarray(f, dtype=np.float32) for f in r.subcall_fingerprints],
                clusters=list(clusters),
                calltype=str(ct),
                caller=str(caller),
                file_id=r.call_id,
                call_idx=i,
            )
        )
    return calls


def summarize(name, motifs, n_calls):
    s = sm.summarize_discovery(motifs, n_calls)
    print(
        f"  {name:22} motifs={s['n_motifs']:4d}  support={s['total_support']:5d}  "
        f"cov={s['coverage']:.3f}  ct_V={s['mean_calltype_V']:.3f} "
        f"(sig {s['frac_calltype_sig']:.2f})  ca_V={s['mean_caller_V']:.3f}"
    )
    return s


def main():
    ap = argparse.ArgumentParser(
        description="HECB-DUSTED 3-way motif discovery (InfantMarmosetsVox)"
    )
    ap.add_argument("--n-train-clf", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=1500, help="calls for the 3-way comparison")
    ap.add_argument("--n-null", type=int, default=600, help="calls for the null/bootstrap subset")
    ap.add_argument("--k-clusters", type=int, default=12)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument(
        "--max-n", type=int, default=4, help="window length cap (4 = tractable; 6 = full)"
    )
    ap.add_argument("--dist-threshold", type=float, default=0.35)
    ap.add_argument("--centroid-radius", type=float, default=0.22)
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading GT calls + training HECB typed classifier...", flush=True)
    all_calls = OM.load_all_calls(n_files=999)
    rng = random.Random(1)
    pool = list(all_calls)
    rng.shuffle(pool)
    half = len(pool) // 2
    clf = OM.train_typed_clf(pool[:half], n_train=args.n_train_clf)
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="marmoset").combined  # noqa: E731

    print(f"building oracle HECB sub-call units on {args.n_eval} eval calls...", flush=True)
    reps, ct_labels, caller_labels = OM.oracle_call_events(pool[half:][: args.n_eval], clf, fp_fn)
    print(f"  {len(reps)} vocalisations with sub-calls", flush=True)

    km, tok_fn = SM.tokenize_subcalls(reps, k=args.k_clusters)
    if km is None:
        print("ERROR: too few sub-calls for k-means tokenization")
        sys.exit(1)

    calls = build_call_units(reps, ct_labels, caller_labels, tok_fn)
    n_calls = len(calls)
    n_sub = sum(c.n_units for c in calls)
    print(
        f"  {n_calls} CallUnits, {n_sub} sub-call units, mean {n_sub / max(n_calls, 1):.1f} units/call",
        flush=True,
    )

    common = dict(min_n=args.min_n, max_n=args.max_n, min_support=3, min_calls=2)

    print(f"\n{'=' * 78}")
    print(
        f"3-WAY RECURRENT-ACOUSTIC-TERM DISCOVERY (n={n_calls} calls, k={args.k_clusters}, "
        f"L={args.min_n}-{args.max_n}, dtw_thr={args.dist_threshold})"
    )
    print(f"{'=' * 78}")
    print(f"  {'method':22} {'#mot':>5} {'support':>7} {'cov':>5}  {'calltype':>16}  {'caller':>7}")
    print("  " + "-" * 74)

    print("\n[A] exact n-grams (sequence-exact; fragments variants)")
    exact = sm.exact_ngram_clusters(calls, **common)
    summarize("exact n-grams", exact, n_calls)

    print("\n[B] fixed-length cosine (substitutions only; misses indels)")
    fixed = sm.fixed_length_cosine_clusters(
        calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    summarize("fixed-length cosine", fixed, n_calls)

    print("\n[C] DTW with indels (DUSTED; recovers variant families)")
    dtw = sm.discover_recurrent_subsequences(
        calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    summarize("DTW with indels", dtw, n_calls)

    # ─── DTW null + stability on a subset (tractable) ───
    print(
        f"\n{'=' * 78}\nDTW order-shuffle null + bootstrap stability (n={args.n_null} subset)\n{'=' * 78}"
    )
    sub = calls[: args.n_null] if len(calls) >= args.n_null else calls
    null = sm.discovery_null(
        sub,
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        min_calls=2,
        n_perm=args.n_perm,
        seed=args.seed,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    print(
        f"  order-shuffle null: observed support={null['observed_support']}  "
        f"null={null['null_mean']:.1f}±{null['null_std']:.1f}  p={null['p']:.3f}  "
        f"pct={null['percentile']:.0f}%"
    )
    print(
        f"  → {'ORDER CARRIES MOTIFS' if null['observed_support'] > null['null_mean'] else 'within null'}"
    )

    jac = sm.bootstrap_stability(
        sub,
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        min_calls=2,
        n_boot=args.n_boot,
        top_k=10,
        seed=args.seed,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    print(
        f"  bootstrap top-10 Jaccard: {jac:.3f} ({'STABLE' if jac > 0.5 else 'MODERATE' if jac > 0.3 else 'UNSTABLE'})"
    )

    # ─── Top DTW motifs (recurrent acoustic terms) ───
    print(f"\n{'=' * 78}\nTop DTW-discovered recurrent acoustic terms\n{'=' * 78}")
    print(sm.format_motif_table(dtw, top=12))

    print("\nHECB-DUSTED DONE  (discoveries = recurrent acoustic terms, NOT words)")


if __name__ == "__main__":
    main()
