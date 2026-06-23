#!/usr/bin/env python3
"""Bat held-out emitter validation (70/30 split).

The core bat claim (§13): recurrent acoustic-term families are tied to emitter identity
(~2.8× more than context). This script tests whether that holds on HELD-OUT bouts —
train motif families on 70%, assign held-out 30% to nearest family centroid, measure
Emitter V + Context V on the unseen data. If the emitter signal survives held-out, it
is NOT a session/file artifact — it generalizes.

Run:  python scripts/bat_heldout.py --n-bouts 2000

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

from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.similarity import cosine  # noqa: E402

import dusted_bat as DB  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Bat held-out emitter validation (70/30)")
    ap.add_argument("--n-bouts", type=int, default=2000)
    ap.add_argument("--k-clusters", type=int, default=24)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument(
        "--assign-cosine",
        type=float,
        default=0.70,
        help="cosine threshold for held-out window→motif assignment",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading bat bouts (raw NBD sub-call units)...", flush=True)
    ann = DB.parse_annotations(DB.DATA / "annotations.csv")
    calls = DB.build_calls(ann, args.n_bouts, args.seed)
    n_sub = sum(c.n_units for c in calls)
    print(
        f"  {len(calls)} bouts, {n_sub} sub-call units, "
        f"{len({c.caller for c in calls})} emitters, {len({c.calltype for c in calls})} contexts",
        flush=True,
    )
    if not DB.tokenize(calls, args.k_clusters):
        print("ERROR: too few sub-call units for k-means")
        sys.exit(1)

    # normalize all fingerprints together (consistent train/test space)
    calls = sm._normalize_calls(calls)

    # 70/30 split
    rng = random.Random(args.seed)
    idxs = list(range(len(calls)))
    rng.shuffle(idxs)
    split = int(0.7 * len(idxs))
    train_calls = [calls[i] for i in idxs[:split]]
    test_calls = [calls[i] for i in idxs[split:]]
    # reassign call_idx for clean indexing
    for ci, c in enumerate(train_calls):
        c.call_idx = ci
    for ci, c in enumerate(test_calls):
        c.call_idx = ci
    print(f"  train: {len(train_calls)} bouts | test: {len(test_calls)} bouts", flush=True)

    common = dict(min_n=args.min_n, max_n=args.max_n, min_support=3, min_calls=2)

    # ── Train: discover motif families on 70% ──
    print("discovering motif families on train (70%)...", flush=True)
    train_motifs = sm.discover_recurrent_subsequences(
        train_calls,
        **common,
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        normalize=False,
    )
    train_s = sm.summarize_discovery(train_motifs, len(train_calls))
    print(
        f"  train: {train_s['n_motifs']} motifs, cov={train_s['coverage']:.3f}, "
        f"H(norm)={train_s['family_entropy_norm']:.2f}, top={train_s['top_family_support_fraction']:.2f}, "
        f"emit_V={train_s['mean_caller_V']:.3f}, ctx_V={train_s['mean_calltype_V']:.3f}",
        flush=True,
    )

    # extract motif centroids from prototypes
    motif_centroids = []
    for m in train_motifs:
        if m.prototype is not None and m.prototype.fps:
            mc = np.mean(np.asarray(m.prototype.fps, dtype=np.float32), axis=0)
            motif_centroids.append((m.motif_id, mc, m))
    if not motif_centroids:
        print("ERROR: no motif prototypes with fingerprints")
        sys.exit(1)
    print(f"  {len(motif_centroids)} motif centroids extracted", flush=True)

    # ── Held-out: assign test windows to nearest motif centroid ──
    print("assigning held-out test windows to nearest motif...", flush=True)
    test_windows = sm.window_subsequences(test_calls, args.min_n, args.max_n)
    # call_idx → set of motif_ids present
    call_motifs: dict[int, set[str]] = {}
    n_assigned = 0
    for w in test_windows:
        wc = np.mean(np.asarray(w.fps, dtype=np.float32), axis=0)
        best_sim, best_id = 0.0, None
        for mid, mc, _ in motif_centroids:
            sim = cosine(wc, mc)
            if sim > best_sim:
                best_sim, best_id = sim, mid
        if best_sim >= args.assign_cosine and best_id is not None:
            call_motifs.setdefault(w.call_idx, set()).add(best_id)
            n_assigned += 1
    print(
        f"  {n_assigned}/{len(test_windows)} test windows assigned "
        f"({n_assigned / max(len(test_windows), 1) * 100:.1f}%)",
        flush=True,
    )

    # ── Measure Emitter V + Context V on held-out ──
    from scipy.stats import chi2_contingency  # noqa: PLC0415

    def enrichment(has, labels):
        ulabels = sorted(set(labels))
        if len(ulabels) < 2:
            return 0.0, 1.0
        idx = {lab: k for k, lab in enumerate(ulabels)}
        table = np.zeros((2, len(ulabels)))
        for h, lab in zip(has, labels):
            table[int(bool(h)), idx[lab]] += 1
        if table.sum() < 10 or table.sum(axis=0).min() == 0:
            return 0.0, 1.0
        try:
            chi2, p, _, _ = chi2_contingency(table)
            V = min(float(np.sqrt(chi2 / (table.sum() * (min(2, len(ulabels)) - 1)))), 1.0)
            return V, float(p)
        except Exception:  # noqa: BLE001
            return 0.0, 1.0

    emitter_labels = [c.caller for c in test_calls]
    context_labels = [c.calltype for c in test_calls]
    emit_Vs, ctx_Vs = [], []
    for mid, _, m in motif_centroids:
        has = [1 if mid in call_motifs.get(c.call_idx, set()) else 0 for c in test_calls]
        eV, _ = enrichment(has, emitter_labels)
        cV, _ = enrichment(has, context_labels)
        emit_Vs.append(eV)
        ctx_Vs.append(cV)
    # family entropy on held-out assignments
    supports = [len(call_motifs.get(c.call_idx, set())) for c in test_calls]
    total_assigned = sum(1 for s in supports if s > 0)
    coverage_heldout = total_assigned / max(len(test_calls), 1)

    # held-out family support distribution (how many motifs each test call matches)
    from collections import Counter

    motif_match_counts = Counter()
    for c in test_calls:
        for mid in call_motifs.get(c.call_idx, set()):
            motif_match_counts[mid] += 1
    if motif_match_counts:
        total_matches = sum(motif_match_counts.values())
        probs = [v / total_matches for v in motif_match_counts.values()]
        heldout_entropy = float(-sum(p * np.log2(p) for p in probs if p > 0))
        heldout_top = max(motif_match_counts.values()) / max(total_matches, 1)
    else:
        heldout_entropy, heldout_top = 0.0, 0.0

    mean_emit_V = float(np.mean(emit_Vs)) if emit_Vs else 0.0
    mean_ctx_V = float(np.mean(ctx_Vs)) if ctx_Vs else 0.0

    print(f"\n{'=' * 80}")
    print(f"BAT HELD-OUT EMITTER VALIDATION (70/30 split, {len(test_calls)} held-out bouts)")
    print(f"{'=' * 80}")
    print("\n  Train (70%):")
    print(
        f"    #motifs={train_s['n_motifs']}  cov={train_s['coverage']:.3f}  "
        f"H={train_s['family_entropy_norm']:.2f}  top={train_s['top_family_support_fraction']:.2f}"
    )
    print(
        f"    Emitter V={train_s['mean_caller_V']:.3f}  Context V={train_s['mean_calltype_V']:.3f}"
    )
    print("\n  Held-out (30%):")
    print(
        f"    windows assigned: {n_assigned}/{len(test_windows)} ({n_assigned / max(len(test_windows), 1) * 100:.1f}%)"
    )
    print(f"    bouts covered: {total_assigned}/{len(test_calls)} ({coverage_heldout:.3f})")
    print(f"    held-out entropy={heldout_entropy:.2f}  top-family-frac={heldout_top:.2f}")
    print(
        f"    ** Emitter V={mean_emit_V:.3f}  Context V={mean_ctx_V:.3f}  (ratio {mean_emit_V / max(mean_ctx_V, 1e-6):.2f}x) **"
    )
    verdict = (
        "EMITTER > CONTEXT on held-out — claim GENERALIZES"
        if mean_emit_V > mean_ctx_V
        else "context ≥ emitter on held-out — does NOT generalize"
    )
    print(f"\n  → {verdict}")
    print("\nBAT HELD-OUT DONE")


if __name__ == "__main__":
    main()
