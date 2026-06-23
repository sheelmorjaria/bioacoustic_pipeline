#!/usr/bin/env python3
"""Marmoset pooled-representation V/NMI: k-means vs cosine-HAC vs GMM.

Phase 3 cross-species-ablation cluster-function test. The architectural feedback is that
spherical k-means confuses the pooled representation (whole-call fingerprint ‖ mean
sub-call fp ‖ [n_sub, internal_frac]) because of its spherical-cluster bias. We now have
``cosine_hac_tokens`` and ``gmm_tokens`` (``motif_miner.subsequence_motifs``) as
non-spherical alternatives. This script runs the standard whole/sub/pooled ablation on
marmoset call-types with all three clusterers and reports whether pooled V/NMI improves
when the clusterer matches the cosine/elliptical geometry of the fingerprint space.

Uses ``whole_sub_pooled_ablation`` from ``representation.subcall`` (now carrying a
``cluster_fn`` parameter) on the 6 named marmoset call-types (Twitter/Tsik/Phee/Trill/
Infant/Seep). This is the lightweight path (no sequence view); the full 4-way
``cross_species_ablation.py`` shares the same cluster_fn seam via
``whole_sub_pooled_sequence_ablation``.

Run:  python scripts/marmoset_clusterfn_compare.py --n 2000 --k 8

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner.subsequence_motifs import (  # noqa: E402
    cosine_hac_tokens,
    gmm_tokens,
)
from bioacoustic_pipeline.representation.subcall import (  # noqa: E402
    build_subcall_representations,
    whole_sub_pooled_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import CallEvent, InternalSegment  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

MARMOSET = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
SR = 96000
_PROPOSAL = {"floor_energy": 0.05, "floor_flux": 0.03, "debounce_ms": 30.0}
MIN_INTERNAL_MS = 120.0
MAX_INTERNAL = 6
NAMED_TYPES = {"Twitter", "Tsik", "Phee", "Trill", "Infant", "Seep"}


def nbd_internal_call_event(audio, sr, *, max_internal=MAX_INTERNAL, hop_ms=10.0):
    """Whole file = oracle parent; NBD peaks inside → internal sub-call boundaries."""
    dur_s = len(audio) / sr
    dur_ms = dur_s * 1000.0
    call = CallEvent(call_id="oracle", onset_s=0.0, offset_s=dur_s)
    if dur_ms < MIN_INTERNAL_MS:
        return call
    sig = compute_signals(audio, sr, hop_ms=hop_ms)
    props = generate_proposals(sig, **_PROPOSAL)
    cuts = sorted(p.time_ms for p in props if 50 < p.time_ms < dur_ms - 50)
    cuts = cuts[:max_internal]
    edges = [0.0] + [c / 1000.0 for c in cuts] + [dur_s]
    call.internal_segments = [
        InternalSegment(
            parent_call_id="oracle", start_s=edges[k], end_s=edges[k + 1], boundary_type="internal"
        )
        for k in range(len(edges) - 1)
    ]
    return call


def load_marmoset(n, seed):
    rng = random.Random(seed)
    rows = []
    with open(MARMOSET / "Annotations.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = r["label"].strip()
            if lab in NAMED_TYPES and r["file_name"].strip():
                rows.append((lab, r["parent_name"].strip(), r["file_name"].strip()))
    rng.shuffle(rows)
    out = []
    for lab, parent, fname in rows:
        flac = MARMOSET / parent / fname
        if flac.exists():
            out.append((str(flac), lab))
        if len(out) >= n:
            break
    return out


def build_reps(files):
    fp_fn = lambda seg, s: afp.extract(seg, s, species="marmoset").combined  # noqa: E731
    reps, labels = [], []
    for fi, (path, lab) in enumerate(files):
        audio, _ = load_audio(path, target_sr=SR)
        if audio is None or audio.size < int(SR * 0.05):
            continue
        call = nbd_internal_call_event(audio, SR)
        rs = build_subcall_representations([call], audio, SR, fp_fn)
        if rs:
            reps.append(rs[0])
            labels.append(lab)
        if (fi + 1) % 500 == 0:
            print(f"  {fi + 1}/{len(files)} → {len(reps)} reps", flush=True)
    print(f"  built {len(reps)} reps | {dict(Counter(labels))}", flush=True)
    return reps, labels


def _print_row(name, r):
    if r is None:
        print(f"  {name:24} {'(n/a)':>10}")
        return
    print(f"  {name:24} V={r['V']:.3f}  nmi={r['nmi']:.3f}  p={r['p']}  probe={r['probe']}")


def main():
    ap = argparse.ArgumentParser(description="Marmoset pooled V/NMI: k-means vs cosine-HAC vs GMM")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = load_marmoset(args.n, args.seed)
    print(f"loaded {len(files)} marmoset calls (96 kHz, named call-types)", flush=True)
    reps, labels = build_reps(files)
    if len(reps) < 50:
        print("ERROR: too few reps")
        sys.exit(1)

    clusterers = {
        "k-means (spherical)": None,
        "cosine-HAC": cosine_hac_tokens,
        "GMM (diag)": gmm_tokens,
    }

    print(f"\n{'=' * 78}")
    print(f"Marmoset whole/sub/pooled ablation — clusterer comparison (n={len(reps)}, k={args.k})")
    print(f"{'=' * 78}")

    summary = {}
    for cname, fn in clusterers.items():
        print(f"\n--- cluster_fn = {cname} ---")
        res = whole_sub_pooled_ablation(
            reps, labels, k=args.k, n_perm=200, min_per_class=5, cluster_fn=fn
        )
        _print_row("whole call", res["whole"])
        _print_row("whole + sub pool", res["pooled"])
        _print_row("sub-call (parent lbl)", res["subcall"])
        summary[cname] = res

    print(f"\n{'=' * 78}")
    print("POOLED V/NMI across clusterers (the architectural question)")
    print(f"{'-' * 78}")
    print(f"  {'clusterer':24} {'pooled V':>9} {'pooled nmi':>11} {'whole V':>9} {'sub V':>8}")
    for cname, res in summary.items():
        pv, pn = res["pooled"]["V"], res["pooled"]["nmi"]
        wv = res["whole"]["V"]
        sv = res["subcall"]["V"] if res["subcall"] else 0.0
        print(f"  {cname:24} {pv:9.3f} {pn:11.3f} {wv:9.3f} {sv:8.3f}")

    print("\n=== read ===")
    print("  If cosine-HAC / GMM pooled V >> k-means pooled V:")
    print("    → k-means spherical bias WAS depressing pooled; pooled > whole is real once")
    print("      the clusterer matches the cosine/elliptical geometry (confirms architectural fix)")
    print("  If all three agree on pooled V:")
    print("    → the pooled representation is clusterer-robust; k-means was NOT the culprit")
    print("MARMOSET CLUSTERFN COMPARE DONE")


if __name__ == "__main__":
    main()
