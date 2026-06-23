#!/usr/bin/env python3
"""RookID-DUSTED: recurrent acoustic-term discovery in individually-identified rook calls.

Extends the cross-corpus motif arc (marmoset §11 / Bengalese §12 / bat §13) to a 5th
corpus: noisy outdoor corvid recordings with individually-identified rooks. Each call =
one CallUnits (raw-NBD sub-call units, k-means clusters). calltype=Event, caller=Source.
3-way DUSTED + null + bootstrap. The key question: does RookID form diverse motif families
(like Bengalese) or degenerate (like marmoset)? Are motifs Source-associated (individual
signature) or Event-associated (call type)?

Run:  python scripts/rookid_dusted.py --max-calls 2000

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402

import rookid_individual_id as RI  # noqa: E402  (reuse parse_tsvs, seek_audio)

K_CLUSTERS = 24


def tokenize(calls, k):
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    all_fp = [fp for c in calls for fp in c.fingerprints]
    if len(all_fp) < k * 5:
        return False
    scaler = StandardScaler().fit(np.asarray(all_fp, dtype=np.float32))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init="auto", batch_size=4096).fit(
        scaler.transform(np.asarray(all_fp, dtype=np.float32))
    )
    cents = km.cluster_centers_
    for c in calls:
        sv = scaler.transform(np.asarray(c.fingerprints, dtype=np.float32))
        c.clusters = np.argmin(
            ((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1
        ).tolist()
    return True


def main():
    ap = argparse.ArgumentParser(description="RookID-DUSTED motif discovery")
    ap.add_argument("--max-calls", type=int, default=2000)
    ap.add_argument("--k-clusters", type=int, default=K_CLUSTERS)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--n-null", type=int, default=400)
    ap.add_argument("--n-perm", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("parsing RookID TSVs...", flush=True)
    raw_calls = RI.parse_tsvs()
    print(f"  {len(raw_calls)} calls (excl Inc/Pls)", flush=True)

    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="").combined  # noqa: E731
    print(f"building sub-call units on {args.max_calls} calls...", flush=True)
    calls = []
    n_sing = 0
    for ci, (stem, src, ev, s, e) in enumerate(raw_calls):
        if len(calls) >= args.max_calls:
            break
        wav = RI.DATA / f"{stem}.wav"
        if not wav.exists():
            continue
        audio = RI.seek_audio(wav, s, e)
        if audio is None or len(audio) < int(0.03 * RI.SR):
            continue
        sig = compute_signals(audio, RI.SR, hop_ms=10.0)
        props = generate_proposals(sig, floor_energy=0.04, floor_flux=0.02, debounce_ms=25.0)
        fps = []
        for p in props:
            si = max(0, int(p.time_ms / 1000 * RI.SR) - int(0.02 * RI.SR))
            sj = si + int(0.04 * RI.SR)
            seg = audio[si:sj]
            if len(seg) >= int(0.012 * RI.SR):
                v = np.asarray(fp_fn(seg, RI.SR), dtype=np.float32)
                if np.any(v):
                    fps.append(v)
        if len(fps) >= 2:
            is_sing = "sing" in ev.lower()
            if is_sing:
                n_sing += 1
            calls.append(
                sm.CallUnits(
                    fingerprints=fps,
                    calltype=ev,
                    caller=src,
                    file_id=stem,
                    call_idx=len(calls),
                )
            )
        if (ci + 1) % 500 == 0:
            print(f"  {ci + 1}/{len(raw_calls)} → {len(calls)} calls", flush=True)

    n_sub = sum(c.n_units for c in calls)
    n_sources = len({c.caller for c in calls})
    n_events = len({c.calltype for c in calls})
    print(
        f"  {len(calls)} calls, {n_sub} sub-call units ({n_sub / max(len(calls), 1):.1f}/call), "
        f"{n_sources} sources, {n_events} events, {n_sing} sing calls",
        flush=True,
    )
    if not tokenize(calls, args.k_clusters):
        print("ERROR: too few sub-call units for k-means")
        sys.exit(1)

    common = dict(min_n=args.min_n, max_n=args.max_n, min_support=3, min_calls=2)

    def _sum(name, motifs):
        s = sm.summarize_discovery(motifs, len(calls))
        print(
            f"  {name:20} fams={s['n_motifs']:4d} cov={s['coverage']:.3f} "
            f"H(norm)={s['family_entropy_norm']:.2f} top={s['top_family_support_fraction']:.2f} "
            f"src_V={s['mean_caller_V']:.3f} evt_V={s['mean_calltype_V']:.3f} "
            f"purity(w)={s['wmean_label_purity']:.2f}"
        )
        return s

    print(
        f"\n{'=' * 90}\nROOKID-DUSTED RECURRENT-ACOUSTIC-TERM DISCOVERY "
        f"(n={len(calls)} calls, k={args.k_clusters}, L={args.min_n}-{args.max_n}, "
        f"thr={args.dist_threshold})\n{'=' * 90}"
    )
    print(
        f"  {'method':20} {'#fam':>4} {'cov':>5} {'H/n':>5} {'top':>4} "
        f"{'src_V':>6} {'evt_V':>6} {'pur':>5}"
    )
    print("  " + "-" * 60)

    print("\n[A] exact n-grams")
    exact = sm.exact_ngram_clusters(calls, **common)
    _sum("exact n-grams", exact)
    print("\n[B] fixed-length cosine")
    fixed = sm.fixed_length_cosine_clusters(
        calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    _sum("fixed-length cosine", fixed)
    print("\n[C] DTW with indels")
    dtw = sm.discover_recurrent_subsequences(
        calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    sd = _sum("DTW with indels", dtw)

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
        f"\n  DTW null: obs={null['observed_support']} null={null['null_mean']:.0f}"
        f"±{null['null_std']:.0f} ({null['observed_support'] / max(null['null_mean'], 1):.2f}x) "
        f"p={null['p']:.3f}  |  bootstrap Jaccard={jac:.3f}",
        flush=True,
    )
    print(
        f"\n  top-frac={sd['top_family_support_fraction']:.2f}  "
        f"({'DIVERSE inventory' if sd['top_family_support_fraction'] < 0.5 else 'DEGENERATE'})"
    )
    print(
        f"  Source V ({sd['mean_caller_V']:.3f}) vs Event V ({sd['mean_calltype_V']:.3f}) → "
        f"{'SOURCE/individual > event' if sd['mean_caller_V'] > sd['mean_calltype_V'] else 'event ≥ source'}"
    )

    print(f"\n{'=' * 90}\nTop DTW-discovered recurrent acoustic terms\n{'=' * 90}")
    print(sm.format_motif_table(dtw, top=10))
    print("\nROOKID-DUSTED DONE  (discoveries = recurrent acoustic terms, NOT words)")


if __name__ == "__main__":
    main()
