#!/usr/bin/env python3
"""DUSTED recurrent-acoustic-term discovery on Egyptian fruit-bat bouts.

Each wav is a 1.5–3 s multi-call BOUT with per-bout Emitter/Addressee/Context labels
(annotations.csv) but NO within-bout timestamps. So we segment each bout into sub-call
units with **raw NBD** (bat_call_granular) — NOT a marmoset-trained HECB classifier (the
onset/body/offset roles in DTW are positional/geometry-derived, safe). Each bout = one
CallUnits (the bout-as-sentence view). The decisive question: do bat bouts form stable
recurrent acoustic-term families, and are they tied more to EMITTER/social structure than
to CONTEXT labels?

Discovery target = RECURRENT ACOUSTIC TERMS, NOT words. `semantic_validation` stays `none`.
Three-way comparison (exact n-grams / fixed-length cosine / DTW with indels) + the
diverse-vs-degenerate diagnostics (`family_entropy`, `top_family_support_fraction`) +
order-shuffle null + bootstrap. calltype=Context, caller=Emitter → Context V vs Emitter V.

Run:  python scripts/dusted_bat.py --n-bouts 2000

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
SPECIES = "egyptian_fruit_bat"
_PROPOSAL = dict(floor_energy=0.05, floor_flux=0.05, debounce_ms=25.0)  # bat_call_granular
HOP_MS = 10.0


def parse_annotations(csv_path: Path) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["File Name"]] = (row["Emitter"], row["Addressee"], row["Context"])
    return out


def segment_bout(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Raw NBD → sub-call (start_ms, end_ms) segments between consecutive proposals."""
    sig = compute_signals(audio, sr, hop_ms=HOP_MS)
    props = generate_proposals(sig, **_PROPOSAL)
    times = [0.0] + [p.time_ms for p in props] + [len(audio) / sr * 1000.0]
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def build_calls(ann, n_bouts, seed):
    files = sorted(ann.keys())
    random.Random(seed).shuffle(files)
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species=SPECIES).combined  # noqa: E731
    calls: list[sm.CallUnits] = []
    for fn in files:
        if len(calls) >= n_bouts:
            break
        try:
            audio, sr = load_audio(str(DATA / "audio" / fn), target_sr=None)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.03 * sr):
            continue
        fps = []
        for s_ms, e_ms in segment_bout(audio, sr):
            si = int(s_ms / 1000 * sr)
            sj = max(int(e_ms / 1000 * sr), si + int(0.012 * sr))
            seg = audio[si:sj]
            if len(seg) < int(0.012 * sr):
                continue
            fp = np.asarray(fp_fn(seg, sr), dtype=np.float32)
            if np.any(fp):
                fps.append(fp)
        if len(fps) >= 2:
            emitter, _, context = ann[fn]
            calls.append(
                sm.CallUnits(
                    fingerprints=fps,
                    calltype=str(context),
                    caller=str(emitter),
                    file_id=fn,
                    call_idx=len(calls),
                )
            )
    return calls


def tokenize(calls, k):
    """Global k-means over all sub-call fingerprints → assign cluster ids back."""
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    all_fp = np.asarray([fp for c in calls for fp in c.fingerprints], dtype=np.float32)
    if len(all_fp) < k * 5:
        return False
    scaler = StandardScaler().fit(all_fp)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(scaler.transform(all_fp))
    cents = km.cluster_centers_
    for c in calls:
        sv = scaler.transform(np.asarray(c.fingerprints, dtype=np.float32))
        c.clusters = np.argmin(
            ((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1
        ).tolist()
    return True


def main():
    ap = argparse.ArgumentParser(description="DUSTED on Egyptian fruit-bat bouts")
    ap.add_argument("--n-bouts", type=int, default=2000)
    ap.add_argument("--k-clusters", type=int, default=24)
    ap.add_argument("--n-null", type=int, default=600)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("parsing annotations.csv...", flush=True)
    ann = parse_annotations(DATA / "annotations.csv")
    print(f"  {len(ann)} bouts annotated", flush=True)
    print(f"building sub-call units on {args.n_bouts} bouts (raw NBD)...", flush=True)
    calls = build_calls(ann, args.n_bouts, args.seed)
    n_sub = sum(c.n_units for c in calls)
    n_emit = len({c.caller for c in calls})
    n_ctx = len({c.calltype for c in calls})
    print(
        f"  {len(calls)} bouts, {n_sub} sub-call units ({n_sub / max(len(calls), 1):.1f}/bout), "
        f"{n_emit} emitters, {n_ctx} contexts",
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
            f"ctx_V={s['mean_calltype_V']:.3f} emit_V={s['mean_caller_V']:.3f} "
            f"purity(w)={s['wmean_label_purity']:.2f}"
        )
        return s

    print(
        f"\n{'=' * 90}\nBAT BOUT RECURRENT-ACOUSTIC-TERM DISCOVERY "
        f"(n={len(calls)} bouts, k={args.k_clusters}, L={args.min_n}-{args.max_n}, "
        f"thr={args.dist_threshold})\n{'=' * 90}"
    )
    print(
        f"  {'method':20} {'#fam':>4} {'cov':>5} {'H/norm':>6} {'topFrac':>7} "
        f"{'ctx_V':>6} {'emit_V':>6} {'purity':>6}"
    )
    print("  " + "-" * 86)

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
        f"\n  DTW null: obs_support={null['observed_support']} null={null['null_mean']:.0f}"
        f"±{null['null_std']:.0f} (ratio {null['observed_support'] / max(null['null_mean'], 1):.2f}x) "
        f"p={null['p']:.3f}  |  bootstrap Jaccard={jac:.3f}",
        flush=True,
    )
    print(
        f"\n  top-family support fraction (DTW) = {sd['top_family_support_fraction']:.2f}  "
        f"({'DEGENERATE (one dominant)' if sd['top_family_support_fraction'] > 0.5 else 'DIVERSE inventory'})"
    )
    print(
        f"  Emitter V ({sd['mean_caller_V']:.3f}) vs Context V ({sd['mean_calltype_V']:.3f}) "
        f"→ {'EMITTER/social > context' if sd['mean_caller_V'] > sd['mean_calltype_V'] else 'context >= emitter'}"
    )

    print(f"\n{'=' * 90}\nTop DTW-discovered recurrent acoustic terms\n{'=' * 90}")
    print(sm.format_motif_table(dtw, top=10))
    print("\nDUSTED BAT DONE  (discoveries = recurrent acoustic terms, NOT words)")


if __name__ == "__main__":
    main()
