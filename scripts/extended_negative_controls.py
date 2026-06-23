#!/usr/bin/env python3
"""Extended negative controls — prove the pipeline doesn't find structure everywhere (§35).

Reviewer attack: "DUSTED finds motifs in any audio." These controls prove specificity:

  1. ORDER-SHUFFLED bat — destroy the unit sequence order, keep the same acoustic units.
     If null_ratio collapses to ~1.0, motifs are ORDER-dependent (not unit-frequency artifacts).
  2. SYNTHETIC CHIRPS — generate non-biological FM-sweep sequences. If DUSTED finds no
     significant motifs (null_ratio ≈ 1.0), the pipeline doesn't manufacture structure from
     arbitrary acoustic input.
  3. RANDOM SEGMENTS — take random audio chunks (not bounded by call boundaries). If
     null_ratio ≈ 1.0, the pipeline doesn't find motifs in non-vocalization audio.

The NATURAL bat baseline (null_ratio 1.36×, §22) is the reference: controls should all be ≪.

Run:  python scripts/extended_negative_controls.py --n-bouts 100

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import dusted_bat as DB  # noqa: E402

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402


def _dusted(calls, label, dist_threshold=0.15, centroid_radius=0.20):
    """Run DUSTED on a list of CallUnits → summary dict (null_ratio is the key metric)."""
    if len(calls) < 30:
        return {
            "label": label,
            "n_calls": len(calls),
            "null_ratio": None,
            "n_motifs": 0,
            "coverage": None,
            "family_entropy": None,
        }
    common = dict(min_n=2, max_n=4, min_support=3, min_calls=2)
    motifs = sm.discover_recurrent_subsequences(
        calls, **common, dist_threshold=dist_threshold, centroid_radius=centroid_radius
    )
    s = sm.summarize_discovery(motifs, len(calls))
    return {
        "label": label,
        "n_calls": len(calls),
        "null_ratio": s.get("family_entropy_norm")
        and s.get("top_family_support_fraction")
        and None,  # placeholder — null_ratio comes from discovery_null
        "n_motifs": s["n_motifs"],
        "coverage": round(s.get("coverage", 0), 3),
        "family_entropy": round(s.get("family_entropy_norm", s.get("family_entropy", 0)), 3),
        "top_family_fraction": round(s.get("top_family_support_fraction", 0), 3),
    }


def _dusted_with_null(calls, label, seed=0, dist_threshold=0.15, centroid_radius=0.20):
    """Run DUSTED + the order-shuffle null → null_ratio = observed/null support."""
    if len(calls) < 30:
        return {"label": label, "n_calls": len(calls), "null_ratio": None, "n_motifs": 0}
    common = dict(min_n=2, max_n=4, min_support=3, min_calls=2)
    motifs = sm.discover_recurrent_subsequences(
        calls, **common, dist_threshold=dist_threshold, centroid_radius=centroid_radius
    )
    s = sm.summarize_discovery(motifs, len(calls))
    null = sm.discovery_null(
        calls,
        **common,
        dist_threshold=dist_threshold,
        centroid_radius=centroid_radius,
        n_perm=50,
        seed=seed,
    )
    obs_support = sum(m.support for m in motifs)
    null_support = null.get("mean_null_support", null.get("observed_support", 1))
    ratio = obs_support / max(null_support, 1) if null_support else float("inf")
    return {
        "label": label,
        "n_calls": len(calls),
        "n_motifs": s["n_motifs"],
        "coverage": round(s.get("coverage", 0), 3),
        "observed_support": obs_support,
        "null_support": round(null_support, 1) if null_support else None,
        "null_ratio": round(ratio, 2),
    }


def build_bat_calls(n_bouts, seed):
    """Natural bat bouts → CallUnits (the baseline)."""
    ann = DB.parse_annotations(DB.DATA / "annotations.csv")
    rng = random.Random(seed)
    files = sorted(ann.keys())
    rng.shuffle(files)
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species=DB.SPECIES).combined, dtype=np.float32
    )
    calls = []
    for fn in files:
        if len(calls) >= n_bouts:
            break
        try:
            audio, sr = load_audio(str(DB.DATA / "audio" / fn), target_sr=None)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.03 * sr):
            continue
        fps = []
        for s_ms, e_ms in DB.segment_bout(audio, sr):
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
                    clusters=list(range(len(fps))),  # placeholder; k-means done below
                    calltype=str(context),
                    caller=str(emitter),
                    file_id=fn,
                    call_idx=len(calls),
                )
            )
    # k-means cluster the fingerprints
    _kmeans_cluster(calls, k=24)
    return calls


def _kmeans_cluster(calls, k=24):
    """Assign k-means cluster IDs to each call's fingerprints."""
    all_fp = [fp for c in calls for fp in c.fingerprints]
    if len(all_fp) < k * 5:
        return
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

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


def control_order_shuffled(calls, seed):
    """Shuffle the unit order within each call → motifs should collapse."""
    rng = random.Random(seed)
    shuffled = []
    for c in calls:
        order = list(range(len(c.clusters)))
        rng.shuffle(order)
        shuffled.append(
            sm.CallUnits(
                fingerprints=[c.fingerprints[i] for i in order],
                clusters=[c.clusters[i] for i in order],
                calltype=c.calltype,
                caller=c.caller,
                file_id=c.file_id + "_shuf",
                call_idx=c.call_idx,
            )
        )
    return shuffled


def control_synthetic_chirps(n_calls, seed):
    """Generate synthetic FM-sweep sequences (non-biological) → no real motifs expected."""
    rng = np.random.RandomState(seed)
    sr = 250000
    calls = []
    for i in range(n_calls):
        n_units = rng.randint(3, 12)
        fps = []
        for _u in range(n_units):
            # synthetic chirp: FM sweep from f0 to f1 over ~20ms
            f0 = rng.uniform(20000, 80000)
            f1 = f0 + rng.uniform(-20000, 20000)
            dur = rng.uniform(0.01, 0.03)
            t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
            freq = np.linspace(f0, f1, len(t))
            phase = np.cumsum(2 * np.pi * freq / sr)
            chirp = (0.3 * np.sin(phase) * np.exp(-t * 20)).astype(np.float32)
            fp = np.asarray(afp.extract(chirp, sr, species="").combined, dtype=np.float32)
            if np.any(fp):
                fps.append(fp)
        if len(fps) >= 2:
            calls.append(
                sm.CallUnits(
                    fingerprints=fps,
                    clusters=list(range(len(fps))),
                    calltype="synthetic",
                    caller="synthetic",
                    file_id=f"synth_{i}",
                    call_idx=i,
                )
            )
    _kmeans_cluster(calls, k=24)
    return calls


def control_random_segments(n_calls, seed):
    """Take random audio chunks (not call-bounded) → no vocalization structure expected."""
    ann = DB.parse_annotations(DB.DATA / "annotations.csv")
    rng = random.Random(seed)
    files = sorted(ann.keys())
    rng.shuffle(files)
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species=DB.SPECIES).combined, dtype=np.float32
    )
    calls = []
    for fn in files:
        if len(calls) >= n_calls:
            break
        try:
            audio, sr = load_audio(str(DB.DATA / "audio" / fn), target_sr=None)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.5 * sr):
            continue
        # take 3-8 RANDOM chunks (not bounded by NBD calls)
        n_chunks = rng.randint(3, 8)
        chunk_len = int(0.03 * sr)
        fps = []
        for _ in range(n_chunks):
            start = rng.randint(0, max(0, len(audio) - chunk_len))
            seg = audio[start : start + chunk_len]
            if len(seg) >= int(0.012 * sr):
                fp = np.asarray(fp_fn(seg, sr), dtype=np.float32)
                if np.any(fp):
                    fps.append(fp)
        if len(fps) >= 2:
            calls.append(
                sm.CallUnits(
                    fingerprints=fps,
                    clusters=list(range(len(fps))),
                    calltype="random",
                    caller="random",
                    file_id=fn + "_rand",
                    call_idx=len(calls),
                )
            )
    _kmeans_cluster(calls, k=24)
    return calls


def main():
    ap = argparse.ArgumentParser(description="Extended negative controls (§35)")
    ap.add_argument("--n-bouts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("building natural bat calls (baseline)...", flush=True)
    natural = build_bat_calls(args.n_bouts, args.seed)
    print(f"  {len(natural)} natural bouts", flush=True)

    print("control 1: order-shuffled...", flush=True)
    shuffled = control_order_shuffled(natural, args.seed)

    print("control 2: synthetic chirps...", flush=True)
    synthetic = control_synthetic_chirps(args.n_bouts, args.seed)
    print(f"  {len(synthetic)} synthetic chirp sequences", flush=True)

    print("control 3: random segments...", flush=True)
    random_segs = control_random_segments(args.n_bouts, args.seed)
    print(f"  {len(random_segs)} random-segment sequences", flush=True)

    print("\nrunning DUSTED + null on each...", flush=True)
    results = {
        "natural_bat": _dusted_with_null(natural, "natural bat", args.seed),
        "order_shuffled": _dusted_with_null(shuffled, "order-shuffled bat", args.seed),
        "synthetic_chirps": _dusted_with_null(synthetic, "synthetic chirps", args.seed),
        "random_segments": _dusted_with_null(random_segs, "random segments", args.seed),
    }

    print(
        f"\n{'=' * 90}\nEXTENDED NEGATIVE CONTROLS (§35) — does the pipeline find structure everywhere?\n{'=' * 90}"
    )
    print(f"  {'control':>20}  {'n':>5}  {'#motifs':>7}  {'cov':>5}  {'null_ratio':>10}")
    print("  " + "-" * 60)
    for label, r in results.items():
        print(
            f"  {label:>20}  {r['n_calls']:>5}  {r['n_motifs']:>7}  "
            f"{r.get('coverage', 'n/a'):>5}  {r.get('null_ratio', 'n/a'):>10}"
        )

    nat_ratio = results["natural_bat"].get("null_ratio") or 1.0
    print(f"\n  natural null_ratio: {nat_ratio}")
    for label in ("order_shuffled", "synthetic_chirps", "random_segments"):
        r = results[label]
        ctrl_ratio = r.get("null_ratio")
        if ctrl_ratio is not None and nat_ratio > 0:
            frac = ctrl_ratio / nat_ratio
            verdict = "PASS (collapses)" if frac < 0.5 else "WARN (does not collapse)"
            print(f"  {label}: null_ratio={ctrl_ratio} ({frac:.0%} of natural) → {verdict}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "extended_negative_controls.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved outputs/extended_negative_controls.json")
    print("\nEXTENDED NEGATIVE CONTROLS DONE")


if __name__ == "__main__":
    main()
