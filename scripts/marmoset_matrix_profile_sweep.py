#!/usr/bin/env python3
"""Marmoset global Matrix Profile sweep (§34 targeted reanalysis #2).

A reviewer could argue the infant-marmoset degenerate inventory (§11/§22: collapsed to one
repeated unit) is a HECB-tokenization / k-means artifact. This reanalysis is token-free +
assumption-free: run Matrix Profile directly on the 44-D frame-fingerprint trajectory of a LARGE
marmoset sample (bypassing HECB and DUSTED entirely) and check whether MP finds ANY structure
beyond the dominant stutter.

Headline: the per-call recurrence_fraction distribution. If recurrence is uniformly high (most
windows have a near-duplicate ⇒ pervasive stutter) with NO low-recurrence / high-diversity tail,
the degenerate-inventory claim is tokenization-independent. The "secondary motif" check looks for
a bimodal motif-distance distribution (a second recurrent structure would show as a distinct
low-distance cluster). Firewall-clean (all-species fingerprint). NOT meaning.

Run:  python scripts/marmoset_matrix_profile_sweep.py --n-calls 200

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.structural_alternatives import (  # noqa: E402
    matrix_profile,
    rqa_for_trajectory,
)
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

WIN_MS = 80.0
HOP_MS = 40.0
MAX_FRAMES = 120
MAR_SR = 96000
MAR_GLOB = "/home/sheel/birdsong_analysis/data/Vocalizations/*/Vocalization_*.flac"


def frame_trajectory(audio, sr, fp_fn):
    win = int(WIN_MS / 1000 * sr)
    hop = int(HOP_MS / 1000 * sr)
    frames = []
    i = 0
    while i + win <= len(audio) and len(frames) < MAX_FRAMES:
        v = np.asarray(fp_fn(audio[i : i + win], sr), dtype=np.float32)
        if np.any(v):
            frames.append(v)
        i += hop
    return np.asarray(frames, dtype=np.float32) if frames else np.zeros((0, 1), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Marmoset global Matrix Profile sweep (§34 #2)")
    ap.add_argument("--n-calls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species="").combined, dtype=np.float32
    )

    files = sorted(glob.glob(MAR_GLOB))
    random.Random(args.seed).shuffle(files)
    print(f"{len(files)} marmoset calls available; sampling {args.n_calls}", flush=True)

    rec_fracs = []
    motif_dists = []
    lams = []
    n = 0
    for f in files:
        if n >= args.n_calls:
            break
        try:
            audio, sr = load_audio(f, target_sr=MAR_SR)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.2 * MAR_SR):
            continue
        t = frame_trajectory(audio, sr, fp_fn)
        if len(t) < 16:
            continue
        mp = matrix_profile(t, m=8)
        rqa = rqa_for_trajectory(t)
        if mp.get("motif_distance") is not None:
            rec_fracs.append(mp["recurrence_fraction"])
            motif_dists.append(mp["motif_distance"])
        lams.append(rqa["laminarity"])
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{args.n_calls}", flush=True)

    rec = np.asarray(rec_fracs)
    md = np.asarray(motif_dists)
    print(
        f"\n{'=' * 90}\nMARMOSET GLOBAL MATRIX PROFILE SWEEP (§34 #2) — token-free degeneracy check\n{'=' * 90}"
    )
    print(f"  n_calls processed: {n}")
    print(
        f"  recurrence_fraction: mean={rec.mean():.3f} median={np.median(rec):.3f} "
        f"min={rec.min():.3f} max={rec.max():.3f}"
    )
    print(f"  fraction calls with recurrence > 0.5 (stutter-dominant): {(rec > 0.5).mean():.1%}")
    print(f"  fraction calls with recurrence > 0.7 (strong stutter): {(rec > 0.7).mean():.1%}")
    print(f"  motif_distance (closest repeat): mean={md.mean():.3f} median={np.median(md):.3f}")
    print(f"  RQA laminarity (state-holding): mean={np.mean(lams):.3f}")
    # secondary-motif check: bimodality of motif_distance (a 2nd recurrent structure ⇒ a 2nd mode)
    hist, edges = np.histogram(md, bins=10)
    n_modes = int(np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])))
    print(
        f"  motif_distance histogram modes (interior peaks): {n_modes} (1 ⇒ unimodal/no 2nd family)"
    )
    high_div = (rec < 0.3).sum()
    print(f"  calls with LOW recurrence (<0.3 ⇒ diverse, non-stutter): {high_div}/{n}")
    print("\n  VERDICT:")
    if (rec > 0.5).mean() > 0.8 and high_div == 0 and n_modes <= 1:
        print("    token-free MP confirms the degenerate stutter is pervasive and there is NO")
        print("    secondary recurrent structure ⇒ the degenerate-inventory claim is")
        print("    tokenization-INDEPENDENT (not a HECB/k-means artifact). Bulletproof.")
    else:
        print("    MP finds structure BEYOND the dominant stutter — the inventory is richer than")
        print("    the discrete pipeline suggested (an honest positive that updates §11).")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "n_calls": n,
        "recurrence_fraction": {
            "mean": float(rec.mean()),
            "median": float(np.median(rec)),
            "min": float(rec.min()),
            "max": float(rec.max()),
            "frac_gt_0.5": float((rec > 0.5).mean()),
            "frac_gt_0.7": float((rec > 0.7).mean()),
        },
        "motif_distance": {
            "mean": float(md.mean()),
            "median": float(np.median(md)),
            "interior_modes": n_modes,
        },
        "low_recurrence_calls": int(high_div),
        "rqa_laminarity_mean": float(np.mean(lams)),
    }
    with open(out_dir / "marmoset_matrix_profile_sweep.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved outputs/marmoset_matrix_profile_sweep.json")
    print("\nMARMOSET MATRIX PROFILE SWEEP DONE")


if __name__ == "__main__":
    main()
