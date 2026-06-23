#!/usr/bin/env python3
"""Bat RQA dynamics BY CONTEXT (§34 targeted reanalysis #1).

DUSTED over-merged graded bat bouts (low discrete purity, §13/§22). This reanalysis proves the
graded bouts carry CONTEXT-DEPENDENT dynamical structure that DTW can't see: compute RQA
Determinism (DET) + Laminarity (LAM) on the 44-D frame-fingerprint trajectory of each bout,
stratified by the per-bout Context label, and test whether DET/LAM differ across contexts.

Contexts are the numeric codes in annotations.csv (0–12; top-6 have ≥2k bouts each). We do not
have the code→ethological-label key, so contexts are reported by code. A Kruskal-Wallis test
across contexts on DET (and LAM) is the headline: significant ⇒ graded dynamics are
context-structured (the lens DUSTED lacked). Firewall-clean (all-species fingerprint). NOT meaning.

Run:  python scripts/bat_rqa_by_context.py --per-context 30 --top-k 6

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import dusted_bat as DB  # noqa: E402

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.structural_alternatives import rqa_for_trajectory  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

WIN_MS = 80.0
HOP_MS = 40.0
MAX_FRAMES = 120


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
    ap = argparse.ArgumentParser(description="Bat RQA dynamics by context (§34 #1)")
    ap.add_argument("--per-context", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species=DB.SPECIES).combined, dtype=np.float32
    )

    ann = DB.parse_annotations(DB.DATA / "annotations.csv")
    by_ctx: dict[str, list[str]] = defaultdict(list)
    for fn, (_em, _ad, ctx) in ann.items():
        by_ctx[ctx].append(fn)
    top = sorted(by_ctx, key=lambda c: -len(by_ctx[c]))[: args.top_k]
    print(f"top-{args.top_k} contexts: {[(c, len(by_ctx[c])) for c in top]}", flush=True)

    rng = random.Random(args.seed)
    per_ctx: dict[str, list[dict]] = defaultdict(list)
    for ctx in top:
        files = list(by_ctx[ctx])
        rng.shuffle(files)
        done = 0
        for fn in files:
            if done >= args.per_context:
                break
            try:
                audio, sr = load_audio(str(DB.DATA / "audio" / fn), target_sr=None)
            except Exception:  # noqa: BLE001
                continue
            if audio is None or len(audio) < int(0.2 * sr):
                continue
            t = frame_trajectory(audio, sr, fp_fn)
            if len(t) < 8:
                continue
            r = rqa_for_trajectory(t)
            per_ctx[ctx].append(
                {"DET": r["determinism"], "LAM": r["laminarity"], "diag_len": r["mean_diag_length"]}
            )
            done += 1
        print(f"  ctx {ctx}: {done} bouts", flush=True)

    # Report per-context means + Kruskal-Wallis across contexts.
    from scipy import stats  # noqa: PLC0415

    def _means(metric):
        groups = [np.asarray([b[metric] for b in per_ctx[c]]) for c in top if per_ctx[c]]
        means = {
            c: round(float(np.mean([b[metric] for b in per_ctx[c]])), 3) if per_ctx[c] else None
            for c in top
        }
        valid = [g for g in groups if len(g) >= 5]
        h, p = stats.kruskal(*valid) if len(valid) >= 2 else (float("nan"), float("nan"))
        return means, h, p

    det_means, det_h, det_p = _means("DET")
    lam_means, lam_h, lam_p = _means("LAM")
    print(
        f"\n{'=' * 90}\nBAT RQA BY CONTEXT (§34 #1) — graded dynamics DUSTED couldn't see\n{'=' * 90}"
    )
    print(f"  {'context':>8}  {'n':>4}  {'DET':>7}  {'LAM':>7}")
    for c in top:
        n = len(per_ctx[c])
        print(f"  {c:>8}  {n:>4}  {det_means[c]:>7.3f}  {lam_means[c]:>7.3f}")
    print(
        f"\n  Kruskal-Wallis across contexts: DET H={det_h:.2f} p={det_p:.4f} | "
        f"LAM H={lam_h:.2f} p={lam_p:.4f}"
    )
    if det_p < 0.05 or lam_p < 0.05:
        print("  ⇒ graded bat bouts have CONTEXT-DEPENDENT dynamical structure (RQA DET/LAM differ")
        print("    by context) — the dynamical-systems lens DUSTED lacked. Upgrades bat from")
        print("    'DUSTED struggled' to 'requires a dynamical lens, which disambiguates context.'")
    else:
        print("  ⇒ RQA DET/LAM do NOT differ significantly by context — graded dynamics are")
        print(
            "    context-invariant (still a valid dynamical finding, just not context-structured)."
        )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "per_context": {c: per_ctx[c] for c in top},
        "det_means": det_means,
        "lam_means": lam_means,
        "det_kruskal": {"H": det_h, "p": det_p},
        "lam_kruskal": {"H": lam_h, "p": lam_p},
    }
    with open(out_dir / "bat_rqa_by_context.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved outputs/bat_rqa_by_context.json")
    print("\nBAT RQA BY CONTEXT DONE")


if __name__ == "__main__":
    main()
