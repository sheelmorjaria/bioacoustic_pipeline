#!/usr/bin/env python3
"""Track C bat CCA: intrinsic acoustic<->social subspace alignment (no classification).

Phase 4 complement to ``bat_trackc_ablation.py``. The ablation script CONCATENATED
acoustic + social features for a supervised probe (4.7x lift over chance) — that tests
whether the modalities are useful TOGETHER for predicting Context, but says nothing about
whether they share an INTRINSIC common subspace. Canonical Correlation Analysis (CCA)
answers the unsupervised question directly: find linear projections of the acoustic and
social spaces that are maximally correlated, WITHOUT any classification target. High
canonical correlations mean the two modalities align on a shared subspace; low means they
are largely independent (the supervised lift came from a classifier exploiting residual
cue combinations, not from genuine modality alignment).

For each bat bout we compute:
  * acoustic:   whole-call acoustic fingerprint (~44-D physics+macro+micro)
  * social:     Emitter one-hot ‖ Context one-hot (the per-bout social metadata)

Then ``sklearn.cross_decomposition.CCA`` finds canonical variates; we report the first 5
canonical correlations + total R² (sum of squared canonical correlations, an overall
shared-subspace strength), and a permutation null on the leading canonical correlation.

Reuses ``dusted_bat.py`` infrastructure (parse_annotations, build_calls, tokenize are not
needed here — we want whole-call fingerprints + the per-bout Emitter/Context, not
sub-call units). Acoustic = whole-bout fingerprint; social = Emitter‖Context one-hot.

Run:  python scripts/bat_trackc_cca.py --n 1500

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
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

from dusted_bat import DATA, parse_annotations  # noqa: E402

SR = 250000
SPECIES = "bat"


def load_bouts(n, seed):
    """Per-bout whole-call fingerprint + (Emitter, Context) labels."""
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    ann = parse_annotations(DATA / "annotations.csv")
    files = sorted(ann.keys())
    rng.shuffle(files)
    out = []
    for fn in files:
        if len(out) >= n:
            break
        try:
            audio, sr = load_audio(str(DATA / "audio" / fn), target_sr=None)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.03 * sr):
            continue
        fp = np.asarray(afp.extract(audio, sr, species=SPECIES).combined, dtype=np.float32)
        if not np.any(fp):
            continue
        emitter, _, context = ann[fn]
        out.append({"fp": fp, "emitter": int(emitter), "context": int(context)})
        if (len(out) + 1) % 200 == 0:
            print(f"  extracted {len(out)}/{n} bouts", flush=True)
    print(f"  loaded {len(out)} bouts", flush=True)
    return out


def onehot(val, max_val):
    v = np.zeros(max_val + 1, dtype=np.float32)
    v[val] = 1.0
    return v


def cca(acoustic, social, n_components=5):
    """Fit CCA; return (canonical_correlations, X_scores, Y_scores, model)."""
    from sklearn.cross_decomposition import CCA  # noqa: PLC0415

    # n_components bounded by min(n_samples, n_features_acoustic, n_features_social)
    nc = int(min(n_components, acoustic.shape[0], acoustic.shape[1], social.shape[1]))
    if nc < 1:
        return np.zeros(0), None, None, None
    cca_obj = CCA(n_components=nc, scale=True, max_iter=2000)
    X_c, Y_c = cca_obj.fit_transform(acoustic, social)
    # post-hoc canonical correlations = Pearson corr of each canonical variate pair
    corrs = np.zeros(nc)
    for i in range(nc):
        x, y = X_c[:, i], Y_c[:, i]
        sx, sy = x.std(), y.std()
        if sx > 1e-9 and sy > 1e-9:
            corrs[i] = float(np.corrcoef(x, y)[0, 1])
    return corrs, X_c, Y_c, cca_obj


def perm_null(acoustic, social, n_components, n_perm, seed):
    """Permute rows of social, recompute leading canonical correlation. Phipson-Smyth p."""
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    n = acoustic.shape[0]
    null_lead = []
    for _ in range(n_perm):
        idx = list(range(n))
        rng.shuffle(idx)
        corrs, _, _, _ = cca(acoustic, social[idx], n_components=n_components)
        if corrs.size:
            null_lead.append(float(corrs[0]))
    null = np.asarray(null_lead, dtype=np.float64)
    return null


def main():
    ap = argparse.ArgumentParser(description="Bat Track C CCA: acoustic <-> social alignment")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading {args.n} bat bouts + whole-call fingerprints (250 kHz)...", flush=True)
    bouts = load_bouts(args.n, args.seed)
    if len(bouts) < 100:
        print("ERROR: too few bouts for CCA")
        sys.exit(1)

    acoustic = np.asarray([b["fp"] for b in bouts], dtype=np.float32)
    max_emit = max(b["emitter"] for b in bouts)
    max_ctx = max(b["context"] for b in bouts)
    social = np.asarray(
        [
            np.concatenate([onehot(b["emitter"], max_emit), onehot(b["context"], max_ctx)])
            for b in bouts
        ],
        dtype=np.float32,
    )

    # drop zero-variance columns (degenerate one-hot dims, etc.)
    ac_keep = acoustic.std(axis=0) > 1e-9
    so_keep = social.std(axis=0) > 1e-9
    acoustic = acoustic[:, ac_keep]
    social = social[:, so_keep]
    print(
        f"  acoustic: {acoustic.shape[0]} bouts x {acoustic.shape[1]}-D  |  "
        f"social: {social.shape[1]}-D ({max_emit + 1} emitters + {max_ctx + 1} contexts)",
        flush=True,
    )

    print(f"\nFitting CCA (acoustic <-> social), n_components={args.n_components}...", flush=True)
    corrs, X_c, Y_c, _ = cca(acoustic, social, n_components=args.n_components)
    if corrs.size == 0:
        print("ERROR: CCA produced no components")
        sys.exit(1)
    total_r2 = float(np.sum(corrs**2))
    top5 = corrs[: min(5, corrs.size)]

    print(f"\n{'=' * 70}")
    print(f"Bat Track C CCA — intrinsic acoustic<->social alignment (n={len(bouts)})")
    print(f"{'=' * 70}")
    print("  canonical correlations (first {}):".format(min(5, corrs.size)))
    for i, c in enumerate(top5):
        print(f"    rho_{i + 1} = {c:.4f}   (rho^2 = {c**2:.4f})")
    print(
        f"\n  total R^2 (sum of squared canonical correlations, all {corrs.size} comps): "
        f"{total_r2:.4f}"
    )
    print(f"  total R^2 over first {min(5, corrs.size)} comps: {float(np.sum(top5**2)):.4f}")

    print(f"\n  Permutation null on rho_1 (n_perm={args.n_perm})...", flush=True)
    null = perm_null(acoustic, social, args.n_components, args.n_perm, args.seed)
    ge = int(np.sum(null >= corrs[0]))
    p = (1 + ge) / (len(null) + 1)  # Phipson-Smyth
    print(f"    observed rho_1 = {corrs[0]:.4f}")
    print(f"    null rho_1     = {null.mean():.4f} +/- {null.std():.4f}")
    print(f"    p = {p:.4f}  ({ge}/{len(null)} null >= observed)")

    print(f"\n{'=' * 70}")
    print("=== Track C read ===")
    print("  Compare against the supervised-probe lift (bat_trackc_ablation.py ~4.7x over")
    print("  chance on Context from combined acoustic+social+behavior):")
    print(f"  - rho_1 = {corrs[0]:.4f}, total R^2 = {total_r2:.4f}, p = {p:.4f}")
    if corrs[0] > 0.4 and p < 0.05:
        print("    -> STRONG intrinsic acoustic<->social alignment: the modalities share a real")
        print("       common subspace, so the supervised lift reflects genuine alignment, not")
        print("       just a classifier exploiting residual cue combinations.")
    elif corrs[0] > 0.2 and p < 0.05:
        print("    -> MODERATE intrinsic alignment: a shared subspace exists but is weak; the")
        print("       supervised lift partly reflects genuine alignment, partly cue exploitation.")
    else:
        print("    -> WEAK/ABSENT intrinsic alignment: the acoustic and social spaces are largely")
        print("       independent. The supervised 4.7x lift came from a classifier exploiting")
        print("       residual cue COMBINATIONS, NOT from a genuine shared subspace — the two")
        print("       modalities contribute orthogonal information that the probe fuses.")
    print("BAT TRACK C CCA DONE")


if __name__ == "__main__":
    main()
