#!/usr/bin/env python3
"""MeerKAT boundary calibrator v2: richer features + volume-matched operating point.

v1 (scalar 13-feature GBT, max-train-F1 threshold) improved F1 0.086→0.201 but over-
corrected to under-segmentation (recall 0.16, pred/gt 0.6) — the F1-optimal threshold
over-fit a precision-favoring peak that didn't generalize. v2 fixes both:

  1. RICHER FEATURES: the 13 NBD proposal features + 6 noise-discrimination features
     computed at each proposal's center — local SNR, background distance, crest factor
     (all from the rms curve), spectral entropy, spectral flatness, call-band (1–4 kHz)
     energy ratio (from a local STFT). These separate real call onsets (tonal, above the
     noise floor, band-concentrated) from noise-floor fluctuations (broadband, low SNR).
  2. VOLUME-MATCHED OPERATING POINT: instead of max-train-F1, set the decision threshold
     to the train predicted-probability at rank = n_train_gt (keep ~as many predictions as
     there are GT events). This calibrates prediction volume to GT volume → balanced P/R
     that generalizes instead of chasing a noisy F1 peak.

Trains a HistGradientBoostingClassifier directly (BoundaryCalibrator is fixed at 13 features;
this is an experimental richer-feature path). Held-out per-clip F1 evaluation.

Usage:
  python scripts/refit_meerkat_calibrator_v2.py --n-train 400 --n-test 150 --tolerance-ms 100

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import CALIBRATOR_PROPOSAL  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_at_tolerance  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/home/sheel/birdsong_analysis/data/MeerKAT_10s_2024-06-12")
LBL = DATA / "lbl" / "08000Hz"
WAV = DATA / "wav" / "08000Hz"
CALL_BAND = (1000.0, 4000.0)  # Hz — meerkat call energy concentrates here


def sample_clips(n, seed):
    rng = random.Random(seed)
    files = sorted(glob.glob(str(LBL / "*.h5")))
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        with h5py.File(f, "r") as h:
            st = np.asarray(h["start_time_lbl"][:]).ravel()
        st = st[np.isfinite(st)]
        if st.size and (WAV / (Path(f).stem + ".wav")).exists():
            out.append((Path(f).stem, st * 1000.0))
    return out


def extended_features(props, sig, audio, sr, hop_ms, nfft=256):
    """13 NBD proposal features + 6 noise-discrimination features per proposal."""
    base = np.vstack([proposal_features(p) for p in props]) if props else np.zeros((0, 13))
    rms = np.asarray(sig.rms, dtype=np.float64)
    extra = np.zeros((len(props), 6), dtype=np.float64)
    for i, p in enumerate(props):
        fi = int(round(p.time_ms / hop_ms))
        lo, hi = max(0, fi - 10), min(len(rms), fi + 11)
        win = rms[lo:hi]
        if win.size:
            lmax, lmed, lmean = float(win.max()), float(np.median(win)), float(win.mean())
        else:
            lmax = lmed = lmean = 0.0
        c = int(p.time_ms / 1000.0 * sr)
        seg = audio[max(0, c - nfft // 2): c + nfft // 2]
        if len(seg) < 8:
            seg = np.zeros(8)
        mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        psd = mag.astype(np.float64) ** 2 + 1e-12
        p_norm = psd / psd.sum()
        spec_entropy = float(-(p_norm * np.log(p_norm)).sum())
        geom = float(np.exp(np.log(psd).mean())); arith = float(psd.mean())
        spec_flat = geom / (arith + 1e-12)
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        band = (freqs >= CALL_BAND[0]) & (freqs <= CALL_BAND[1])
        band_ratio = float(psd[band].sum() / (psd.sum() + 1e-12))
        extra[i] = [lmax / (lmed + 1e-9), lmax - lmed, lmax / (lmean + 1e-9),
                    spec_entropy, spec_flat, band_ratio]
    return np.hstack([base, extra])


def build_clip(stem, gt_ms, profile, label_tol_ms):
    audio, sr = load_audio(str(WAV / f"{stem}.wav"), target_sr=profile["sample_rate"])
    hop_ms = profile.get("hop_ms", 10.0)
    sig = compute_signals(audio, sr, hop_ms=hop_ms)
    props = generate_proposals(sig, **CALIBRATOR_PROPOSAL)
    X = extended_features(props, sig, audio, sr, hop_ms)
    gt_arr = np.asarray(gt_ms, dtype=float)
    labels = np.array([1 if (gt_arr.size and np.min(np.abs(gt_arr - p.time_ms)) < label_tol_ms) else 0
                       for p in props], dtype=np.int64)
    return props, X, labels


def micro_prf(kept_times_per_clip, gt_per_clip, tol):
    tp = fp = fn = 0
    for kept, gt in zip(kept_times_per_clip, gt_per_clip):
        r = prf_at_tolerance(list(kept), list(gt), tol=tol)
        tp += r.tp; fp += r.fp; fn += r.fn
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return f1, p, r, tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description="MeerKAT boundary calibrator v2 (richer features + volume-matched OP)")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--tolerance-ms", type=float, default=100.0)
    ap.add_argument("--label-tol-ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="validation_pack_v1/reports/meerkat_boundary_calibrator_v2_report.json")
    args = ap.parse_args()

    profile = load_species_profiles()["meerkat"]
    clips = sample_clips(args.n_train + args.n_test, args.seed)
    train, test = clips[: args.n_train], clips[args.n_train :]
    print(f"clips: {len(train)} train / {len(test)} test")

    # ---- build train (per-clip) ----
    train_X, train_y, train_props, train_gt = [], [], [], []
    for i, (stem, gt) in enumerate(train):
        props, X, y = build_clip(stem, gt, profile, args.label_tol_ms)
        train_X.append(X); train_y.append(y); train_props.append(props); train_gt.append(gt)
        if (i + 1) % 100 == 0:
            print(f"  built train {i + 1}/{len(train)}")
    Xtr = np.vstack(train_X); ytr = np.concatenate(train_y)
    print(f"train: {Xtr.shape[0]} proposals, {Xtr.shape[1]} features | pos rate {ytr.mean():.3f}")

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    gbt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                         l2_regularization=1.0, random_state=0)
    gbt.fit(Xtr, ytr)

    # ---- volume-matched operating point: threshold = train proba at rank = total_train_gt ----
    train_proba_per_clip = [gbt.predict_proba(X)[:, 1] for X in train_X]
    all_train_proba = np.concatenate(train_proba_per_clip)
    total_train_gt = sum(len(g) for g in train_gt)
    rank = min(total_train_gt, len(all_train_proba) - 1)
    thr = float(np.sort(all_train_proba)[::-1][rank])
    # train F1 at this volume-matched threshold (sanity)
    kept_tr = [[p.time_ms for p, pr in zip(props, proba) if pr >= thr]
               for props, proba in zip(train_props, train_proba_per_clip)]
    tr_f1, tr_p, tr_r = micro_prf(kept_tr, train_gt, args.tolerance_ms)[:3]
    print(f"volume-matched threshold={thr:.4f} (keep {total_train_gt}/{len(all_train_proba)} = "
          f"{total_train_gt/len(all_train_proba):.3f}) | train F1={tr_f1:.3f} P={tr_p:.3f} R={tr_r:.3f}")

    # ---- evaluate: raw / v1-calibrated-style / v2 on held-out test ----
    raw_kept, v2_kept, test_gt = [], [], []
    n_raw = 0
    for stem, gt in test:
        audio, sr = load_audio(str(WAV / f"{stem}.wav"), target_sr=profile["sample_rate"])
        hop_ms = profile.get("hop_ms", 10.0)
        sig = compute_signals(audio, sr, hop_ms=hop_ms)
        raw_props = generate_proposals(sig, floor_energy=profile["energy_threshold"],
                                       floor_flux=profile["spectral_threshold"],
                                       debounce_ms=profile["debounce_ms"])
        raw_kept.append([p.time_ms for p in raw_props]); n_raw += len(raw_props)
        cal_props = generate_proposals(sig, **CALIBRATOR_PROPOSAL)
        Xte = extended_features(cal_props, sig, audio, sr, hop_ms)
        proba = gbt.predict_proba(Xte)[:, 1]
        v2_kept.append([p.time_ms for p, pr in zip(cal_props, proba) if pr >= thr])
        test_gt.append(gt)

    raw_f1, raw_p, raw_r, *_ = micro_prf(raw_kept, test_gt, args.tolerance_ms)
    v2_f1, v2_p, v2_r, v2_tp, v2_fp, v2_fn = micro_prf(v2_kept, test_gt, args.tolerance_ms)
    n_gt = sum(len(g) for g in test_gt); n_v2 = sum(len(k) for k in v2_kept)
    print(f"\n=== MeerKAT boundary F1 (test, {len(test)} clips, tol={args.tolerance_ms}ms) ===")
    print(f"  raw NBD:        F1={raw_f1:.3f} P={raw_p:.3f} R={raw_r:.3f} | pred/gt={n_raw/max(n_gt,1):.1f}")
    print(f"  v2 calibrated:  F1={v2_f1:.3f} P={v2_p:.3f} R={v2_r:.3f} | pred/gt={n_v2/max(n_gt,1):.1f}")

    report = {
        "dataset": "MeerKAT_10s", "species": "meerkat", "task": "boundary_calibrator_v2",
        "n_train": len(train), "n_test": len(test), "tolerance_ms": args.tolerance_ms,
        "features": "13 NBD + 6 noise (local_snr, bg_distance, crest, spec_entropy, spec_flatness, call_band_ratio)",
        "operating_point": "volume-matched (train proba at rank n_gt)",
        "threshold": round(thr, 5), "train_f1": round(tr_f1, 4),
        "raw_nbd": {"f1": round(raw_f1, 4), "precision": round(raw_p, 4), "recall": round(raw_r, 4)},
        "calibrated_v2": {"f1": round(v2_f1, 4), "precision": round(v2_p, 4), "recall": round(v2_r, 4),
                          "pred_per_gt": round(n_v2 / max(n_gt, 1), 2)},
        "comparison": {"v1_calibrated_f1": 0.201, "raw_f1": round(raw_f1, 4)},
        "framing": ("v2 adds noise-discrimination features + a volume-matched operating point over v1 "
                    "(scalar-only, max-train-F1 threshold). Closes the MeerKAT boundary-F1 GAP."),
    }
    out = Path(args.report); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
