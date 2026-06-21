#!/usr/bin/env python3
"""PCEN + CNN candidate classifier for MeerKAT boundary segmentation.

The tabular calibrator (v2, F1~0.26 one-to-one) reached its ceiling — post-onset scalar
features didn't help. This replaces the tabular features with a learned spectrotemporal
shape detector: PCEN spectrogram patches around NBD proposals → small 2D CNN → P(call).

Architecture:
  NBD proposals → loose species-band junk filter →
  PCEN spectrogram patch (±1s, 64 mel × 100 frames) →
  3-layer 2D CNN → P(call) → threshold → one-to-one F1

PCEN (per-channel energy normalization) performs adaptive gain control in the
time-frequency domain — the noise-floor normalization that the FP taxonomy identified
as the missing ingredient (60% off-band + 22% broadband FPs).

Usage:
  python scripts/train_pcen_cnn_calibrator.py --n-train 400 --n-test 150

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import CALIBRATOR_PROPOSAL  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E202
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/home/sheel/birdsong_analysis/data/MeerKAT_10s_2024-06-12")
LBL = DATA / "lbl" / "08000Hz"
WAV = DATA / "wav" / "08000Hz"
BAND = (1000, 4000)
N_MELS = 64
HOP = 160  # 20ms at 8kHz
N_FFT = 512
PATCH_FRAMES = 100  # ±1s context at 20ms hop
SR = 8000


def compute_pcen(audio, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    """PCEN spectrogram: mel-magnitude STFT → PCEN adaptive gain normalization.

    PCEN formula per band: gain = (s / (eps + smoothed_s))^alpha; out = (gain + delta)^r
    where smoothed_s is an IIR low-pass. Implements the standard bioacoustic PCEN
    (Lostanlen et al.) without torchaudio dependency.
    """
    import torchaudio  # noqa: PLC0415

    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    window = torch.hann_window(n_fft)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = stft.abs().pow(2).numpy()  # (n_fft//2+1, n_frames)
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sr / 2, n_mels=n_mels, sample_rate=sr,
    ).numpy()
    mel = fb.T @ mag  # (n_mels, n_frames)

    # PCEN parameters
    alpha, delta, r, eps, t_const = 0.98, 2.0, 0.5, 1e-6, 0.06
    b = 1.0 - np.exp(-hop / (t_const * sr))  # IIR smoothing coefficient
    smoothed = np.zeros_like(mel)
    for t in range(mel.shape[1]):
        smoothed[:, t] = (1 - b) * (smoothed[:, t - 1] if t > 0 else 0) + b * mel[:, t]
    gain = mel / (eps + smoothed) ** alpha
    pcen = np.maximum(0.0, gain + delta) ** r
    return pcen.astype(np.float32)  # (n_mels, n_frames)


def extract_patch(pcen_spec, frame_idx, patch_frames=PATCH_FRAMES):
    """Extract a centered (n_mels, patch_frames) patch; zero-pad at edges."""
    half = patch_frames // 2
    n_frames = pcen_spec.shape[1]
    lo, hi = frame_idx - half, frame_idx + half
    pad_lo, pad_hi = max(0, -lo), max(0, hi - n_frames)
    lo, hi = max(0, lo), min(n_frames, hi)
    patch = pcen_spec[:, lo:hi]
    if pad_lo or pad_hi:
        patch = np.pad(patch, ((0, 0), (pad_lo, pad_hi)), mode="constant")
    return patch


class CallCNN(nn.Module):
    """Small 2D CNN: PCEN patch → P(call)."""

    def __init__(self, n_mels=N_MELS, patch_frames=PATCH_FRAMES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, n_mels, frames)
        x = self.conv(x).squeeze(-1).squeeze(-1)  # (B, 128)
        return self.fc(x).squeeze(-1)  # (B,)


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
        stem = Path(f).stem
        if st.size and (WAV / f"{stem}.wav").exists():
            out.append((stem, st * 1000.0))
    return out


def build_dataset(clips, profile, label_tol=100.0, max_patches=8000):
    """Extract PCEN patches at NBD proposals + binary labels (call/non-call)."""
    patches, labels = [], []
    hop_ms = profile.get("hop_ms", 10.0)
    for ci, (stem, gt) in enumerate(clips):
        if len(patches) >= max_patches:
            break
        try:
            audio, sr = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
        except Exception:
            continue
        sig = compute_signals(audio, sr, hop_ms=hop_ms)
        props = generate_proposals(sig, **CALIBRATOR_PROPOSAL)
        if not props:
            continue
        pcen = compute_pcen(audio, sr)
        gt_arr = np.asarray(gt)
        for p in props:
            frame_idx = int(p.time_ms / (HOP / SR * 1000))
            patch = extract_patch(pcen, frame_idx)
            is_call = 1 if (gt_arr.size and np.min(np.abs(gt_arr - p.time_ms)) < label_tol) else 0
            patches.append(patch)
            labels.append(is_call)
        if (ci + 1) % 100 == 0:
            print(f"  built {ci + 1} clips, {len(patches)} patches", flush=True)
    X = np.stack(patches)
    y = np.array(labels, dtype=np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser(description="PCEN + CNN candidate classifier for MeerKAT")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance-ms", type=float, default=100.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    profile = load_species_profiles()["meerkat"]
    clips = sample_clips(args.n_train + args.n_test, args.seed)
    train_clips, test_clips = clips[: args.n_train], clips[args.n_train :]
    print(f"{len(train_clips)} train / {len(test_clips)} test", flush=True)

    # ---- build training dataset ----
    print("building PCEN patches (train)...", flush=True)
    Xtr, ytr = build_dataset(train_clips, profile, label_tol=args.tolerance_ms)
    print(f"train: {Xtr.shape} pos_rate={ytr.mean():.3f}", flush=True)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)])
    ds = TensorDataset(Xtr_t, ytr_t)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ---- train CNN ----
    model = CallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/{args.epochs} loss={total_loss/len(loader):.4f}", flush=True)

    # ---- evaluate: raw NBD vs CNN-calibrated (one-to-one F1) ----
    model.eval()
    # find volume-matched threshold on train predictions
    with torch.no_grad():
        train_proba = torch.sigmoid(model(Xtr_t.to(device))).cpu().numpy()
    total_gt = sum(len(g) for _, g in train_clips)
    thr = float(np.sort(train_proba)[::-1][min(total_gt, len(train_proba) - 1)])
    print(f"volume-matched threshold: {thr:.4f}", flush=True)

    raw_tp = raw_fp = raw_fn = cnn_tp = cnn_fp = cnn_fn = 0
    n_raw = n_cnn = n_gt = 0
    hop_ms = profile.get("hop_ms", 10.0)
    for stem, gt in test_clips:
        try:
            audio, sr = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
        except Exception:
            continue
        sig = compute_signals(audio, sr, hop_ms=hop_ms)
        # raw NBD
        raw_props = generate_proposals(
            sig, floor_energy=profile["energy_threshold"],
            floor_flux=profile["spectral_threshold"], debounce_ms=profile["debounce_ms"],
        )
        raw_pred = [p.time_ms for p in raw_props]
        rr = prf_one_to_one(raw_pred, list(gt), tol=args.tolerance_ms)
        raw_tp += rr.tp; raw_fp += rr.fp; raw_fn += rr.fn
        n_raw += len(raw_pred); n_gt += len(gt)
        # CNN-calibrated
        props = generate_proposals(sig, **CALIBRATOR_PROPOSAL)
        if not props:
            continue
        pcen = compute_pcen(audio, sr)
        patches = []
        times = []
        for p in props:
            frame_idx = int(p.time_ms / (HOP / SR * 1000))
            patches.append(extract_patch(pcen, frame_idx))
            times.append(p.time_ms)
        if not patches:
            continue
        with torch.no_grad():
            proba = torch.sigmoid(model(torch.from_numpy(np.stack(patches)).to(device))).cpu().numpy()
        cnn_pred = [t for t, p in zip(times, proba) if p >= thr]
        cr = prf_one_to_one(cnn_pred, list(gt), tol=args.tolerance_ms)
        cnn_tp += cr.tp; cnn_fp += cr.fp; cnn_fn += cr.fn
        n_cnn += len(cnn_pred)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-8), p, r

    raw_f1, raw_p, raw_r = prf(raw_tp, raw_fp, raw_fn)
    cnn_f1, cnn_p, cnn_r = prf(cnn_tp, cnn_fp, cnn_fn)
    print(f"\n=== PCEN + CNN Candidate Classifier (MeerKAT, {len(test_clips)} test, one-to-one) ===")
    print(f"  raw NBD:       F1={raw_f1:.3f} P={raw_p:.3f} R={raw_r:.3f} p/g={n_raw/max(n_gt,1):.1f}")
    print(f"  CNN calibrated: F1={cnn_f1:.3f} P={cnn_p:.3f} R={cnn_r:.3f} p/g={n_cnn/max(n_gt,1):.1f}")
    print(f"  (v2 tabular ref: F1~0.26 one-to-one)")
    print(f"  (v2 old metric: F1=0.323)")
    print("CNN DONE")


if __name__ == "__main__":
    main()
