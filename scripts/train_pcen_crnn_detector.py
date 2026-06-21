#!/usr/bin/env python3
"""PCEN + CRNN frame-level detector for MeerKAT boundary segmentation.

Improvements over the first CNN attempt:
  1. More training data (2000 clips, not 400)
  2. LR schedule (cosine annealing) + 50 epochs
  3. PCEN parameter tuning (species-appropriate time_constant)
  4. CRNN architecture (conv → BiLSTM → FC for temporal continuity)
  5. Frame-level sliding detector (no NBD dependency — full P(call) timeline)

The frame-level approach:
  audio → PCEN spectrogram → CRNN → P(call) per frame →
  hysteresis (enter/exit) → connected components → onset events → one-to-one F1

Training uses GT call SPANS (start_time to end_time), not just onsets — every frame
inside a call is labeled positive, giving ~100× more training signal.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/home/sheel/birdsong_analysis/data/MeerKAT_10s_2024-06-12")
LBL = DATA / "lbl" / "08000Hz"
WAV = DATA / "wav" / "08000Hz"
SR = 8000
N_FFT = 512
HOP = 160  # 20ms
N_MELS = 64


# ─── PCEN ────────────────────────────────────────────────────────────────────

def compute_pcen(audio, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS,
                 t_const=0.08, alpha=0.98, delta=2.0, r=0.5):
    """Manual PCEN: mel-STFT → IIR smoothing → adaptive gain → power compression."""
    import torchaudio  # noqa: PLC0415

    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    window = torch.hann_window(n_fft)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = stft.abs().pow(2).numpy()
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sr / 2, n_mels=n_mels, sample_rate=sr,
    ).numpy()
    mel = fb.T @ mag  # (n_mels, n_frames)

    eps = 1e-6
    b = 1.0 - np.exp(-hop / (t_const * sr))
    smoothed = np.zeros_like(mel)
    for t in range(mel.shape[1]):
        smoothed[:, t] = (1 - b) * (smoothed[:, t - 1] if t > 0 else 0) + b * mel[:, t]
    gain = mel / (eps + smoothed) ** alpha
    pcen = np.maximum(0.0, gain + delta) ** r
    return pcen.astype(np.float32)  # (n_mels, n_frames)


# ─── CRNN model ──────────────────────────────────────────────────────────────

class CRNNDetector(nn.Module):
    """Conv2D feature extractor → BiLSTM temporal model → per-frame P(call)."""

    def __init__(self, n_mels=N_MELS, hidden=128, n_lstm=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # pool freq only — keep temporal resolution
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # (hidden, 1, T) — collapse freq
        )
        self.lstm = nn.LSTM(hidden, hidden, n_lstm, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        c = self.conv(x)  # (B, hidden, 1, T)
        c = c.squeeze(2).permute(0, 2, 1)  # (B, T, hidden)
        lstm_out, _ = self.lstm(c)  # (B, T, hidden*2)
        return self.fc(lstm_out).squeeze(-1)  # (B, T)


# ─── Data ────────────────────────────────────────────────────────────────────

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
            et = np.asarray(h["end_time_lbl"][:]).ravel()
            lbl = np.asarray(h["lbl"][:]).ravel()
        st = st[np.isfinite(st)]
        stem = Path(f).stem
        if st.size and (WAV / f"{stem}.wav").exists():
            spans = []
            for s, e in zip(st, et):
                spans.append((float(s), float(e) if np.isfinite(e) else float(s) + 0.08))
            out.append((stem, spans))
    return out


def clip_to_tensors(stem, spans, pcen_params):
    """Load clip → PCEN → frame-level labels (call spans → binary mask)."""
    try:
        audio, sr = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
    except Exception:
        return None, None
    pcen = compute_pcen(audio, sr, **pcen_params)  # (n_mels, n_frames)
    n_frames = pcen.shape[1]
    frame_dur_s = HOP / SR
    # frame-level label: 1 if any call span covers this frame
    labels = np.zeros(n_frames, dtype=np.float32)
    for s, e in spans:
        fi = int(s / frame_dur_s)
        fj = int(e / frame_dur_s) + 1
        labels[fi:min(fj, n_frames)] = 1.0
    return pcen, labels


def collate_batches(pcen_list, label_list, max_frames=500):
    """Pad/truncate to fixed length; return (B, 1, n_mels, T) + (B, T)."""
    B = len(pcen_list)
    T = max_frames
    X = np.zeros((B, 1, N_MELS, T), dtype=np.float32)
    Y = np.zeros((B, T), dtype=np.float32)
    for i, (pcen, lab) in enumerate(zip(pcen_list, label_list)):
        t = min(pcen.shape[1], T)
        X[i, 0, :, :t] = pcen[:, :t]
        Y[i, :t] = lab[:t]
    return torch.from_numpy(X), torch.from_numpy(Y)


# ─── Hysteresis + connected components ───────────────────────────────────────

def proba_to_events(proba, enter_thr=0.5, exit_thr=0.2, min_frames=2):
    """P(call) timeline → onset events via hysteresis + connected components.

    Enter state when proba >= enter_thr; exit when proba < exit_thr.
    Onset = first frame of each event (converted to ms).
    """
    in_call = False
    onsets = []
    for i, p in enumerate(proba):
        if not in_call and p >= enter_thr:
            in_call = True
            start = i
        elif in_call and p < exit_thr:
            in_call = False
            if i - start >= min_frames:
                onsets.append(start * (HOP / SR * 1000))
    if in_call and len(proba) - start >= min_frames:
        onsets.append(start * (HOP / SR * 1000))
    return onsets


def tune_hysteresis(train_results, train_gt_list, tol=100.0):
    """Grid-search enter/exit thresholds to maximize train one-to-one F1."""
    best_f1, best_enter, best_exit = -1, 0.5, 0.2
    for enter in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for exit in [0.1, 0.15, 0.2, 0.3]:
            tp = fp = fn = 0
            for proba, gt in zip(train_results, train_gt_list):
                events = proba_to_events(proba, enter, exit)
                r = prf_one_to_one(events, list(gt), tol=tol)
                tp += r.tp; fp += r.fp; fn += r.fn
            p = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
            f1 = 2 * p * r_ / max(p + r_, 1e-8)
            if f1 > best_f1:
                best_f1, best_enter, best_exit = f1, enter, exit
    return best_enter, best_exit, best_f1


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance-ms", type=float, default=100.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # PCEN parameters (tuned: slightly longer time constant for meerkat calls)
    pcen_params = {"t_const": 0.08, "alpha": 0.98, "delta": 2.0, "r": 0.5}

    clips = sample_clips(args.n_train + args.n_test, args.seed)
    train_clips, test_clips = clips[: args.n_train], clips[args.n_train :]
    print(f"{len(train_clips)} train / {len(test_clips)} test", flush=True)

    # ---- build train data (frame-level) ----
    print("building PCEN + frame labels (train)...", flush=True)
    train_pcen, train_labels, train_gt = [], [], []
    for ci, (stem, spans) in enumerate(train_clips):
        pcen, labs = clip_to_tensors(stem, spans, pcen_params)
        if pcen is None:
            continue
        train_pcen.append(pcen)
        train_labels.append(labs)
        train_gt.append(np.array([s * 1000 for s, _ in spans]))
        if (ci + 1) % 500 == 0:
            pos_rate = np.mean([l.mean() for l in train_labels])
            print(f"  {ci+1}/{len(train_clips)} clips, pos_rate={pos_rate:.3f}", flush=True)
    pos_rate = np.mean([l.mean() for l in train_labels])
    print(f"train: {len(train_pcen)} clips, pos_rate={pos_rate:.3f}", flush=True)

    # ---- train CRNN ----
    model = CRNNDetector().to(device)
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    indices = list(range(len(train_pcen)))
    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(indices)
        total_loss = 0
        n_batches = 0
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            bp = [train_pcen[j] for j in batch_idx]
            bl = [train_labels[j] for j in batch_idx]
            X, Y = collate_batches(bp, bl)
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            logits = model(X)  # (B, T)
            loss = crit(logits, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{args.epochs} loss={total_loss/n_batches:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

    # ---- tune hysteresis on train ----
    model.eval()
    print("tuning hysteresis on train...", flush=True)
    train_results = []
    with torch.no_grad():
        for i in range(0, len(train_pcen), args.batch_size):
            bp = train_pcen[i : i + args.batch_size]
            X, _ = collate_batches(bp, [np.zeros(1)] * len(bp))
            logits = model(X.to(device))
            proba = torch.sigmoid(logits).cpu().numpy()
            for j in range(len(bp)):
                t = min(bp[j].shape[1], X.shape[3])
                train_results.append(proba[j, :t])
    best_enter, best_exit, train_f1 = tune_hysteresis(
        train_results, train_gt, tol=args.tolerance_ms)
    print(f"hysteresis: enter={best_enter} exit={best_exit} train_F1={train_f1:.3f}", flush=True)

    # ---- evaluate on test ----
    print("evaluating test...", flush=True)
    raw_tp = raw_fp = raw_fn = crnn_tp = crnn_fp = crnn_fn = 0
    n_raw = n_crnn = n_gt = 0
    for ci, (stem, spans) in enumerate(test_clips):
        try:
            audio, sr = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
        except Exception:
            continue
        gt = [s * 1000 for s, _ in spans]
        n_gt += len(gt)

        # raw NBD baseline
        from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
        from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: PLC0415
        profile = load_species_profiles()["meerkat"]
        sig = compute_signals(audio, sr, hop_ms=profile["hop_ms"])
        raw_props = generate_proposals(
            sig, floor_energy=profile["energy_threshold"],
            floor_flux=profile["spectral_threshold"], debounce_ms=profile["debounce_ms"],
        )
        raw_pred = [p.time_ms for p in raw_props]
        rr = prf_one_to_one(raw_pred, gt, tol=args.tolerance_ms)
        raw_tp += rr.tp; raw_fp += rr.fp; raw_fn += rr.fn; n_raw += len(raw_pred)

        # CRNN frame-level detector
        pcen = compute_pcen(audio, sr, **pcen_params)
        if pcen.shape[1] == 0:
            continue
        X = torch.from_numpy(pcen).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, T)
        with torch.no_grad():
            proba = torch.sigmoid(model(X)).cpu().numpy()[0]  # (T,)
        events = proba_to_events(proba, best_enter, best_exit)
        cr = prf_one_to_one(events, gt, tol=args.tolerance_ms)
        crnn_tp += cr.tp; crnn_fp += cr.fp; crnn_fn += cr.fn; n_crnn += len(events)

        if (ci + 1) % 100 == 0:
            print(f"  test {ci+1}/{len(test_clips)}", flush=True)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-8), p, r

    rf, rp, rr_ = prf(raw_tp, raw_fp, raw_fn)
    cf, cp_, cr_ = prf(crnn_tp, crnn_fp, crnn_fn)
    print(f"\n=== PCEN + CRNN Frame-Level Detector (MeerKAT, {len(test_clips)} test, one-to-one) ===")
    print(f"  raw NBD:         F1={rf:.3f} P={rp:.3f} R={rr_:.3f} p/g={n_raw/max(n_gt,1):.1f}")
    print(f"  CRNN detector:   F1={cf:.3f} P={cp_:.3f} R={cr_:.3f} p/g={n_crnn/max(n_gt,1):.1f}")
    print(f"  (v2 tabular ref: F1~0.26 one-to-one)")
    print(f"  (CNN v1 ref:     F1=0.146)")
    print(f"  config: {args.n_train} train clips, {args.epochs} epochs, "
          f"enter={best_enter} exit={best_exit}, pcen t_const={pcen_params['t_const']}")
    print("CRNN DONE")


if __name__ == "__main__":
    main()
