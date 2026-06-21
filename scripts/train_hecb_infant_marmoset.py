#!/usr/bin/env python3
"""HECB-Seg on InfantMarmosetsVox: cross-dataset validation (step 5).

Adapts the PCEN+CRNN+HECB stack to a NEW species (infant marmosets, 44.1 kHz)
with per-segment call-type + caller-ID annotations. Tests whether the HECB
architecture (validated on MeerKAT) generalizes:

  Stage 1: PCEN+CRNN call detector → call events (onset/offset F1)
  Stage 2-4: typed classifier + hecb_decode → internal boundaries
  Eval: onset F1 + call-type recovery (whole/sub/pooled ablation)

Dataset: InfantMarmosetsVox
  350 × 10-min wavs (44.1 kHz mono), 5 twin pairs
  73k vocalization segments, 11 call types, 10 caller identities
  labels.csv: filename,start,end,duration,calltype,caller

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
from bioacoustic_pipeline.representation.subcall import (  # noqa: E402
    build_subcall_representations,
    format_ablation_table,
    whole_sub_pooled_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import fit_hecb_priors, hecb_decode  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

import train_hecb_seg as H  # reuse CRNNDetector arch + helpers  # noqa: E402

DATA = Path("/home/sheel/InfantMarmosetsVox")
SR = 44100
N_FFT, HOP, N_MELS = 1024, 441, 64
FRAME_MS = HOP / SR * 1000.0
CALL_BAND = (200, 15000)  # Hz — broad band for marmoset calls
MAX_FRAMES = 2000  # ~20s clips (enough for 10-min files — we chunk)
PAD_MS = 50.0
_PROPOSAL = {"floor_energy": 0.04, "floor_flux": 0.02, "debounce_ms": 25.0}

CALL_TYPE_NAMES = {
    0: "Peep(Pre-Phee)",
    1: "Phee",
    2: "Twitter",
    3: "Trill",
    4: "Trillphee",
    5: "TsikTse",
    6: "Egg",
    7: "Pheecry",
    8: "TrllTwitter",
    9: "Pheetwitter",
    10: "Peep",
    11: "Silence",
    12: "Noise",
}
N_CALL_TYPES = 11  # 0..10 are real calls


def _pcen_44k(audio, sr=SR):
    """Manual PCEN at 44.1 kHz."""
    import torchaudio  # noqa: PLC0415

    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    stft = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP, window=torch.hann_window(N_FFT), return_complex=True
    )
    mag = stft.abs().pow(2).numpy()
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1, f_min=0.0, f_max=sr / 2, n_mels=N_MELS, sample_rate=sr
    ).numpy()
    mel = fb.T @ mag
    b = 1.0 - np.exp(-HOP / (0.08 * sr))
    sm = np.zeros_like(mel)
    for t in range(mel.shape[1]):
        sm[:, t] = (1 - b) * (sm[:, t - 1] if t > 0 else 0) + b * mel[:, t]
    return (np.maximum(0.0, mel / (1e-6 + sm) ** 0.98 + 2.0) ** 0.5).astype(np.float32)


def _crnn_proba(model, audio, device, max_frames=MAX_FRAMES):
    model.eval()
    pcen = _pcen_44k(audio)
    nf = pcen.shape[1]
    proba = np.zeros(nf, dtype=np.float32)
    for start in range(0, nf, max_frames):
        chunk = pcen[:, start : start + max_frames]
        t = chunk.shape[1]
        x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            proba[start : start + t] = torch.sigmoid(model(x)).cpu().numpy()[0]
    return proba


def load_labels(csv_path):
    """labels.csv → {filename: [(start_s, end_s, calltype, caller), ...]}"""
    by_file = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            start = float(row["start"])
            end = float(row["end"])
            ct = int(row["calltype"])
            caller = int(row["caller"])
            by_file[fn].append((start, end, ct, caller))
    return dict(by_file)


def find_wav(filename, data_root=DATA):
    """filename (e.g. '20160907_Twin1_marmoset1') → wav path."""
    # Extract twin number from filename
    parts = filename.lower().split("_")
    twin = next((p for p in parts if p.startswith("twin")), None)
    if twin:
        twin_num = twin.replace("twin", "")
        for d in sorted(data_root.glob(f"data/twin_{twin_num}")):
            wav = d / f"{filename}.wav"
            if wav.exists():
                return wav
    # fallback: search all twin dirs
    for d in sorted(data_root.glob("data/twin_*")):
        wav = d / f"{filename}.wav"
        if wav.exists():
            return wav
    return None


def get_call_spans(labels):
    """Filter to vocalization segments (calltype 0-10)."""
    return [(s, e, ct, caller) for s, e, ct, caller in labels if ct < N_CALL_TYPES]


def sample_files(n, seed):
    """Sample n unique filenames from labels.csv."""
    rng = random.Random(seed)
    all_labels = load_labels(DATA / "labels.csv")
    files = sorted(all_labels.keys())
    rng.shuffle(files)
    out = []
    for fn in files:
        if len(out) >= n:
            break
        wav = find_wav(fn)
        if wav and get_call_spans(all_labels[fn]):
            out.append((fn, wav, all_labels[fn]))
    return out


def train_crnn(train_files, device, epochs, max_frames=MAX_FRAMES):
    """Stage 1: train PCEN+CRNN call detector."""
    train_data = []
    for fn, wav, labels in train_files:
        try:
            audio, _ = load_audio(str(wav), target_sr=SR)
        except Exception:
            continue
        call_spans = get_call_spans(labels)
        if not call_spans:
            continue
        pcen = _pcen_44k(audio)
        nf = pcen.shape[1]
        lab = np.zeros(nf, dtype=np.float32)
        for s, e, _ct, _caller in call_spans:
            lab[int(s * 1000 / FRAME_MS) : min(int(e * 1000 / FRAME_MS) + 1, nf)] = 1.0
        # Chunk into max_frames windows
        for start in range(0, nf, max_frames):
            end = min(start + max_frames, nf)
            if end - start < 100:  # skip tiny chunks
                continue
            train_data.append((pcen[:, start:end], lab[start:end]))

    pos_rate = float(np.mean([l.mean() for _, l in train_data]))
    print(f"  CRNN train: {len(train_data)} chunks, pos_rate={pos_rate:.3f}", flush=True)

    model = H.CRNNDetector().to(device)
    pw = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    BS, MF = 4, max_frames
    for ep in range(epochs):
        model.train()
        random.Random(42 + ep).shuffle(train_data)
        tl, nb = 0.0, 0
        for i in range(0, len(train_data), BS):
            batch = train_data[i : i + BS]
            B = len(batch)
            T = min(pc.shape[1] for pc, _ in batch)
            X = np.zeros((B, 1, N_MELS, T), dtype=np.float32)
            Y = np.zeros((B, T), dtype=np.float32)
            for j, (pc, lb) in enumerate(batch):
                X[j, 0, :, :T] = pc[:, :T]
                Y[j, :T] = lb[:T]
            Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            opt.zero_grad()
            loss = crit(model(Xt), Yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs} loss={tl / nb:.4f}", flush=True)
    return model


def proba_to_events(proba, enter=0.7, exit_=0.3, min_ms=50.0):
    """Hysteresis → call events."""
    in_call = False
    events, start, conf_sum = [], 0, 0.0
    for i, p in enumerate(proba):
        if not in_call and p >= enter:
            in_call, start, conf_sum = True, i, p
        elif in_call:
            if p < exit_:
                in_call = False
                if (i - start) * FRAME_MS >= min_ms:
                    events.append((start * FRAME_MS, i * FRAME_MS, conf_sum / (i - start)))
            else:
                conf_sum += p
    if in_call and (len(proba) - start) * FRAME_MS >= min_ms:
        events.append((start * FRAME_MS, len(proba) * FRAME_MS, conf_sum / (len(proba) - start)))
    return events


def _band_ratio(psd, freqs):
    band = (freqs >= CALL_BAND[0]) & (freqs <= CALL_BAND[1])
    return float(psd[band].sum() / (psd.sum() + 1e-12))


def _win(audio, sr, t0_ms, t1_ms):
    i, j = int(t0_ms / 1000 * sr), int(t1_ms / 1000 * sr)
    seg = audio[i : max(j, i + 8)]
    if len(seg) < 8:
        return np.ones(8) * 1e-12, np.zeros(8)
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return mag**2 + 1e-12, np.fft.rfftfreq(len(seg), 1 / sr)


def _extract_evidence(prop, audio, sr, crnn_proba, event):
    t = prop.time_ms
    fi = int(t / FRAME_MS)
    nbd = proposal_features(prop)
    p_at = float(crnn_proba[fi]) if fi < len(crnn_proba) else 0.0
    p_b = float(crnn_proba[max(0, fi - 5)]) if fi - 5 < len(crnn_proba) else 0.0
    p_a = float(crnn_proba[min(len(crnn_proba) - 1, fi + 5)]) if fi + 5 < len(crnn_proba) else 0.0
    crnn_ctx = np.array(
        [
            p_at,
            p_b,
            p_a,
            max(0, p_at - p_b),
            max(0, p_b - p_at),
            1.0 if event and event[0] <= t <= event[1] else 0.0,
            abs(t - event[0]) if event else 9999.0,
            abs(t - event[1]) if event else 9999.0,
            event[2] if event else 0.0,
        ]
    )
    psd_pre, fr_pre = _win(audio, sr, t - 40, t)
    psd_post, fr_post = _win(audio, sr, t, t + 40)
    psd_body, fr_body = _win(audio, sr, t + 20, t + 60)
    body_rms = float(np.sqrt(psd_body.mean()))
    patch = np.array(
        [
            _band_ratio(psd_pre, fr_pre),
            _band_ratio(psd_post, fr_post),
            float(np.exp(np.log(psd_pre + 1e-12).mean()) / (psd_pre.mean() + 1e-12)),
            float(np.exp(np.log(psd_post + 1e-12).mean()) / (psd_post.mean() + 1e-12)),
            _band_ratio(psd_body, fr_body),
            body_rms / (float(np.sqrt(psd_pre.mean())) + 1e-8),
            float(np.exp(np.log(psd_body + 1e-12).mean()) / (psd_body.mean() + 1e-12)),
            body_rms,
        ]
    )
    return np.concatenate([nbd, crnn_ctx, patch, np.array([0.0, 0.0])])


def label_vs_gt(t, gt_starts, gt_ends, in_word_calls, tol=100.0):
    if gt_starts and min(abs(t - s) for s in gt_starts) < tol:
        return 1  # ONSET
    if gt_ends and min(abs(t - e) for e in gt_ends) < tol:
        return 2  # OFFSET
    for s, e, _ct, _caller in in_word_calls:
        if s * 1000 + tol < t < e * 1000 - tol:
            return 3  # INTERNAL
    return 0  # NOISE


def match_events_to_gt(events_onset_ms, gt_calls, tol=100.0):
    used = set()
    matched = {}
    for ei, pon in enumerate(events_onset_ms):
        best, bd = None, tol
        for gi, (s, e, ct, caller) in enumerate(gt_calls):
            if gi in used:
                continue
            d = abs(pon - s * 1000)
            if d < bd:
                bd, best = d, gi
        if best is not None:
            used.add(best)
            matched[ei] = (gt_calls[best][2], gt_calls[best][3])  # (calltype, caller)
    return matched


def _proba_4(clf, ev_row):
    """Expand classifier output to the fixed [NOISE, ONSET, OFFSET, INTERNAL] 4-vector,
    defaulting absent classes to 0.0 — robust to a class missing from training (the GBT
    only emits columns for classes it saw)."""
    pred = clf.predict_proba(ev_row.reshape(1, -1))[0]
    p4 = np.zeros(4)
    for ci, cls in enumerate(clf.classes_):
        p4[int(cls)] = pred[ci]
    return p4


def main():
    ap = argparse.ArgumentParser(description="HECB-Seg on InfantMarmosetsVox")
    ap.add_argument("--n-train", type=int, default=200, help="number of train files")
    ap.add_argument("--n-test", type=int, default=50, help="number of test files")
    ap.add_argument("--crnn-epochs", type=int, default=30)
    ap.add_argument("--k", type=int, default=11, help="k-means token count (call types)")
    ap.add_argument("--tol-ms", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    files = sample_files(args.n_train + args.n_test, args.seed)
    train_files, test_files = files[: args.n_train], files[args.n_train :]
    print(f"{len(train_files)} train / {len(test_files)} test files", flush=True)

    # ─── Stage 1: train CRNN ───
    print("Stage 1: training PCEN+CRNN call detector (44.1 kHz)...", flush=True)
    crnn = train_crnn(train_files, device, args.crnn_epochs)

    # ─── Duration priors ───
    all_call_durs = []
    for _fn, _wav, labels in train_files:
        for s, e, ct, _caller in get_call_spans(labels):
            all_call_durs.append([(s, e)])
    priors = fit_hecb_priors(all_call_durs)
    print(
        f"Duration priors: min={priors.min_call_ms:.0f}ms median={priors.median_call_ms:.0f}ms "
        f"max={priors.max_call_ms:.0f}ms",
        flush=True,
    )

    # ─── Stages 2-4: typed classifier ───
    print("Stage 2-4: typed evidence + classifier (train)...", flush=True)
    X_tr, y_tr = [], []
    for fi, (fn, wav, labels) in enumerate(train_files):
        try:
            audio, _ = load_audio(str(wav), target_sr=SR)
        except Exception:
            continue
        proba = _crnn_proba(crnn, audio, device)
        events = proba_to_events(proba)
        gt_calls = get_call_spans(labels)
        gt_starts = [s * 1000 for s, _e, _ct, _c in gt_calls]  # seconds → ms
        gt_ends = [e * 1000 for _s, e, _ct, _c in gt_calls]
        for on_ms, off_ms, conf in events:
            ev_audio = audio[int((on_ms - PAD_MS) / 1000 * SR) : int((off_ms + PAD_MS) / 1000 * SR)]
            if len(ev_audio) < int(SR * 0.02):
                continue
            ev_sig = compute_signals(ev_audio, SR, hop_ms=10.0)
            for prop in generate_proposals(ev_sig, **_PROPOSAL):
                prop.time_ms += on_ms - PAD_MS
                ev = _extract_evidence(prop, audio, SR, proba, (on_ms, off_ms, conf))
                X_tr.append(ev)
                y_tr.append(label_vs_gt(prop.time_ms, gt_starts, gt_ends, gt_calls, args.tol_ms))
        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(train_files)} files, {len(X_tr)} proposals", flush=True)

    X_tr, y_tr = np.asarray(X_tr), np.asarray(y_tr)
    print(
        f"typed proposals: "
        + " ".join(
            f"{['noise', 'onset', 'offset', 'internal'][k]}={v}"
            for k, v in sorted(Counter(y_tr.tolist()).items())
        ),
        flush=True,
    )
    if len(X_tr) == 0:
        print("ERROR: no proposals", flush=True)
        sys.exit(1)
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=0
    )
    clf.fit(X_tr, y_tr)

    # ─── Eval: onset F1 + call-type recovery ablation ───
    print(f"\nEval (onset tol={args.tol_ms}ms)...", flush=True)
    tp = fp = fn_ = 0
    all_reps, ct_labels, caller_labels = [], [], []
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="marmoset").combined  # noqa: E731
    for fi, (fn, wav, labels) in enumerate(test_files):
        try:
            audio, _ = load_audio(str(wav), target_sr=SR)
        except Exception:
            continue
        gt_calls = get_call_spans(labels)
        gt_onsets = [s * 1000 for s, _e, _ct, _c in gt_calls]
        proba = _crnn_proba(crnn, audio, device)
        events = proba_to_events(proba)
        event_onsets = [e[0] for e in events]
        r = prf_one_to_one(event_onsets, gt_onsets, tol=args.tol_ms)
        tp += r.tp
        fp += r.fp
        fn_ += r.fn

        # HECB decode + subcall reps for matched calls
        typed_props = []
        for on_ms, off_ms, conf in events:
            ev_audio = audio[int((on_ms - PAD_MS) / 1000 * SR) : int((off_ms + PAD_MS) / 1000 * SR)]
            if len(ev_audio) < int(SR * 0.02):
                continue
            ev_sig = compute_signals(ev_audio, SR, hop_ms=10.0)
            for prop in generate_proposals(ev_sig, **_PROPOSAL):
                prop.time_ms += on_ms - PAD_MS
                ev = _extract_evidence(prop, audio, SR, proba, (on_ms, off_ms, conf))
                typed_props.append((prop.time_ms, _proba_4(clf, ev)))
        from bioacoustic_pipeline.segmentation.hierarchical import HECBDurationPriors  # noqa: PLC0415

        decoded = hecb_decode(events, typed_props, priors or HECBDurationPriors(), file_id=fn)
        matched = match_events_to_gt([d.onset_s * 1000 for d in decoded], gt_calls, args.tol_ms)
        # Per-event rep building (label alignment guaranteed — build_subcall_representations
        # may skip any event as too-short/silent, so attach the label only when a rep is produced).
        for di, d in enumerate(decoded):
            if di not in matched:
                continue
            reps = build_subcall_representations([d], audio, SR, fp_fn)
            if reps:
                all_reps.append(reps[0])
                ct_labels.append(matched[di][0])
        if (fi + 1) % 10 == 0:
            print(f"  test {fi + 1}/{len(test_files)}", flush=True)

    p = tp / max(tp + fp, 1)
    r_ = tp / max(tp + fn_, 1)
    f1 = 2 * p * r_ / max(p + r_, 1e-8)
    print(f"\n=== InfantMarmosetsVox HECB-Seg ({len(test_files)} test, one-to-one) ===")
    print(f"  CRNN call detection: F1={f1:.3f} P={p:.3f} R={r_:.3f} (tol={args.tol_ms}ms)")
    print(f"  matched labeled calls: {len(all_reps)} | call types: {dict(Counter(ct_labels))}")

    if all_reps:
        result = whole_sub_pooled_ablation(all_reps, ct_labels, k=args.k)
        print(f"\n=== Whole / sub-call / pooled call-type recovery (k={args.k}) ===")
        print(format_ablation_table(result))
    print("INFANT MARMOSET HECB DONE")


if __name__ == "__main__":
    main()
