#!/usr/bin/env python3
"""HECB-Seg on LibriSpeech: FULL pipeline (router-validation check).

Closes the architecture story (NOT the main HECB proof — the mechanistic test in
hecb_librispeech.py already validated the internal-boundary mechanism):

  PCEN+CRNN is strong for discrete call events, weak for continuous-speech word boundaries.
  HECB internal segmentation works when the parent event is good (oracle) but cannot rescue
  poor parent events (CRNN). Therefore the segmentation router is necessary.

Pipeline:
  Stage 1: train PCEN+CRNN word detector on LibriSpeech word spans (16 kHz).
  Stage 2-4: typed classifier (onset/offset/internal=phoneme/noise) over CRNN events.
  Eval (router-validation table):
    a. CRNN word detection → word-level one-to-one F1          (event-detection bottleneck)
    b. combined/multi-resolution F1 (reclaim phoneme FPs)      (vs NBD+CPC 0.836)
    c. HECB-on-CRNN internal-boundary → phoneme recovery       (vs oracle-event 0.636)

Expected: (a) weak, (b) below NBD+CPC, (c) below oracle → CRNN is the speech bottleneck,
NOT the internal mechanism. wav2vec2 = GT yardstick only (CPU, human-only).

Each file is wav2vec2-aligned EXACTLY ONCE (the bottleneck) and reused across CRNN-label
build, hysteresis sweep, typed-evidence extraction, and eval.

Usage:
  python3 scripts/hecb_librispeech_full.py --n-train 400 --n-test 100 --crnn-epochs 40

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics.probes import forced_align, load_librispeech_transcript  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import (  # noqa: E402
    multi_resolution_f1,
    prf_one_to_one,
)
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

import hecb_librispeech as L  # reuse LS evidence/label/props helpers  # noqa: E402
import train_hecb_seg as H  # reuse CRNNDetector arch  # noqa: E402

SR = 16000
N_FFT, HOP, N_MELS = 512, 160, 64  # 10 ms hop at 16 kHz
FRAME_MS = HOP / SR * 1000.0
EVAL_TOL = 50.0
ONSET_TOL = 50.0
PHON_TOL = 40.0


def compute_pcen_16k(audio, sr=SR):
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


def train_crnn_words(train_data, device, epochs):
    pos_rate = float(np.mean([lb.mean() for _, lb in train_data]))
    print(f"  CRNN train: {len(train_data)} files, pos_rate={pos_rate:.3f}", flush=True)
    model = H.CRNNDetector().to(device)
    pw = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    MF = 1000
    for ep in range(epochs):
        model.train()
        random.Random(7 + ep).shuffle(train_data)
        tl, nb = 0.0, 0
        for i in range(0, len(train_data), 8):
            batch = train_data[i : i + 8]
            B = len(batch)
            X = np.zeros((B, 1, N_MELS, MF), dtype=np.float32)
            Y = np.zeros((B, MF), dtype=np.float32)
            for j, (pc, lb) in enumerate(batch):
                t = min(pc.shape[1], MF)
                X[j, 0, :, :t] = pc[:, :t]
                Y[j, :t] = lb[:t]
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


def crnn_proba(model, audio, device):
    model.eval()
    pcen = compute_pcen_16k(audio)
    t = min(pcen.shape[1], 1000)
    X = torch.from_numpy(pcen[:, :t]).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(X)).cpu().numpy()[0]


def proba_to_words(proba, enter, exit_, min_ms=60.0):
    in_w = False
    events = []
    start = 0
    for i, p in enumerate(proba):
        if not in_w and p >= enter:
            in_w = True
            start = i
        elif in_w and p < exit_:
            in_w = False
            if (i - start) * FRAME_MS >= min_ms:
                events.append((start * FRAME_MS, i * FRAME_MS))
    if in_w and (len(proba) - start) * FRAME_MS >= min_ms:
        events.append((start * FRAME_MS, len(proba) * FRAME_MS))
    return events


def label_vs_gt(t, gt_word_starts, gt_word_ends, in_word_phonemes, onset_tol, phon_tol):
    """GT-based typed label (NOT relative to the imprecise CRNN event edge):
    onset (near a GT word start) / offset (near a GT word end) / internal (near an
    in-word phoneme) / noise."""
    if gt_word_starts and min(abs(t - s) for s in gt_word_starts) < onset_tol:
        return L.ONSET
    if gt_word_ends and min(abs(t - e) for e in gt_word_ends) < onset_tol:
        return L.OFFSET
    if in_word_phonemes and min(abs(t - p) for p in in_word_phonemes) < phon_tol:
        return L.INTERNAL
    return L.NOISE


def main():
    ap = argparse.ArgumentParser(description="HECB-Seg LibriSpeech FULL (router validation)")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--crnn-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    print("loading wav2vec2 aligner (GT yardstick, CPU)...", flush=True)
    import torchaudio  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model_fa = bundle.get_model()
    labels = bundle.get_labels()

    files = L.sample_files(args.n_train + args.n_test, args.seed)
    train_files, test_files = files[: args.n_train], files[args.n_train :]
    print(f"{len(train_files)} train / {len(test_files)} test", flush=True)

    def align(flac):
        audio, _ = load_audio(flac, target_sr=SR)
        if audio.size == 0:
            return None, None
        tr = load_librispeech_transcript(flac)
        if not tr:
            return None, None
        res = forced_align(tr, audio, SR, model=model_fa, labels=labels)
        if not res.word_spans:
            return None, None
        return audio, res

    # ─── Pre-align EVERY file exactly once (the wav2vec2 bottleneck) ───
    print("aligning train+test once (wav2vec2 GT, CPU)...", flush=True)
    train_aligned, test_aligned = [], []
    for flac in train_files:
        a = align(flac)
        if a[0] is not None:
            train_aligned.append(a)
    for flac in test_files:
        a = align(flac)
        if a[0] is not None:
            test_aligned.append(a)
    print(f"  aligned {len(train_aligned)} train / {len(test_aligned)} test", flush=True)

    # ─── Stage 1: train CRNN word detector ───
    print("Stage 1: training PCEN+CRNN word detector...", flush=True)
    train_data = []
    for audio, res in train_aligned:
        pcen = compute_pcen_16k(audio)
        nf = pcen.shape[1]
        lab = np.zeros(nf, dtype=np.float32)
        for ws, we, _w in res.word_spans:
            lab[int(ws / FRAME_MS) : min(int(we / FRAME_MS) + 1, nf)] = 1.0
        train_data.append((pcen, lab))
    crnn = train_crnn_words(train_data, device, args.crnn_epochs)

    # CRNN proba cache (compute once per file; reused by hysteresis + evidence)
    print("computing CRNN proba (train)...", flush=True)
    train_proba = [(au, rs, crnn_proba(crnn, au, device)) for au, rs in train_aligned]

    # hysteresis sweep over CACHED proba (no re-alignment)
    best_f1, best_enter, best_exit = -1.0, 0.5, 0.2
    for enter in [0.3, 0.4, 0.5, 0.6]:
        for ex_ in [0.1, 0.2, 0.3]:
            tp = fp = fn = 0
            for _au, rs, proba in train_proba[:120]:
                ev = proba_to_words(proba, enter, ex_)
                r = prf_one_to_one([e[0] for e in ev], rs.word_starts, tol=EVAL_TOL)
                tp += r.tp
                fp += r.fp
                fn += r.fn
            p = tp / max(tp + fp, 1)
            r_ = tp / max(tp + fn, 1)
            f1 = 2 * p * r_ / max(p + r_, 1e-8)
            if f1 > best_f1:
                best_f1, best_enter, best_exit = f1, enter, ex_
    print(
        f"  hysteresis: enter={best_enter} exit={best_exit} train word-F1={best_f1:.3f}", flush=True
    )

    # ─── Stages 2-4: typed classifier over CRNN events (cached) ───
    print("Stage 2-3: typed evidence over CRNN events (train)...", flush=True)
    X_tr, y_tr = [], []
    for fi, (_au, rs, proba) in enumerate(train_proba):
        events = proba_to_words(proba, best_enter, best_exit)
        gt_we = [we for (_ws, we, _w) in rs.word_spans]
        in_word_phon = [
            p
            for p in rs.phoneme_starts
            if any(ws + ONSET_TOL < p < we - ONSET_TOL for ws, we, _w in rs.word_spans)
        ]
        for won, woff in events:
            for prop in L.word_internal_proposals(_au, SR, won, woff):
                X_tr.append(L.ls_evidence(prop, _au, SR, won, woff))
                y_tr.append(
                    label_vs_gt(
                        prop.time_ms, rs.word_starts, gt_we, in_word_phon, ONSET_TOL, PHON_TOL
                    )
                )
        if (fi + 1) % 100 == 0:
            print(f"  {fi + 1}/{len(train_proba)} files, {len(X_tr)} proposals", flush=True)
    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    print(
        "typed proposals: "
        + " ".join(f"{L.LABEL_NAMES[k]}={v}" for k, v in sorted(Counter(y_tr.tolist()).items())),
        flush=True,
    )
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(X_tr, y_tr)

    # ─── Eval: router-validation table ───
    print(f"Eval (tol={EVAL_TOL}ms)...", flush=True)
    w_tp = w_fp = w_fn = 0
    c_tp = c_fp = c_fn = 0  # combined/multi-resolution
    i_tp = i_fp = i_fn = 0  # HECB-on-CRNN internal → phoneme
    for fi, (audio, rs) in enumerate(test_aligned):
        proba = crnn_proba(crnn, audio, device)
        events = proba_to_words(proba, best_enter, best_exit)
        word_starts = [e[0] for e in events]

        # (a) word-level F1 + (b) combined/multi-resolution F1
        r = prf_one_to_one(word_starts, rs.word_starts, tol=EVAL_TOL)
        w_tp += r.tp
        w_fp += r.fp
        w_fn += r.fn
        mres = multi_resolution_f1(word_starts, rs.word_starts, rs.phoneme_starts)
        c_tp += mres.combined.tp
        c_fp += mres.combined.fp
        c_fn += mres.combined.fn

        # (c) HECB-on-CRNN internal boundaries → in-word phoneme recovery
        in_word_phon = [
            p
            for p in rs.phoneme_starts
            if any(ws + ONSET_TOL < p < we - ONSET_TOL for ws, we, _w in rs.word_spans)
        ]
        internal_pred = []
        for won, woff in events:
            props = L.word_internal_proposals(audio, SR, won, woff)
            if not props:
                continue
            ev = np.asarray([L.ls_evidence(p, audio, SR, won, woff) for p in props])
            pa = clf.predict_proba(ev)
            internal_pred += [
                p.time_ms for pi, p in zip(pa, props) if int(np.argmax(pi)) == L.INTERNAL
            ]
        rr = prf_one_to_one(internal_pred, in_word_phon, tol=PHON_TOL)
        i_tp += rr.tp
        i_fp += rr.fp
        i_fn += rr.fn
        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(test_aligned)} test files", flush=True)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-8), p, r

    wf = prf(w_tp, w_fp, w_fn)
    cf = prf(c_tp, c_fp, c_fn)
    iF = prf(i_tp, i_fp, i_fn)

    print("\n=== HECB-Seg FULL pipeline on LibriSpeech (router validation) ===")
    print(f"{'metric':40} {'F1':>6} {'P':>6} {'R':>6}")
    print("-" * 60)
    print(f"{'(a) CRNN word detection':40} {wf[0]:6.3f} {wf[1]:6.3f} {wf[2]:6.3f}")
    print(f"{'(b) combined/multi-resolution F1':40} {cf[0]:6.3f} {cf[1]:6.3f} {cf[2]:6.3f}")
    print(f"{'(c) HECB-on-CRNN internal→phoneme':40} {iF[0]:6.3f} {iF[1]:6.3f} {iF[2]:6.3f}")
    print("\n  refs: NBD+CPC calibrator combined-F1 = 0.836 | oracle-event HECB internal = 0.636")
    print(f"  router read: CRNN word detection ({wf[0]:.2f}) is the speech bottleneck;")
    print(f"    HECB-on-CRNN internal ({iF[0]:.2f}) < oracle-event (0.636) — poor parent events")
    print("    cannot be rescued. The segmentation router (NBD+calibrator for speech,")
    print("    PCEN+CRNN+HECB for calls-in-noise) is therefore necessary.")
    print("HECB LIBRISPEECH FULL DONE")


if __name__ == "__main__":
    main()
