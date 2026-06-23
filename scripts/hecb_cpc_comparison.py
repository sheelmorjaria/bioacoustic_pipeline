#!/usr/bin/env python3
"""HECB typed classifier + CPC vs NBD+CPC calibrator on LibriSpeech boundary F1.

The architectural question: does the HECB 4-class typed classifier (onset/offset/internal/
noise) + CPC context match the NBD+CPC calibrator's combined F1 (0.836) on the same
full-audio proposals?

Both methods see the SAME proposals, SAME CPC frames, SAME GT (wav2vec2 forced-aligned
word starts + phoneme starts). The comparison is purely: typed-classifier boundary
selection vs binary-calibrator boundary selection, each ± CPC learned-context.

Evaluated on the §1 combined/multi-resolution F1: predicted boundaries matched against
word starts (word-F1) AND word-starts ∪ phoneme-starts (combined-F1, reclaiming word-FPs
that land on real phoneme boundaries).

Run:  python scripts/hecb_cpc_comparison.py --n-train 400 --n-test 100

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics.probes import forced_align, load_librispeech_transcript  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_context, proposal_features  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

LS = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
SR = 16000
PAD_MS = 30.0
ONSET_TOL = 40.0
PHON_TOL = 40.0
EVAL_TOL_WORD = 50.0
EVAL_TOL_COMBINED = 40.0
NOISE, ONSET, OFFSET, INTERNAL = 0, 1, 2, 3


def sample_files(n, seed):
    rng = random.Random(seed)
    flacs = sorted(glob.glob(str(LS / "**" / "*.flac"), recursive=True))
    rng.shuffle(flacs)
    return flacs[:n]


def label_full_audio(t, word_spans, phoneme_starts):
    """Label a proposal time against GT: onset/offset/internal/noise."""
    for ws, we, _w in word_spans:
        if abs(t - ws) < ONSET_TOL:
            return ONSET
        if abs(t - we) < ONSET_TOL:
            return OFFSET
    for ph in phoneme_starts:
        if abs(t - ph) < PHON_TOL:
            return INTERNAL
    return NOISE


def _band_ratio(psd, freqs):
    band = (freqs >= 300) & (freqs <= 3000)
    return float(psd[band].sum() / (psd.sum() + 1e-12))


def _flatness(psd):
    return float(np.exp(np.log(psd + 1e-12).mean()) / (psd.mean() + 1e-12))


def _win(audio, sr, t0_ms, t1_ms):
    i, j = int(t0_ms / 1000 * sr), int(t1_ms / 1000 * sr)
    seg = audio[i : max(j, i + int(0.005 * sr))]
    if len(seg) < 8:
        return np.ones(8) * 1e-12, np.zeros(8)
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return mag**2 + 1e-12, np.fft.rfftfreq(len(seg), 1 / sr)


def ls_evidence(prop, audio, sr):
    """HECB 25-D: NBD scalars (13) + geometry (5) + patch (7). Word-geometry uses
    nearest word boundaries (for full-audio proposals, find closest word)."""
    t = prop.time_ms
    nbd = proposal_features(prop)  # 13-D
    geom = np.array([t, abs(t), t / 1000.0, 1.0, 500.0])  # simplified geometry
    psd_pre, fr_pre = _win(audio, sr, t - 30, t)
    psd_post, fr_post = _win(audio, sr, t, t + 30)
    psd_body, _ = _win(audio, sr, t + 10, t + 40)
    patch = np.array(
        [
            _band_ratio(psd_pre, fr_pre),
            _band_ratio(psd_post, fr_post),
            _flatness(psd_pre),
            _flatness(psd_post),
            _flatness(psd_body),
            float(np.sqrt(psd_post.mean())) / (float(np.sqrt(psd_pre.mean())) + 1e-8),
            float(np.sqrt(psd_body.mean())),
        ]
    )
    return np.concatenate([nbd, geom, patch])


def combined_f1(pred_times, word_starts, phoneme_starts):
    """Word-level F1 + combined/multi-resolution F1 (reclaim word-FPs on phonemes)."""
    r_word = prf_one_to_one(pred_times, word_starts, tol=EVAL_TOL_WORD)
    p1 = r_word.tp / max(r_word.tp + r_word.fp, 1)
    r1 = r_word.tp / max(r_word.tp + r_word.fn, 1)
    f1_word = 2 * p1 * r1 / max(p1 + r1, 1e-8)
    # combined: predicted vs (word_starts ∪ phoneme_starts)
    all_gt = sorted(set(list(word_starts) + list(phoneme_starts)))
    r_comb = prf_one_to_one(pred_times, all_gt, tol=EVAL_TOL_COMBINED)
    p2 = r_comb.tp / max(r_comb.tp + r_comb.fp, 1)
    r2 = r_comb.tp / max(r_comb.tp + r_comb.fn, 1)
    f1_combined = 2 * p2 * r2 / max(p2 + r2, 1e-8)
    return f1_word, f1_combined


def main():
    ap = argparse.ArgumentParser(description="HECB+CPC vs NBD+CPC calibrator boundary F1")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner + CPC backbone...", flush=True)
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model, labels = bundle.get_model(), bundle.get_labels()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpc = CPCBackbone()
    cpc_ok = cpc.available
    print(f"  CPC available={cpc_ok}, device={device}", flush=True)

    files = sample_files(args.n_train + args.n_test, args.seed)
    train_files, test_files = files[: args.n_train], files[args.n_train :]
    print(f"{len(train_files)} train / {len(test_files)} test files", flush=True)

    def process_file(flac):
        """Returns (proposals, scalars_13d, hecb_25d, cpc_ctx, word_starts, phon_starts) or None."""
        audio, _ = load_audio(flac, target_sr=SR)
        if audio.size == 0:
            return None
        tr = load_librispeech_transcript(flac)
        if not tr:
            return None
        res = forced_align(tr, audio, SR, model=fa_model, labels=labels)
        if not res.word_spans:
            return None
        # NBD proposals on FULL audio
        sig = compute_signals(audio, SR, hop_ms=10.0)
        props = generate_proposals(sig, floor_energy=0.04, floor_flux=0.02, debounce_ms=25.0)
        if len(props) < 5:
            return None
        # Features
        scalars = np.asarray([proposal_features(p) for p in props], dtype=np.float32)
        hecb = np.asarray([ls_evidence(p, audio, SR) for p in props], dtype=np.float32)
        # CPC context
        if cpc_ok:
            frames = cpc.embed_frames(audio, SR)
            cpc_ctx = proposal_context(props, frames, hop_ms=10.0, window=5).astype(np.float32)
        else:
            cpc_ctx = np.zeros((len(props), 0), dtype=np.float32)
        # Labels
        word_starts = [ws for ws, we, _w in res.word_spans]
        phon_starts = list(res.phoneme_starts)
        y = np.asarray([label_full_audio(p.time_ms, res.word_spans, phon_starts) for p in props])
        return props, scalars, hecb, cpc_ctx, y, word_starts, phon_starts

    # ── Train ──
    print("processing train files...", flush=True)
    tr_scalars, tr_hecb, tr_cpc, tr_y = [], [], [], []
    for fi, flac in enumerate(train_files):
        r = process_file(flac)
        if r is None:
            continue
        _, scalars, hecb, cpc_ctx, y, _, _ = r
        tr_scalars.append(scalars)
        tr_hecb.append(hecb)
        tr_cpc.append(cpc_ctx)
        tr_y.append(y)
        if (fi + 1) % 100 == 0:
            n = sum(len(x) for x in tr_y)
            print(f"  {fi + 1}/{len(train_files)} → {n} proposals", flush=True)

    X_s = np.vstack(tr_scalars)
    X_h = np.vstack(tr_hecb)
    X_c = np.vstack(tr_cpc) if tr_cpc and tr_cpc[0].shape[1] > 0 else np.zeros((len(X_s), 0))
    y_tr = np.concatenate(tr_y)
    print(f"  train: {len(y_tr)} proposals | {Counter(y_tr.tolist())}", flush=True)

    # Binary labels for calibrator (boundary vs noise)
    y_bin = (y_tr != NOISE).astype(int)

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    print("training 4 classifiers...", flush=True)
    clf_cal_s = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf_cal_s.fit(X_s, y_bin)

    clf_cal_sc = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf_cal_sc.fit(np.hstack([X_s, X_c]), y_bin)

    clf_hecb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf_hecb.fit(X_h, y_tr)

    clf_hecb_cpc = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf_hecb_cpc.fit(np.hstack([X_h, X_c]), y_tr)

    # ── Eval ──
    print(f"evaluating on {len(test_files)} test files...", flush=True)
    methods = {
        "calibrator scalar-only (13-D)": lambda s, h, c: clf_cal_s.predict(s),
        "calibrator scalar+CPC (2317-D)": lambda s, h, c: clf_cal_sc.predict(np.hstack([s, c])),
        "HECB typed (25-D)": lambda s, h, c: clf_hecb.predict(h),
        "HECB+CPC typed (2329-D)": lambda s, h, c: clf_hecb_cpc.predict(np.hstack([h, c])),
    }
    results = {m: {"word_f1": [], "combined_f1": []} for m in methods}

    for fi, flac in enumerate(test_files):
        r = process_file(flac)
        if r is None:
            continue
        props, scalars, hecb, cpc_ctx, y_true, word_starts, phon_starts = r
        if len(props) < 3:
            continue
        for mname, predict_fn in methods.items():
            # Calibrator: binary predict → 1 = boundary → proposal times
            # HECB: 4-class → {ONSET, OFFSET, INTERNAL} = boundary
            pred = predict_fn(scalars, hecb, cpc_ctx)
            if "typed" in mname:
                boundary_mask = np.isin(pred, [ONSET, OFFSET, INTERNAL])
            else:
                boundary_mask = pred == 1
            pred_times = [props[i].time_ms for i in range(len(props)) if boundary_mask[i]]
            f1w, f1c = combined_f1(pred_times, word_starts, phon_starts)
            results[mname]["word_f1"].append(f1w)
            results[mname]["combined_f1"].append(f1c)
        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(test_files)} test files", flush=True)

    # ── Report ──
    print(f"\n{'=' * 80}")
    print(f"HECB+CPC vs NBD+CPC CALIBRATOR — LibriSpeech boundary F1 (n_test={len(test_files)})")
    print(f"{'=' * 80}")
    print(f"  {'method':38} {'word-F1':>8} {'combined-F1':>12}")
    print("  " + "-" * 62)
    for mname in methods:
        wf1 = np.mean(results[mname]["word_f1"]) if results[mname]["word_f1"] else 0
        cf1 = np.mean(results[mname]["combined_f1"]) if results[mname]["combined_f1"] else 0
        print(f"  {mname:38} {wf1:8.3f} {cf1:12.3f}")
    print("\n  §1 reference: NBD+CPC calibrator combined F1 = 0.836 (150-file progression)")
    print("  Combined-F1 = predicted boundaries vs (word-starts ∪ phoneme-starts), tol=40ms")
    print("  If HECB+CPC ≈ calibrator+CPC → typed classifier is competitive as primary detector")
    print("  If HECB+CPC < calibrator+CPC → binary keep/abstain/drop is the right speech detector")
    print(
        "  If HECB+CPC ≈ HECB-only → CPC context adds little to typed classifier (unlike calibrator)"
    )
    print("\nHECB CPC COMPARISON DONE")


if __name__ == "__main__":
    main()
