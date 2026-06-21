#!/usr/bin/env python3
"""HECB-Seg on LibriSpeech: mechanistic test (do typed internal boundaries find phonemes?).

The PCEN+CRNN front-end is a calls-in-noise detector and is weak on continuous speech
(F1~0.31 word). But HECB's *internal-boundary* mechanism is, in principle, regime-
agnostic: given an event (here a word), do typed NBD proposals inside it recover sub-word
(phoneme) structure? This decouples the internal-boundary mechanism from CRNN quality.

Experiment (oracle events = GT word spans from wav2vec2 forced alignment):
  - For each GT word, run fine-grained NBD proposals inside it.
  - Train a typed classifier: onset (word start) / offset (word end) / internal (a phoneme
    onset strictly inside the word) / noise.
  - Evaluate on held-out files: do INTERNAL-typed boundaries recover in-word phoneme GT?

Metric: INTERNAL-boundary F1 vs in-word phoneme starts (one-to-one, ~40 ms tol). This is
the speech analog of the MeerKAT internal-boundary test — if it works, HECB's internal
mechanism finds real sub-word structure and the combined/multi-resolution F1 reclaim
(word-FPs landing on phonemes) is mechanistically justified.

NB: wav2vec2 is the GT yardstick (human-only, measurement instrument, not firewall-
registered) — used only to PRODUCE ground truth, never as pipeline features.

Usage:
  python3 scripts/hecb_librispeech.py --n-train 400 --n-test 100

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

from bioacoustic_pipeline.diagnostics.probes import (  # noqa: E402
    forced_align,
    load_librispeech_transcript,
)
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

LS = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
SR = 16000
PAD_MS = 30.0
ONSET_TOL = 40.0  # ms — a proposal this close to a word edge is onset/offset
PHON_TOL = 40.0  # ms — a proposal this close to an in-word phoneme is internal
EVAL_TOL = 40.0  # ms — eval matching tolerance for internal-boundary F1
SPEECH_BAND = (300, 3000)  # Hz — formant-ish band for patch features

# Typed labels (match segmentation.hierarchical HECB_* ordering)
NOISE, ONSET, OFFSET, INTERNAL = 0, 1, 2, 3
LABEL_NAMES = {0: "noise", 1: "onset", 2: "offset", 3: "internal"}


def _band_ratio(psd, freqs):
    band = (freqs >= SPEECH_BAND[0]) & (freqs <= SPEECH_BAND[1])
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


def ls_evidence(prop, audio, sr, word_onset_ms, word_offset_ms):
    """Typed evidence for one NBD proposal inside a GT word (no CRNN — oracle events).

    = NBD scalars (13) + word-geometry (5) + patch features (7) = 25-D.
    """
    t = prop.time_ms
    nbd = proposal_features(prop)  # 13-D
    geom = np.array(
        [
            abs(t - word_onset_ms),
            abs(t - word_offset_ms),
            (t - word_onset_ms) / max(word_offset_ms - word_onset_ms, 1.0),  # rel position
            1.0 if word_onset_ms <= t <= word_offset_ms else 0.0,  # inside
            word_offset_ms - word_onset_ms,  # word dur
        ]
    )
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
            float(np.sqrt(psd_post.mean())) / (float(np.sqrt(psd_pre.mean())) + 1e-8),  # sustain
            float(np.sqrt(psd_body.mean())),
        ]
    )
    return np.concatenate([nbd, geom, patch])


def label_proposal(
    t, word_onset_ms, word_offset_ms, in_word_phonemes, onset_tol=ONSET_TOL, phon_tol=PHON_TOL
):
    """Typed label: onset (word start) / offset (word end) / internal (in-word phoneme) / noise."""
    if abs(t - word_onset_ms) < onset_tol:
        return ONSET
    if abs(t - word_offset_ms) < onset_tol:
        return OFFSET
    for ph in in_word_phonemes:
        if abs(t - ph) < phon_tol:
            return INTERNAL
    return NOISE


def word_internal_proposals(audio, sr, word_onset_ms, word_offset_ms):
    """Fine-grained NBD proposals inside one GT word (±PAD_MS). Returns list[Proposal] in abs ms."""
    lo = max(0, int((word_onset_ms - PAD_MS) / 1000 * sr))
    hi = min(len(audio), int((word_offset_ms + PAD_MS) / 1000 * sr))
    seg = audio[lo:hi]
    if len(seg) < int(sr * 0.02):
        return []
    sig = compute_signals(seg, sr, hop_ms=10.0)
    props = generate_proposals(sig, floor_energy=0.03, floor_flux=0.02, debounce_ms=20.0)
    for p in props:
        p.time_ms += word_onset_ms - PAD_MS
    return props


def sample_files(n, seed):
    rng = random.Random(seed)
    flacs = sorted(glob.glob(str(LS / "**" / "*.flac"), recursive=True))
    rng.shuffle(flacs)
    out = []
    for f in flacs:
        if len(out) >= n:
            break
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description="HECB-Seg LibriSpeech mechanistic test")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner (GT yardstick, human-only, CPU)...", flush=True)
    import torchaudio  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    # forced_align() builds its input tensor on CPU internally, so the model must stay on
    # CPU (shared code — can't inject device). GT generation only; ~1-2 s/utterance.
    model = bundle.get_model()
    labels = bundle.get_labels()

    files = sample_files(args.n_train + args.n_test, args.seed)
    train_files, test_files = files[: args.n_train], files[args.n_train :]
    print(f"{len(train_files)} train / {len(test_files)} test files", flush=True)

    def align(flac):
        audio, _ = load_audio(flac, target_sr=SR)
        if audio.size == 0:
            return None, None
        tr = load_librispeech_transcript(flac)
        if not tr:
            return None, None
        res = forced_align(tr, audio, SR, model=model, labels=labels)
        if not res.word_spans:
            return None, None
        return audio, res

    # ─── Train: typed classifier over in-word proposals ───
    print("Stage A: generating typed evidence (train)...", flush=True)
    X_tr, y_tr = [], []
    for fi, flac in enumerate(train_files):
        audio, res = align(flac)
        if audio is None:
            continue
        phoneme_starts = res.phoneme_starts
        for ws, we, _w in res.word_spans:
            if we - ws < 80:  # skip tiny words
                continue
            in_word_phon = [p for p in phoneme_starts if ws + ONSET_TOL < p < we - ONSET_TOL]
            for prop in word_internal_proposals(audio, SR, ws, we):
                X_tr.append(ls_evidence(prop, audio, SR, ws, we))
                y_tr.append(label_proposal(prop.time_ms, ws, we, in_word_phon))
        if (fi + 1) % 100 == 0:
            print(f"  {fi + 1}/{len(train_files)} files, {len(X_tr)} proposals", flush=True)

    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    print(
        "typed proposals: "
        + " ".join(f"{LABEL_NAMES[k]}={v}" for k, v in sorted(Counter(y_tr.tolist()).items())),
        flush=True,
    )
    if len(X_tr) == 0:
        print("ERROR: no proposals; check audio/transcript loading.", flush=True)
        sys.exit(1)

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    from sklearn.metrics import classification_report  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(X_tr, y_tr)
    print(
        classification_report(
            y_tr,
            clf.predict(X_tr),
            target_names=[LABEL_NAMES[i] for i in range(4)],
            zero_division=0,
            digits=3,
        ),
        flush=True,
    )

    # ─── Eval: do INTERNAL-typed boundaries recover in-word phoneme GT? ───
    print(f"Stage B: evaluating internal-boundary recovery (tol={EVAL_TOL}ms)...", flush=True)
    int_tp = int_fp = int_fn = 0
    n_words = n_words_with_phon = 0
    for fi, flac in enumerate(test_files):
        audio, res = align(flac)
        if audio is None:
            continue
        for ws, we, _w in res.word_spans:
            if we - ws < 80:
                continue
            n_words += 1
            gt_phon = [p for p in res.phoneme_starts if ws + ONSET_TOL < p < we - ONSET_TOL]
            if gt_phon:
                n_words_with_phon += 1
            props = word_internal_proposals(audio, SR, ws, we)
            ev = (
                np.asarray([ls_evidence(p, audio, SR, ws, we) for p in props])
                if props
                else np.zeros((0, X_tr.shape[1]))
            )
            if len(ev) == 0:
                int_fn += len(gt_phon)
                continue
            proba = clf.predict_proba(ev)
            internal_pred = [
                p.time_ms for pi, p in zip(proba, props) if int(np.argmax(pi)) == INTERNAL
            ]
            r = prf_one_to_one(internal_pred, gt_phon, tol=EVAL_TOL)
            int_tp += r.tp
            int_fp += r.fp
            int_fn += r.fn
        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(test_files)} test files", flush=True)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-8), p, r

    f1, prec, rec = prf(int_tp, int_fp, int_fn)
    print("\n=== HECB internal-boundary → in-word phoneme recovery (LibriSpeech) ===")
    print(f"  words: {n_words} ({n_words_with_phon} with >=1 in-word phoneme)")
    print(
        f"  INTERNAL-boundary F1={f1:.3f} P={prec:.3f} R={rec:.3f}  (tol={EVAL_TOL}ms, one-to-one)"
    )
    print(
        f"  interpretation: typed internal boundaries recover "
        f"{rec * 100:.1f}% of in-word phoneme starts at P={prec:.3f}"
    )
    print(
        "  (oracle word events → word-level F1 = 1.0 by construction; this is purely the "
        "sub-word/phoneme-recovery test of HECB's internal mechanism)"
    )
    print("HECB LIBRISPEECH MECHANISTIC DONE")


if __name__ == "__main__":
    main()
