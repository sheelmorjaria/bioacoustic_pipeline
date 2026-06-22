#!/usr/bin/env python3
"""Apples-to-apples HECB vs NBD on LibriSpeech: same oracle word spans, same phoneme GT.

Three internal-boundary methods inside the SAME oracle word spans, all evaluated against
the SAME in-word phoneme starts (one-to-one F1 @40ms):

  1. Raw NBD:        accept every acoustic-change proposal (no filtering)
  2. Scalar GBT:     binary classifier (is-phoneme vs not) over 25-D evidence
  3. Typed HECB:     4-class classifier (onset/offset/internal/noise) → keep internal-typed

This answers: does typed HECB improve internal phoneme recovery over plain NBD inside the
same parent events?

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402

import hecb_librispeech as L  # reuse all LS machinery  # noqa: E402

ONSET_TOL = 50.0
PHON_TOL = 40.0
EVAL_TOL = 40.0


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="HECB vs NBD on LibriSpeech (apples-to-apples)")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner (CPU)...", flush=True)
    import torchaudio  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()
    labels = bundle.get_labels()

    files = L.sample_files(args.n_train + args.n_test, args.seed)
    train_files, test_files = files[: args.n_train], files[args.n_train :]
    print(f"{len(train_files)} train / {len(test_files)} test", flush=True)

    def align(flac):
        from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

        audio, _ = load_audio(flac, target_sr=L.SR)
        if audio.size == 0:
            return None, None
        tr = L.load_librispeech_transcript(flac)
        if not tr:
            return None, None
        res = L.forced_align(tr, audio, L.SR, model=model, labels=labels)
        if not res.word_spans:
            return None, None
        return audio, res

    # ─── Pre-align ALL files once (the wav2vec2 bottleneck) ───
    print("aligning train+test once...", flush=True)
    train_aligned = []
    for flac in train_files:
        a = align(flac)
        if a[0] is not None:
            train_aligned.append(a)
    test_aligned = []
    for flac in test_files:
        a = align(flac)
        if a[0] is not None:
            test_aligned.append(a)
    print(f"  aligned {len(train_aligned)} train / {len(test_aligned)} test", flush=True)

    # ─── Build training evidence ───
    print("building evidence (train)...", flush=True)
    X_typed, y_typed = [], []  # 4-class: onset/offset/internal/noise
    X_scalar, y_scalar = [], []  # binary: is-phoneme / not
    for audio, res in train_aligned:
        for (ws, we, _w) in res.word_spans:
            if we - ws < 80:
                continue
            in_word_phon = [p for p in res.phoneme_starts if ws + ONSET_TOL < p < we - ONSET_TOL]
            for prop in L.word_internal_proposals(audio, L.SR, ws, we):
                ev = L.ls_evidence(prop, audio, L.SR, ws, we)
                X_typed.append(ev)
                y_typed.append(L.label_proposal(prop.time_ms, ws, we, in_word_phon, ONSET_TOL, PHON_TOL))
                X_scalar.append(ev)
                y_scalar.append(1 if any(abs(prop.time_ms - p) < PHON_TOL for p in in_word_phon) else 0)
    X_typed, y_typed = np.asarray(X_typed), np.asarray(y_typed)
    X_scalar, y_scalar = np.asarray(X_scalar), np.asarray(y_scalar)
    print(f"  typed: {Counter(y_typed.tolist())} | scalar: {Counter(y_scalar.tolist())}", flush=True)

    # ─── Train both classifiers ───
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    typed_clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=0)
    typed_clf.fit(X_typed, y_typed)
    scalar_clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=0)
    scalar_clf.fit(X_scalar, y_scalar)
    print("trained typed + scalar classifiers", flush=True)

    # ─── Helper: proba_4 for typed (handles missing classes) ───
    def _proba4(clf, ev):
        pred = clf.predict_proba(ev.reshape(1, -1))[0]
        p4 = np.zeros(4)
        for ci, cls in enumerate(clf.classes_):
            p4[int(cls)] = pred[ci]
        return p4

    # ─── Eval: 3 methods on same test words ───
    print(f"evaluating 3 methods (tol={EVAL_TOL}ms)...", flush=True)
    raw_tp = raw_fp = raw_fn = 0
    scalar_tp = scalar_fp = scalar_fn = 0
    typed_tp = typed_fp = typed_fn = 0
    for fi, (audio, res) in enumerate(test_aligned):
        for (ws, we, _w) in res.word_spans:
            if we - ws < 80:
                continue
            gt_phon = [p for p in res.phoneme_starts if ws + ONSET_TOL < p < we - ONSET_TOL]
            props = L.word_internal_proposals(audio, L.SR, ws, we)
            if not props:
                raw_fn += len(gt_phon)
                scalar_fn += len(gt_phon)
                typed_fn += len(gt_phon)
                continue
            evidences = np.asarray([L.ls_evidence(p, audio, L.SR, ws, we) for p in props])

            # Method 1: raw NBD — ALL proposals as internal boundaries
            raw_pred = [p.time_ms for p in props]

            # Method 2: scalar GBT — keep proposals with P(phoneme) > 0.5
            scalar_proba = scalar_clf.predict_proba(evidences)
            scalar_col = list(scalar_clf.classes_).index(1) if 1 in scalar_clf.classes_ else -1
            scalar_pred = [props[i].time_ms for i in range(len(props))
                           if scalar_col >= 0 and scalar_proba[i, scalar_col] > 0.5]

            # Method 3: typed HECB — keep INTERNAL-typed (confidence > 0.3)
            typed_pred = []
            for i, p in enumerate(props):
                p4 = _proba4(typed_clf, evidences[i])
                if p4[L.INTERNAL] > 0.3:
                    typed_pred.append(p.time_ms)

            r = prf_one_to_one(raw_pred, gt_phon, tol=EVAL_TOL)
            raw_tp += r.tp; raw_fp += r.fp; raw_fn += r.fn
            r = prf_one_to_one(scalar_pred, gt_phon, tol=EVAL_TOL)
            scalar_tp += r.tp; scalar_fp += r.fp; scalar_fn += r.fn
            r = prf_one_to_one(typed_pred, gt_phon, tol=EVAL_TOL)
            typed_tp += r.tp; typed_fp += r.fp; typed_fn += r.fn

        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(test_aligned)} test files", flush=True)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-8), p, r

    raw_f1, raw_p, raw_r = prf(raw_tp, raw_fp, raw_fn)
    sca_f1, sca_p, sca_r = prf(scalar_tp, scalar_fp, scalar_fn)
    typ_f1, typ_p, typ_r = prf(typed_tp, typed_fp, typed_fn)

    print(f"\n{'='*70}")
    print(f"HECB vs NBD on LibriSpeech (same oracle word spans, phoneme GT @ {EVAL_TOL}ms)")
    print(f"{'='*70}")
    print(f"{'method':28} {'F1':>6} {'P':>6} {'R':>6}  {'n_pred':>7}")
    print(f"{'-'*60}")
    print(f"{'1. raw NBD (all proposals)':28} {raw_f1:6.3f} {raw_p:6.3f} {raw_r:6.3f}  {raw_tp+raw_fp:7d}")
    print(f"{'2. scalar GBT (P>0.5)':28} {sca_f1:6.3f} {sca_p:6.3f} {sca_r:6.3f}  {scalar_tp+scalar_fp:7d}")
    print(f"{'3. typed HECB (INTERNAL>0.3)':28} {typ_f1:6.3f} {typ_p:6.3f} {typ_r:6.3f}  {typed_tp+typed_fp:7d}")
    print(f"\n  typed vs raw:   ΔF1={typ_f1-raw_f1:+.3f}  ΔP={typ_p-raw_p:+.3f}  ΔR={typ_r-raw_r:+.3f}")
    print(f"  typed vs scalar: ΔF1={typ_f1-sca_f1:+.3f}  ΔP={typ_p-sca_p:+.3f}  ΔR={typ_r-sca_r:+.3f}")
    print("HECB VS NBD LIBRISPEECH DONE")


if __name__ == "__main__":
    main()
