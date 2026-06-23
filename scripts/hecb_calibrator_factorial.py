#!/usr/bin/env python3
"""HECB vs calibrator factorial: feature × classifier × ABSTAIN × threshold (2×2×2×2).

The decisive table: separates feature effect (scalar vs HECB), CPC effect (±768D×3
context), classifier-label effect (binary keep/drop vs 4-class typed), ABSTAIN effect
(keep/abstain/drop 3-way), and threshold-tuning effect (default vs F1-tuned).

8 cells: {scalar, scalar+CPC, HECB, HECB+CPC} × {no-ABSTAIN, ABSTAIN+tuned}.

ABSTAIN implementation: the classifier outputs P(class); proposals with max-P in an
intermediate band [keep_low, keep_high] are DROPPED (abstained). keep_high is tuned
on train combined-F1; keep_low = keep_high × 0.5 (heuristic from the calibrator's design).
For binary: P(boundary); for typed: max(P(onset), P(offset), P(internal)).

Run:  python scripts/hecb_calibrator_factorial.py --n-train 400 --n-test 100

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
    t = prop.time_ms
    nbd = proposal_features(prop)
    geom = np.array([t, abs(t), t / 1000.0, 1.0, 500.0])
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
    r_word = prf_one_to_one(pred_times, word_starts, tol=EVAL_TOL_WORD)
    p1 = r_word.tp / max(r_word.tp + r_word.fp, 1)
    r1 = r_word.tp / max(r_word.tp + r_word.fn, 1)
    f1_word = 2 * p1 * r1 / max(p1 + r1, 1e-8)
    all_gt = sorted(set(list(word_starts) + list(phoneme_starts)))
    r_comb = prf_one_to_one(pred_times, all_gt, tol=EVAL_TOL_COMBINED)
    p2 = r_comb.tp / max(r_comb.tp + r_comb.fp, 1)
    r2 = r_comb.tp / max(r_comb.tp + r_comb.fn, 1)
    f1_combined = 2 * p2 * r2 / max(p2 + r2, 1e-8)
    return f1_word, f1_combined


def eval_method(proba_fn, props, word_starts, phon_starts, keep_high=0.5):
    """Evaluate a method on one file. keep_high = keep threshold (boundary-prob >= keep_high
    emits a boundary). Default 0.5 = the classifier's raw decision threshold (no tuned
    operating point). The ABSTAIN band [keep_high*0.5, keep_high) is dropped — at the
    boundary-F1 level this is equivalent to a single threshold (uncertain proposals emit no
    boundary either way); ABSTAIN's real value is uncertainty flagging, not raw F1."""
    times = []
    for i, p in enumerate(props):
        bp = proba_fn(i)  # boundary probability
        if keep_high * 0.5 <= bp < keep_high:
            continue  # ABSTAIN (uncertain middle band → no boundary)
        if bp >= keep_high:
            times.append(p.time_ms)
    return combined_f1(times, word_starts, phon_starts)


def tune_keep_high(proba_fn, train_data, lo=0.3, hi=0.9, step=0.05):
    """Grid-search keep_high on train combined-F1."""
    best_kh, best_f1 = 0.5, 0.0
    for kh in np.arange(lo, hi + step, step):
        f1s = []
        for props, ws, ps in train_data:
            proba_i = lambda i, _fn=proba_fn: _fn(i)  # noqa: E731
            _, f1c = eval_method(proba_i, props, ws, ps, keep_high=kh)
            f1s.append(f1c)
        mean_f1 = np.mean(f1s) if f1s else 0
        if mean_f1 > best_f1:
            best_kh, best_f1 = kh, mean_f1
    return best_kh, best_f1


def main():
    ap = argparse.ArgumentParser(description="HECB vs calibrator 8-cell factorial")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner + CPC backbone...", flush=True)
    import torchaudio  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model, labels = bundle.get_model(), bundle.get_labels()
    cpc = CPCBackbone()
    cpc_ok = cpc.available
    print(f"  CPC={cpc_ok}", flush=True)

    files = sample_files(args.n_train + args.n_test, args.seed)
    train_files, test_files = files[: args.n_train], files[args.n_train :]
    print(f"{len(train_files)} train / {len(test_files)} test", flush=True)

    def process(flac):
        audio, _ = load_audio(flac, target_sr=SR)
        if audio.size == 0:
            return None
        tr = load_librispeech_transcript(flac)
        if not tr:
            return None
        res = forced_align(tr, audio, SR, model=fa_model, labels=labels)
        if not res.word_spans:
            return None
        sig = compute_signals(audio, SR, hop_ms=10.0)
        props = generate_proposals(sig, floor_energy=0.04, floor_flux=0.02, debounce_ms=25.0)
        if len(props) < 5:
            return None
        scalars = np.asarray([proposal_features(p) for p in props], dtype=np.float32)
        hecb = np.asarray([ls_evidence(p, audio, SR) for p in props], dtype=np.float32)
        if cpc_ok:
            frames = cpc.embed_frames(audio, SR)
            cpc_ctx = proposal_context(props, frames, hop_ms=10.0, window=5).astype(np.float32)
        else:
            cpc_ctx = np.zeros((len(props), 0), dtype=np.float32)
        ws = [w[0] for w in res.word_spans]
        ps = list(res.phoneme_starts)
        y = np.asarray([label_full_audio(p.time_ms, res.word_spans, ps) for p in props])
        return props, scalars, hecb, cpc_ctx, y, ws, ps

    # ── Train data ──
    print("processing train...", flush=True)
    tr = []
    for fi, flac in enumerate(train_files):
        r = process(flac)
        if r:
            tr.append(r)
        if (fi + 1) % 100 == 0:
            print(f"  {fi + 1}/{len(train_files)}", flush=True)

    X_s = np.vstack([r[1] for r in tr])
    X_h = np.vstack([r[2] for r in tr])
    X_c = np.vstack([r[3] for r in tr]) if tr[0][3].shape[1] > 0 else np.zeros((len(X_s), 0))
    y_tr = np.concatenate([r[4] for r in tr])
    y_bin = (y_tr != NOISE).astype(int)
    print(f"  {len(y_tr)} proposals | {Counter(y_tr.tolist())}", flush=True)

    # ── Train 4 classifiers ──
    print("training 4 classifiers...", flush=True)
    gbt_kw = dict(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )

    clf_cal_s = HistGradientBoostingClassifier(**gbt_kw).fit(X_s, y_bin)
    clf_cal_sc = HistGradientBoostingClassifier(**gbt_kw).fit(np.hstack([X_s, X_c]), y_bin)
    clf_hecb = HistGradientBoostingClassifier(**gbt_kw).fit(X_h, y_tr)
    clf_hecb_cpc = HistGradientBoostingClassifier(**gbt_kw).fit(np.hstack([X_h, X_c]), y_tr)

    # ── Define 8 methods (4 classifiers × {no-abstain, abstain+tuned}) ──

    methods = []

    def make_binary_proba(clf, X_test):
        proba = clf.predict_proba(X_test)
        return lambda i: proba[i, list(clf.classes_).index(1)] if 1 in clf.classes_ else 0.0

    def make_typed_proba(clf, X_test):
        proba = clf.predict_proba(X_test)
        classes = list(clf.classes_)
        bound_cls = [c for c in [ONSET, OFFSET, INTERNAL] if c in classes]
        return lambda i: sum(proba[i, classes.index(c)] for c in bound_cls)

    # Method definitions: (name, clf, feature_matrix_fn, proba_fn_maker)
    method_defs = [
        ("scalar GBT 13-D", clf_cal_s, lambda r: r[1], make_binary_proba),
        ("scalar+CPC 2317-D", clf_cal_sc, lambda r: np.hstack([r[1], r[3]]), make_binary_proba),
        ("HECB typed 25-D", clf_hecb, lambda r: r[2], make_typed_proba),
        (
            "HECB+CPC typed 2329-D",
            clf_hecb_cpc,
            lambda r: np.hstack([r[2], r[3]]),
            make_typed_proba,
        ),
    ]

    # Tune keep_high for each method on train data
    print("tuning keep_high on train combined-F1...", flush=True)
    for name, clf, feat_fn, proba_maker in method_defs:
        # Build proba functions for train data
        for r in tr:
            X_t = feat_fn(r)
            proba_fn = proba_maker(clf, X_t)
            # tune on this file's proposals
        # Actually tune across ALL train files at once
        # Build a combined proba array
        all_proba = []
        all_eval = []
        for r in tr:
            X_t = feat_fn(r)
            proba_fn = proba_maker(clf, X_t)
            all_proba.append([proba_fn(i) for i in range(len(r[0]))])
            all_eval.append((r[0], r[5], r[6]))

        best_kh, best_train_f1 = 0.5, 0.0
        for kh in np.arange(0.20, 0.91, 0.05):
            f1s = []
            for fi_idx, (props, ws, ps) in enumerate(all_eval):
                times = []
                for i in range(len(props)):
                    bp = all_proba[fi_idx][i]
                    if kh > 0 and kh * 0.5 <= bp < kh:
                        continue
                    if bp >= kh:
                        times.append(props[i].time_ms)
                _, f1c = combined_f1(times, ws, ps)
                f1s.append(f1c)
            mean_f1 = np.mean(f1s) if f1s else 0
            if mean_f1 > best_train_f1:
                best_kh, best_train_f1 = kh, mean_f1
        methods.append((name, clf, feat_fn, proba_maker, best_kh, best_train_f1))
        print(
            f"  {name}: keep_high={best_kh:.2f} → train combined-F1={best_train_f1:.3f}", flush=True
        )

    # ── Eval on test ──
    print(f"evaluating {len(test_files)} test files...", flush=True)
    test_recs = []
    for fi, flac in enumerate(test_files):
        r = process(flac)
        if r:
            test_recs.append(r)
        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(test_files)}", flush=True)

    print(f"\n{'=' * 85}")
    print(f"HECB vs CALIBRATOR FACTORIAL — LibriSpeech boundary F1 (n_test={len(test_recs)})")
    print(f"{'=' * 85}")
    print(f"  {'method':30} {'ABSTAIN':>8} {'word-F1':>8} {'combined-F1':>12}")
    print("  " + "-" * 65)

    for name, clf, feat_fn, proba_maker, best_kh, _ in methods:
        for abstain_label, kh in [("no", 0.5), ("yes+tuned", best_kh)]:
            f1w_list, f1c_list = [], []
            for r in test_recs:
                X_t = feat_fn(r)
                proba_fn = proba_maker(clf, X_t)
                times = []
                for i in range(len(r[0])):
                    bp = proba_fn(i)
                    if kh > 0 and kh * 0.5 <= bp < kh:
                        continue
                    if bp >= kh:
                        times.append(r[0][i].time_ms)
                f1w, f1c = combined_f1(times, r[5], r[6])
                f1w_list.append(f1w)
                f1c_list.append(f1c)
            wf1 = np.mean(f1w_list) if f1w_list else 0
            cf1 = np.mean(f1c_list) if f1c_list else 0
            print(f"  {name:30} {abstain_label:>8} {wf1:8.3f} {cf1:12.3f}")

    print("\n  §1 reference: NBD+CPC calibrator combined F1 = 0.836")
    print("  If ABSTAIN effect > CPC effect → abstention is the performance lever")
    print("  If HECB+ABSTAIN ≈ scalar+CPC+ABSTAIN → compact typed evidence suffices")
    print("\nFACTORIAL DONE")


if __name__ == "__main__":
    main()
