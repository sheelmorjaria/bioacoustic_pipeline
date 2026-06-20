#!/usr/bin/env python3
"""Representation ablation benchmark — which representation best recovers call type?

Uses GROUND-TRUTH call windows (from MeerKAT h5 timestamps) to isolate representation
quality from segmentation quality. Same calls, same split, same labels, same probe.

Representations compared (firewall-legal for meerkat):
  1. acoustic_fingerprint (~43D, deterministic physics+texture)
  2. spectrogram (log-mel time-mean, deterministic DSP)
  3. fingerprint + spectrogram fusion (concat)
  4. fingerprint_tokens (discrete k-means → contingency Cramér's V / NMI / p)
CPC/BioMAE/AVES/β-VAE are firewall-blocked on meerkat (human/bat-restricted).

For continuous reps (1-3): LogisticRegression(class_weight="balanced"), StratifiedKFold(5),
macro-F1 + balanced accuracy + permutation null (200 shuffles).
For tokens (4): contingency-table Cramér's V, NMI, χ², permutation p.

Usage:
  python scripts/representation_benchmark.py --n-calls 2000 --report outputs/representation_benchmark.json

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (  # noqa: E402
    FingerprintTokenizer,
)
from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: E402
from bioacoustic_pipeline.representation.spectrogram_backbone import SpectrogramBackbone  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/home/sheel/birdsong_analysis/data/MeerKAT_10s_2024-06-12")


def collect_calls(n_target, seed=0, min_per_class=20):
    """Sample GT call windows (audio_slice, call_type) from MeerKAT h5 labels."""
    rng = random.Random(seed)
    files = sorted(glob.glob(str(DATA / "lbl" / "08000Hz" / "*.h5")))
    rng.shuffle(files)
    calls = []
    for f in files:
        if len(calls) >= n_target:
            break
        stem = Path(f).stem
        wav = DATA / "wav" / "08000Hz" / f"{stem}.wav"
        if not wav.exists():
            continue
        with h5py.File(f, "r") as h:
            st = np.asarray(h["start_time_lbl"][:]).ravel()
            et = np.asarray(h["end_time_lbl"][:]).ravel()
            lbl = np.asarray(h["lbl"][:]).ravel()
        try:
            audio, sr = load_audio(str(wav))
        except Exception:
            continue
        for s, e, lab in zip(st, et, lbl):
            if len(calls) >= n_target:
                break
            if not np.isfinite(s):
                continue
            s, e = float(s), float(e) if np.isfinite(e) else float(s) + 0.08
            i, j = int(s * sr), max(int(e * sr), int(s * sr) + int(0.03 * sr))
            seg = audio[i:j]
            if len(seg) < int(0.015 * sr):
                continue
            ct = lab.decode() if isinstance(lab, bytes) else str(lab)
            calls.append((seg, sr, ct))
    # filter to call types with >= min_per_class
    counts = Counter(c[2] for c in calls)
    keep = {t for t, n in counts.items() if n >= min_per_class}
    calls = [c for c in calls if c[2] in keep]
    return calls, {t: counts[t] for t in sorted(keep)}


def extract_vectors(calls):
    """Extract fingerprint + spectrogram vectors for each call."""
    spec_bb = SpectrogramBackbone()
    fp_vecs, spec_vecs, labels = [], [], []
    for seg, sr, ct in calls:
        fp = afp.extract(seg, sr, species="meerkat").combined
        if not np.any(fp):
            continue
        sp = spec_bb.embed(seg, sr)
        fp_vecs.append(np.asarray(fp, dtype=np.float32))
        spec_vecs.append(np.asarray(sp, dtype=np.float32))
        labels.append(ct)
    return (np.vstack(fp_vecs), np.vstack(spec_vecs), np.array(labels))


def evaluate_probe(X, y, n_splits=5, seed=0, n_perm=200):
    """LogisticRegression(class_weight='balanced'), StratifiedKFold, macro-F1 + bal-acc + permutation null."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    classes = sorted(set(y))
    if len(classes) < 2:
        return {"error": "need >=2 classes"}

    def _run(y_arg):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for tr, te in skf.split(X, y_arg):
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=seed, C=1.0)
            clf.fit(scaler.transform(X[tr]), y_arg[tr])
            pred = clf.predict(scaler.transform(X[te]))
            scores.append({
                "macro_f1": f1_score(y_arg[te], pred, average="macro", zero_division=0),
                "bal_acc": balanced_accuracy_score(y_arg[te], pred),
                "accuracy": float(np.mean(pred == y_arg[te])),
            })
        return {k: round(float(np.mean([s[k] for s in scores])), 4) for k in scores[0]}

    real = _run(y)
    # permutation null
    rng = np.random.RandomState(seed)
    null_f1 = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        null_f1.append(_run(yp)["macro_f1"])
    real_f1 = real["macro_f1"]
    p = (1 + sum(1 for s in null_f1 if s >= real_f1)) / (1 + n_perm)
    return {**real, "chance_macro_f1": round(float(np.mean(null_f1)), 4),
            "null_95": round(float(np.percentile(null_f1, 95)), 4),
            "permutation_p": round(p, 4), "n_classes": len(classes)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Representation ablation benchmark (GT call windows)")
    ap.add_argument("--n-calls", type=int, default=2000)
    ap.add_argument("--k", type=int, default=64, help="fingerprint k-means codebook size")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="outputs/representation_benchmark.json")
    args = ap.parse_args()

    calls, type_counts = collect_calls(args.n_calls, args.seed)
    print(f"collected {len(calls)} calls | {len(type_counts)} types: {type_counts}")
    if len(type_counts) < 2:
        print("ERROR: need >=2 call types with >=20 instances each"); return 1

    fp_vecs, spec_vecs, labels = extract_vectors(calls)
    print(f"extracted: fingerprint {fp_vecs.shape}, spectrogram {spec_vecs.shape}")
    fusion = np.hstack([fp_vecs, spec_vecs])

    # ---- continuous representations: identical probe ----
    results = {}
    for name, X in [("acoustic_fingerprint", fp_vecs), ("spectrogram", spec_vecs),
                    ("fusion_fp+spec", fusion)]:
        print(f"  probing {name} ({X.shape[1]}D) ...", end=" ", flush=True)
        r = evaluate_probe(X, labels, n_perm=args.n_perm, seed=args.seed)
        results[name] = r
        print(f"macro_f1={r.get('macro_f1','?')} bal_acc={r.get('bal_acc','?')} p={r.get('permutation_p','?')}")

    # ---- discrete tokens: contingency-table metrics ----
    tokenizer = FingerprintTokenizer.fit(fp_vecs, k=args.k, seed=args.seed)
    tokens = np.array([tokenizer.tokenize(v) for v in fp_vecs])
    binding = association_test([[int(t)] for t in tokens], labels.tolist(),
                               statistic="nmi", n_permutations=args.n_perm, seed=args.seed).as_dict()
    results["fingerprint_tokens"] = {
        "cramers_v": binding["cramers_v"], "nmi": binding["observed_nmi"],
        "chi2": binding.get("observed_chi2"), "p_value": binding["p_value"],
        "k": args.k, "n_classes": len(set(labels.tolist())),
    }
    print(f"  tokens: V={binding['cramers_v']:.3f} nmi={binding['observed_nmi']:.3f} p={binding['p_value']}")

    report = {
        "dataset": "MeerKAT_10s", "experiment": "representation_ablation_benchmark",
        "design": "GT call windows → representation → identical probe (LogReg balanced, StratifiedKFold-5, macro-F1 + permutation null)",
        "n_calls": len(labels), "call_type_counts": type_counts,
        "representations": results,
        "blocked": "CPC/BioMAE/AVES/β-VAE are firewall-blocked on meerkat (human/bat-restricted species)",
        "framing": ("Representation quality isolated from segmentation. Same calls, same split, same "
                    "labels, same classifier. Determines which representation encodes call-type information."),
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    # ---- summary table ----
    print(f"\n{'Representation':25} {'macro_F1':>8} {'bal_acc':>8} {'V/nmi':>8} {'perm_p':>8}")
    print("-" * 65)
    for name, r in results.items():
        if "macro_f1" in r:
            print(f"{name:25} {r['macro_f1']:8.3f} {r['bal_acc']:8.3f} {'—':>8} {r['permutation_p']:8.4f}")
        else:
            print(f"{name:25} {'—':>8} {'—':>8} {r['cramers_v']:8.3f} {r['p_value']:>8}")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
