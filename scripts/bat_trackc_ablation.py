#!/usr/bin/env python3
"""Track C bat context ablation: does acoustic + behavioral > either alone?

The semantic ceiling showed bat context is weak from acoustics alone (V~0.15) AND from
behavioral covariates alone (2.1× lift). This tests the COMBINATION: does acoustic +
social + behavioral together recover context substantially better than either modality
alone? If yes, context needs multimodal Track-C features. If no, context is genuinely
hard from any single-bout features.

  1. Acoustic only:   whole fingerprint (43-D)
  2. Social only:     emitter + addressee (one-hot)
  3. Behavior only:   pre/post actions (4 codes)
  4. Combined:        acoustic + social + behavior

Targets: Context, Emitter, Addressee. Metrics: probe acc, macro-F1.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

BAT = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
SR = 250000


def load_bat(n, seed):
    """Load bat bouts with ALL annotation fields + extract acoustic fingerprint."""
    rng = random.Random(seed)
    rows = []
    with open(BAT / "annotations.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rng.shuffle(rows)
    out = []
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="bat").combined  # noqa: E731
    for i, r in enumerate(rows):
        if len(out) >= n:
            break
        wav = BAT / "audio" / r["File Name"].strip()
        if not wav.exists():
            continue
        audio, _ = load_audio(str(wav), target_sr=SR)
        if audio is None or audio.size < int(SR * 0.05):
            continue
        fp = np.asarray(afp.extract(audio, SR, species="bat").combined, dtype=np.float32)
        if not np.any(fp):
            continue
        out.append({
            "fp": fp,
            "context": int(r["Context"]),
            "emitter": int(r["Emitter"]),
            "addressee": int(r["Addressee"]),
            "emit_pre": int(r["Emitter pre-vocalization action"]),
            "addr_pre": int(r["Addressee pre-vocalization action"]),
            "emit_post": int(r["Emitter post-vocalization action"]),
            "addr_post": int(r["Addressee post-vocalization action"]),
        })
        if (i + 1) % 200 == 0:
            print(f"  extracted {len(out)}/{n} bouts", flush=True)
    return out


def onehot(val, max_val):
    v = np.zeros(max_val + 1, dtype=np.float32)
    v[val] = 1.0
    return v


def probe(X, y, name):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    counts = Counter(y.tolist())
    keep = np.array([counts[c] >= 5 for c in y.tolist()])
    X, y = X[keep], y[keep]
    n_cls = len(set(y.tolist()))
    chance = 1.0 / n_cls
    if n_cls < 2 or len(X) < 50:
        print(f"  {name:30} insufficient ({len(X)} samples, {n_cls} classes)")
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    acc = clf.score(sc.transform(Xte), yte)
    mf1 = f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")
    print(f"  {name:30} probe={acc:.3f}  macroF1={mf1:.3f}  chance={chance:.3f}  lift={acc/chance:.1f}×")


def main():
    n = 1500
    print(f"Loading {n} bat bouts + extracting fingerprints (250 kHz)...", flush=True)
    bouts = load_bat(n, seed=0)
    print(f"  {len(bouts)} bouts loaded", flush=True)

    # Build feature sets
    fps = [b["fp"] for b in bouts]
    max_emit = max(b["emitter"] for b in bouts)
    max_addr = max(b["addressee"] for b in bouts)
    max_act = max(max(b["emit_pre"], b["addr_pre"], b["emit_post"], b["addr_post"]) for b in bouts)

    acoustic = np.asarray(fps, dtype=np.float32)
    social = np.asarray([np.concatenate([onehot(b["emitter"], max_emit), onehot(b["addressee"], max_addr)])
                         for b in bouts], dtype=np.float32)
    behavior = np.asarray([np.concatenate([onehot(b["emit_pre"], max_act), onehot(b["addr_pre"], max_act),
                                            onehot(b["emit_post"], max_act), onehot(b["addr_post"], max_act)])
                           for b in bouts], dtype=np.float32)
    combined = np.concatenate([acoustic, social, behavior], axis=1)

    contexts = [b["context"] for b in bouts]
    emitters = [b["emitter"] for b in bouts]
    addressees = [b["addressee"] for b in bouts]

    print(f"\n{'='*70}")
    print(f"Track C Bat Context Ablation ({len(bouts)} bouts)")
    print(f"{'='*70}")

    for target_name, target in [("CONTEXT", contexts), ("EMITTER", emitters), ("ADDRESSEE", addressees)]:
        print(f"\n=== Target: {target_name} ===")
        probe(acoustic, target, "acoustic-only (fp)")
        probe(social, target, "social-only (emit+addr)")
        probe(behavior, target, "behavior-only (actions)")
        probe(combined, target, "combined (acoustic+social+behav)")

    print(f"\n=== Track C read ===")
    print("  If combined >> acoustic-only for CONTEXT:")
    print("    → context requires multimodal/social grounding (strongest Track C claim)")
    print("  If combined ≈ acoustic-only:")
    print("    → context is genuinely hard from any single-bout features")
    print("BAT TRACK C ABLATION DONE")


if __name__ == "__main__":
    main()
