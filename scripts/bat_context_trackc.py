#!/usr/bin/env python3
"""Bat context Track C: do behavioral covariates predict context beyond the acoustic ceiling?

The cross-species ablation showed bat context recovery is near-chance from acoustics alone
(V~0.15, probe~0.55 with 2 dominant classes). This tests whether behavioral/social
covariates (emitter, addressee, pre/post actions from annotations.csv) predict context
substantially better — confirming context is a Track-C (behavioral) label, not acoustic.

If behavioral >> acoustic: context requires multimodal/Track-C features (semantic ceiling).
If behavioral ≈ acoustic: context is genuinely hard from any modality in this setup.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAT = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")


def load_bat_annotations(n=9999):
    """annotations.csv → [(context, emitter, addressee, emit_pre, addr_pre, emit_post, addr_post)]."""
    rows = []
    with open(BAT / "annotations.csv") as f:
        for r in csv.DictReader(f):
            rows.append((
                int(r["Context"]), int(r["Emitter"]), int(r["Addressee"]),
                int(r["Emitter pre-vocalization action"]),
                int(r["Addressee pre-vocalization action"]),
                int(r["Emitter post-vocalization action"]),
                int(r["Addressee post-vocalization action"]),
            ))
    return rows[:n]


def probe_recovery(features, labels, name):
    """LogReg probe: context recovery from features."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    counts = Counter(y.tolist())
    keep = np.array([counts[c] >= 5 for c in y.tolist()])
    X, y = X[keep], y[keep]
    n_classes = len(set(y.tolist()))
    dist = dict(sorted(Counter(y.tolist()).items()))
    print(f"  {name:32} n={len(X)} classes={n_classes} | dist={dist}")
    if n_classes < 2 or len(X) < 50:
        print(f"    insufficient data")
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    acc = clf.score(sc.transform(Xte), yte)
    mf1 = f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")
    chance = 1.0 / n_classes
    print(f"    probe_acc={acc:.4f}  macroF1={mf1:.4f}  chance={chance:.4f}  lift={acc/chance:.1f}×")
    return acc, mf1


def main():
    ap = argparse.ArgumentParser(description="Bat context Track C (behavioral covariates)")
    ap.add_argument("--n", type=int, default=9999)
    args = ap.parse_args()

    rows = load_bat_annotations(args.n)
    print(f"loaded {len(rows)} bat bouts", flush=True)
    contexts = [r[0] for r in rows]
    print(f"context distribution: {dict(sorted(Counter(contexts).items()))}\n")

    # ─── Acoustic-only ceiling (from cross-species ablation, for reference) ───
    print("=== ACOUSTIC CEILING (reference from cross_species_ablation) ===")
    print("  bat context: V=0.152  probe=0.556  (pooled, 2493 bouts)")
    print("  → near-chance for 2 dominant classes; semantic ceiling confirmed\n")

    # ─── Behavioral features: emitter + addressee ───
    print("=== BEHAVIORAL: emitter + addressee ===")
    probe_recovery(
        [[r[1], r[2]] for r in rows],  # emitter, addressee
        contexts, "emitter + addressee")

    # ─── Behavioral features: pre/post actions ───
    print("\n=== BEHAVIORAL: pre/post actions ===")
    probe_recovery(
        [[r[3], r[4], r[5], r[6]] for r in rows],  # 4 action codes
        contexts, "pre/post actions (4)")

    # ─── Full behavioral: emitter + addressee + actions ───
    print("\n=== BEHAVIORAL: all social covariates ===")
    probe_recovery(
        [[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows],
        contexts, "all behavioral (6)")

    # ─── Emitter-only (is context just who is vocalizing?) ───
    print("\n=== BEHAVIORAL: emitter-only ===")
    probe_recovery(
        [[r[1]] for r in rows],
        contexts, "emitter-only")

    print("\n=== Track C read ===")
    print("  If behavioral >> acoustic: context is a SOCIAL label (Track C),")
    print("  not recoverable from acoustics alone — needs multimodal covariates.")
    print("  If behavioral ≈ acoustic: context is genuinely hard from any modality.")
    print("BAT CONTEXT TRACK C DONE")


if __name__ == "__main__":
    main()
