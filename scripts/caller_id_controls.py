#!/usr/bin/env python3
"""Caller-ID controls on InfantMarmosetsVox — does the pooled advantage survive confounds?

The novel finding (pooled wins all 4 metrics on caller-ID) could be confounded by:
  1. Call-type distribution (different callers produce different call types)
  2. Duration (callers differ in call duration)
  3. Session/recording context

Controls:
  A. Within-call-type: caller-ID recovery WITHIN a single call type (Phee/Twitter).
     If pooled still wins within-type, the caller signal is not call-type confounding.
  B. Duration-only baseline: caller recovery from [duration] alone.
  C. Calltype-only baseline: caller recovery from one-hot calltype.
  D. Within-call-type + duration regressed: caller residual after removing duration.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.representation.subcall import (  # noqa: E402
    format_ablation_table,
    whole_sub_pooled_ablation,
)
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

import hecb_oracle_marmoset as OM  # reuse loaders + typed clf + oracle  # noqa: E402

CALLTYPE_NAMES = {
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
}


def baseline_recovery(features, labels, k, name):
    """Caller recovery from trivial features (duration, one-hot calltype)."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    y = np.asarray(labels)
    counts = Counter(y.tolist())
    keep_mask = np.array([counts[c] >= 5 for c in y.tolist()])
    feats, y = np.asarray(features)[keep_mask], y[keep_mask]
    if len(set(y.tolist())) < 2 or len(feats) < 20:
        print(f"  {name}: insufficient data ({len(feats)} samples, {len(set(y.tolist()))} callers)")
        return
    try:
        Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.3, random_state=0, stratify=y)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        acc = clf.score(sc.transform(Xte), yte)
        mf1 = f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")
        print(
            f"  {name:28} probe_acc={acc:.4f}  macroF1={mf1:.4f}  (n={len(feats)}, callers={len(set(y.tolist()))})"
        )
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: failed ({e})")


def main():
    ap = argparse.ArgumentParser(description="Caller-ID controls on InfantMarmosetsVox")
    ap.add_argument("--n-train-clf", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print("loading GT calls...", flush=True)
    all_calls = OM.load_all_calls(n_files=999)

    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="marmoset").combined  # noqa: E731

    # ─── Build vocs on random eval subset (all callers) ───
    rng = random.Random(1)
    pool = list(all_calls)
    rng.shuffle(pool)
    half = len(pool) // 2
    train_pool = pool[:half]  # random 50% for clf training (all callers)
    eval_pool = pool[half:]  # held-out 50% for eval (all callers)
    rng.shuffle(eval_pool)
    print(
        f"  train clf: {len(train_pool)} calls (twin1-3) | eval: {len(eval_pool[: args.n_eval])} calls",
        flush=True,
    )

    clf = OM.train_typed_clf(train_pool, n_train=args.n_train_clf)
    eval_calls = eval_pool[: args.n_eval]
    vocs, ct_labels, caller_labels = OM.oracle_call_events(eval_calls, clf, fp_fn)
    print(
        f"  {len(vocs)} vocs | callers: {dict(sorted(Counter(caller_labels).items()))}", flush=True
    )
    print(f"  calltypes: {dict(sorted(Counter(ct_labels).items()))}", flush=True)

    # ─── Baseline: full caller-ID ablation (replicates OM random split) ───
    print(f"\n=== BASELINE: caller-ID (all call types) ===")
    result = whole_sub_pooled_ablation(vocs, caller_labels, k=args.k)
    print(format_ablation_table(result))

    # ─── Control A: within-call-type caller-ID ───
    for ct in [1, 2]:  # Phee, Twitter (dominant types)
        mask = np.array([ct_labels[i] == ct for i in range(len(ct_labels))])
        sub_vocs = [vocs[i] for i in range(len(vocs)) if mask[i]]
        sub_callers = [caller_labels[i] for i in range(len(vocs)) if mask[i]]
        n_callers = len(set(sub_callers))
        print(
            f"\n=== CONTROL A: caller-ID within {CALLTYPE_NAMES.get(ct, ct)} "
            f"({len(sub_vocs)} calls, {n_callers} callers) ==="
        )
        if len(sub_vocs) >= 100 and n_callers >= 3:
            sub_res = whole_sub_pooled_ablation(sub_vocs, sub_callers, k=min(args.k, n_callers))
            print(format_ablation_table(sub_res))
        else:
            print(f"  insufficient data (need ≥100 calls, ≥3 callers)")

    # ─── Control B: duration-only baseline ───
    print(f"\n=== CONTROL B: duration-only caller baseline ===")
    durations = np.array([[(v.end_ms - v.start_ms) / 1000.0] for v in vocs], dtype=np.float32)
    baseline_recovery(durations, caller_labels, args.k, "duration-only")

    # ─── Control C: calltype-only baseline ───
    print(f"\n=== CONTROL C: calltype-only caller baseline ===")
    n_types = max(ct_labels) + 1
    ct_onehot = np.zeros((len(vocs), n_types), dtype=np.float32)
    for i, ct in enumerate(ct_labels):
        ct_onehot[i, ct] = 1.0
    baseline_recovery(ct_onehot, caller_labels, args.k, "calltype-onehot")

    # ─── Control D: whole-fp with duration regressed out ───
    print(f"\n=== CONTROL D: whole-fp caller after regressing out duration ===")
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415

    whole_vecs = np.asarray([v.whole_fingerprint for v in vocs], dtype=np.float32)
    # regress duration from each fp dimension
    dur_col = durations.ravel()
    resid = whole_vecs.copy()
    for d in range(whole_vecs.shape[1]):
        reg = LinearRegression().fit(dur_col.reshape(-1, 1), whole_vecs[:, d])
        resid[:, d] = whole_vecs[:, d] - reg.predict(dur_col.reshape(-1, 1))
    baseline_recovery(resid, caller_labels, args.k, "whole-fp (dur-regressed)")

    # ─── Summary ───
    print(f"\n=== caller-ID controls summary ===")
    w = result["whole"]["V"]
    p = result["pooled"]["V"]
    print(f"  full:           whole V={w:.3f}  pooled V={p:.3f}  pooled-whole={p - w:+.3f}")
    print(f"  (controls A-D test whether the pooled advantage survives confound removal)")
    print("CALLER ID CONTROLS DONE")


if __name__ == "__main__":
    main()
