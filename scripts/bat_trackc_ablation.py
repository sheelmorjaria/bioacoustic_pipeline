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
        out.append(
            {
                "fp": fp,
                "context": int(r["Context"]),
                "emitter": int(r["Emitter"]),
                "addressee": int(r["Addressee"]),
                "emit_pre": int(r["Emitter pre-vocalization action"]),
                "addr_pre": int(r["Addressee pre-vocalization action"]),
                "emit_post": int(r["Emitter post-vocalization action"]),
                "addr_post": int(r["Addressee post-vocalization action"]),
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  extracted {len(out)}/{n} bouts", flush=True)
    return out


def onehot(val, max_val):
    v = np.zeros(max_val + 1, dtype=np.float32)
    v[val] = 1.0
    return v


def probe(X, y, name, groups=None):
    """LogReg probe. If groups given, use GroupShuffleSplit (held-out by group); else random."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    counts = Counter(y.tolist())
    keep = np.array([counts[c] >= 5 for c in y.tolist()])
    X, y = X[keep], y[keep]
    if groups is not None:
        groups = np.asarray(groups)[keep]
    n_cls = len(set(y.tolist()))
    chance = 1.0 / n_cls
    if n_cls < 2 or len(X) < 50:
        print(f"  {name:30} insufficient ({len(X)} samples, {n_cls} classes)")
        return None
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        idx_tr, idx_te = next(gss.split(X, y, groups))
        Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    acc = clf.score(sc.transform(Xte), yte)
    mf1 = f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")
    print(
        f"  {name:30} probe={acc:.3f}  macroF1={mf1:.3f}  chance={chance:.3f}  lift={acc / chance:.1f}×"
    )
    return acc / chance  # lift (for the comparison table)


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Bat Track C A-G context ablation + leakage controls")
    ap.add_argument("--n-bouts", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading {args.n_bouts} bat bouts + extracting fingerprints (250 kHz)...", flush=True)
    bouts = load_bat(args.n_bouts, seed=args.seed)
    print(f"  {len(bouts)} bouts loaded", flush=True)
    n = len(bouts)

    # Build feature sets
    max_emit = max(b["emitter"] for b in bouts)
    max_addr = max(b["addressee"] for b in bouts)
    max_act = max(max(b["emit_pre"], b["addr_pre"], b["emit_post"], b["addr_post"]) for b in bouts)

    acoustic = np.asarray([b["fp"] for b in bouts], dtype=np.float32)

    def build_social(bouts_list):
        return np.asarray(
            [
                np.concatenate([onehot(b["emitter"], max_emit), onehot(b["addressee"], max_addr)])
                for b in bouts_list
            ],
            dtype=np.float32,
        )

    behavior = np.asarray(
        [
            np.concatenate(
                [
                    onehot(b["emit_pre"], max_act),
                    onehot(b["addr_pre"], max_act),
                    onehot(b["emit_post"], max_act),
                    onehot(b["addr_post"], max_act),
                ]
            )
            for b in bouts
        ],
        dtype=np.float32,
    )
    social = build_social(bouts)
    acoustic_social = np.concatenate([acoustic, social], axis=1)
    acoustic_behavior = np.concatenate([acoustic, behavior], axis=1)
    combined = np.concatenate([acoustic, social, behavior], axis=1)

    contexts = np.asarray([b["context"] for b in bouts])
    emitters = np.asarray([b["emitter"] for b in bouts])

    # ── control 1: random split (the headline A-G matrix) ──
    print(f"\n{'=' * 70}\nCONTROL 1: random split ({n} bouts)\n{'=' * 70}")
    lifts = {"random": {}}
    for name, X in [
        ("A: acoustic", acoustic),
        ("B: social", social),
        ("D: acous+social", acoustic_social),
        ("E: acous+behav", acoustic_behavior),
        ("G: full", combined),
    ]:
        lifts["random"][name] = probe(X, contexts, name)
    print("  D≈G → social is the true complement. G≫a → context-dependent.")

    # ── control 2: social-label shuffle (leakage check) ──
    print(f"\n{'=' * 70}\nCONTROL 2: social-label shuffle (emitter/addressee permuted)\n{'=' * 70}")
    rng = random.Random(args.seed)
    shuffled = list(bouts)
    emit_vals = [b["emitter"] for b in shuffled]
    addr_vals = [b["addressee"] for b in shuffled]
    rng.shuffle(emit_vals)
    rng.shuffle(addr_vals)
    for b, e, a in zip(shuffled, emit_vals, addr_vals):
        b["emitter"], b["addressee"] = e, a
    social_sh = build_social(shuffled)
    acoustic_social_sh = np.concatenate([acoustic, social_sh], axis=1)
    combined_sh = np.concatenate([acoustic, social_sh, behavior], axis=1)
    lifts["shuffle"] = {}
    for name, X in [
        ("A: acoustic", acoustic),
        ("B: social(shuf)", social_sh),
        ("D: acous+social(shuf)", acoustic_social_sh),
        ("G: full(shuf)", combined_sh),
    ]:
        lifts["shuffle"][name] = probe(X, contexts, name)
    print("  If D(shuf) ≈ A → D's lift came from genuine social signal, not a confound.")

    # ── control 3: emitter-held-out (generalization check) ──
    print(f"\n{'=' * 70}\nCONTROL 3: emitter-held-out (GroupShuffleSplit by emitter)\n{'=' * 70}")
    lifts["emit-holdout"] = {}
    for name, X in [
        ("A: acoustic", acoustic),
        ("B: social", social),
        ("D: acous+social", acoustic_social),
        ("G: full", combined),
    ]:
        lifts["emit-holdout"][name] = probe(X, contexts, name, groups=emitters)
    print("  If D > A/B under held-out → social-grounding generalizes to unseen callers.")

    # ── comparison table ──
    print(f"\n{'=' * 70}\nLEAKAGE / GENERALIZATION COMPARISON TABLE (CONTEXT target)\n{'=' * 70}")
    print(f"  {'split':22} {'A acoustic':>11} {'B social':>9} {'D acous+soc':>12} {'G full':>8}")
    print("  " + "-" * 66)
    labels = {
        "random": "random split",
        "shuffle": "social-shuffle",
        "emit-holdout": "emitter-held-out",
    }
    for ctrl in ["random", "shuffle", "emit-holdout"]:
        d = lifts[ctrl]
        row = [labels[ctrl]]
        for key in ["A: acoustic", "B: social(shuf)", "D: acous+social(shuf)", "G: full(shuf)"]:
            row.append(d.get(key, d.get(key.replace("(shuf)", ""))))
        # normalise keys (shuffle uses different names)
        a_val = d.get("A: acoustic")
        b_key = "B: social(shuf)" if ctrl == "shuffle" else "B: social"
        d_key = "D: acous+social(shuf)" if ctrl == "shuffle" else "D: acous+social"
        g_key = "G: full(shuf)" if ctrl == "shuffle" else "G: full"

        def fmt(v):
            return f"{v:.1f}×" if v else "  n/a"

        print(
            f"  {labels[ctrl]:22} {fmt(a_val):>11} {fmt(d.get(b_key)):>9} "
            f"{fmt(d.get(d_key)):>12} {fmt(d.get(g_key)):>8}"
        )

    print("\n  Read: shuffle should collapse D→A (social becomes noise).")
    print("        Emitter-held-out should preserve D > A (generalization to unseen callers).")
    print("BAT TRACK C ABLATION + CONTROLS DONE")


if __name__ == "__main__":
    main()
