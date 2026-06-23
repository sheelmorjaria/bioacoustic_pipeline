#!/usr/bin/env python3
"""Stereotypy vs species-level template — does a caller-invariant call-type template exist? (Step 11)

For each call type, build a MEAN-FINGERPRINT template (averaged across all callers). Then test
whether the template retrieves the call type across CALLERS better than raw cross-caller nearest
neighbour. If template > NN, a species-level acoustic template exists (the call type is
stereotyped across individuals). If NN > template, calls are caller-bound variants (no shared
template — the rook §26 result).

Uses the marmoset corpus (5 call types, multi-session, 871k calls) — the strongest call-type
recovery species (§18: V=0.606, probe=0.829). Firewall-clean (all-species fingerprint).

Run:  python scripts/stereotypy_vs_template.py --n-per-type 200

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

MAR_VOC = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
MAR_CALL_TYPES = ("Phee", "Twitter", "Trill", "Tsik", "Seep")
MAR_SR = 96000


def load_calls(n_per_type, seed):
    """Load n calls per type, balanced, with caller (parent_name) labels."""
    idx = {}
    with (MAR_VOC / "Annotations.tsv").open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fn = (row.get("file_name") or "").strip()
            label = (row.get("label") or "").strip()
            parent = (row.get("parent_name") or "").strip()
            if fn and label in MAR_CALL_TYPES:
                idx[fn] = (parent, label)
    by_type = defaultdict(list)
    for fn, (parent, label) in idx.items():
        by_type[label].append((fn, parent))
    rng = random.Random(seed)
    recs = []
    for ct in MAR_CALL_TYPES:
        pool = list(by_type.get(ct, []))
        rng.shuffle(pool)
        for fn, parent in pool[:n_per_type]:
            path = MAR_VOC / parent / fn
            if not path.exists():
                continue
            try:
                audio, sr = load_audio(str(path), target_sr=MAR_SR)
            except Exception:  # noqa: BLE001
                continue
            if audio is None or len(audio) < int(0.1 * MAR_SR):
                continue
            fp = np.asarray(afp.extract(audio, sr, species="marmoset").combined, dtype=np.float32)
            if np.any(fp):
                recs.append({"fp": fp, "label": ct, "caller": parent})
    return recs


def main():
    ap = argparse.ArgumentParser(description="Stereotypy vs template (Step 11)")
    ap.add_argument("--n-per-type", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load_calls(args.n_per_type, args.seed)
    print(
        f"loaded {len(recs)} calls "
        f"({dict((ct, sum(1 for r in recs if r['label'] == ct)) for ct in MAR_CALL_TYPES)})",
        flush=True,
    )
    if len(recs) < 50:
        print("too few calls — aborting")
        return

    fps = np.asarray([r["fp"] for r in recs], dtype=np.float32)
    labels = [r["label"] for r in recs]
    callers = [r["caller"] for r in recs]
    fps_n = fps / (np.linalg.norm(fps, axis=1, keepdims=True) + 1e-12)

    # Per-call-type template: mean fingerprint (normalised)
    templates = {}
    for ct in MAR_CALL_TYPES:
        mask = [i for i, lab in enumerate(labels) if lab == ct]
        if len(mask) >= 5:
            t = np.mean(fps_n[mask], axis=0)
            templates[ct] = t / (np.linalg.norm(t) + 1e-12)

    # Cross-caller retrieval: for each query, gallery = different caller
    # Method 1: raw NN (closest fp from a different caller)
    # Method 2: template (closest template)
    nn_top1 = template_top1 = 0
    n_eval = 0
    for i in range(len(recs)):
        qc = callers[i]
        ql = labels[i]
        # gallery: all recs from different callers
        gal = [j for j in range(len(recs)) if callers[j] != qc]
        if len(gal) < 5:
            continue
        # NN: cosine to gallery fps
        sims = fps_n[gal] @ fps_n[i]
        nn_label = labels[gal[int(np.argmax(sims))]]
        nn_top1 += nn_label == ql
        # Template: cosine to each template
        if templates:
            best_ct = max(templates, key=lambda ct: float(templates[ct] @ fps_n[i]))
            template_top1 += best_ct == ql
        n_eval += 1

    nn_acc = nn_top1 / max(n_eval, 1)
    tpl_acc = template_top1 / max(n_eval, 1)
    chance = 1.0 / len(MAR_CALL_TYPES)
    print(
        f"\n{'=' * 80}\nSTEREOTYPY vs TEMPLATE — marmoset cross-caller retrieval (Step 11)\n{'=' * 80}"
    )
    print(f"  n_calls={len(recs)} n_eval={n_eval} chance={chance:.2f}")
    print(f"  cross-caller NN top-1:    {nn_acc:.3f} ({nn_acc / chance:.1f}× chance)")
    print(f"  template top-1:          {tpl_acc:.3f} ({tpl_acc / chance:.1f}× chance)")
    print(f"  Δ (template − NN):       {tpl_acc - nn_acc:+.3f}")
    if tpl_acc > nn_acc:
        print("\n  ⇒ template > NN: a species-level call-type template exists (calls are")
        print("    stereotyped across individuals). The mean-fingerprint generalises.")
    else:
        print("\n  ⇒ NN ≥ template: calls are caller-bound variants (no shared template).")
        print("    The species-level mean doesn't generalise as well as individual exemplars.")

    out = ROOT / "outputs" / "stereotypy_vs_template.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(
            {
                "n_calls": len(recs),
                "nn_top1": round(nn_acc, 3),
                "template_top1": round(tpl_acc, 3),
                "delta": round(tpl_acc - nn_acc, 3),
                "chance": round(chance, 3),
            },
            fh,
            indent=2,
        )
    print(f"\nsaved {out}")
    print("\nSTEREOTYPY vs TEMPLATE DONE")


if __name__ == "__main__":
    main()
