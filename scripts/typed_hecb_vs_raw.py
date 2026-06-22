#!/usr/bin/env python3
"""Typed HECB vs raw NBD on adult marmoset — does typed gating recover the pooled advantage?

The cross-species ablation (`cross_species_ablation.py`) used raw NBD internal boundaries
and found pooled ≈ whole (no advantage). The hypothesis: typed HECB gating (a learned
classifier that filters spurious proposals) produces cleaner internal boundaries, recovering
the pooled > whole advantage seen with typed HECB on MeerKAT / InfantMarmosetsVox.

This runs BOTH raw-NBD AND typed-HECB on the SAME eval files → direct comparison.

Typed classifier: GBT over proposal_features (13-D NBD scalars, SR-agnostic) + 2 geometry
features (relative position, distance to nearest edge). Labels: onset (near file start) /
offset (near file end) / internal (body) / noise. hecb_decode(snap=False) for typed mode.

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
from bioacoustic_pipeline.representation.vocalisation import (  # noqa: E402
    build_vocalisation,
    format_4way_table,
    whole_sub_pooled_sequence_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import (  # noqa: E402
    CallEvent,
    HECBDurationPriors,
    InternalSegment,
    hecb_decode,
)
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

import cross_species_ablation as XSA  # reuse load_marmoset + nbd_internal_call_event  # noqa: E402

SR = 96000
_PROPOSAL = {"floor_energy": 0.05, "floor_flux": 0.03, "debounce_ms": 30.0}
INTERNAL = 3


def _geom_feats(prop, dur_ms):
    """2-D geometry: relative position + distance to nearest edge."""
    rel = prop.time_ms / max(dur_ms, 1.0)
    dist_edge = min(prop.time_ms, dur_ms - prop.time_ms)
    return np.array([rel, float(np.log1p(max(dist_edge, 0.0)))])


def train_typed_clf(train_files, n_train=2000):
    """GBT over NBD scalars (13) + geometry (2) → onset/offset/internal/noise."""
    rng = random.Random(0)
    pool = list(train_files)
    rng.shuffle(pool)
    X, y = [], []
    for path, _lab in pool[:n_train]:
        audio, _ = load_audio(path, target_sr=SR)
        if audio is None or audio.size < int(SR * 0.05):
            continue
        dur_ms = len(audio) / SR * 1000.0
        sig = compute_signals(audio, SR, hop_ms=10.0)
        for p in generate_proposals(sig, **_PROPOSAL):
            X.append(np.concatenate([proposal_features(p), _geom_feats(p, dur_ms)]))
            if p.time_ms < 80:
                y.append(1)  # ONSET
            elif p.time_ms > dur_ms - 80:
                y.append(2)  # OFFSET
            elif 80 < p.time_ms < dur_ms - 80:
                y.append(3)  # INTERNAL
            else:
                y.append(0)  # NOISE
    X, y = np.asarray(X), np.asarray(y)
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=0
    )
    clf.fit(X, y)
    print(
        f"  typed clf: {len(X)} proposals | "
        + " ".join(
            f"{['noise', 'onset', 'offset', 'internal'][k]}={v}"
            for k, v in sorted(Counter(y.tolist()).items())
        ),
        flush=True,
    )
    return clf


def typed_hecb_call_event(audio, sr, clf):
    """Typed-HECB oracle: NBD → typed classify → hecb_decode (internal-typed boundaries only)."""
    dur_s = len(audio) / sr
    dur_ms = dur_s * 1000.0
    sig = compute_signals(audio, sr, hop_ms=10.0)
    props = generate_proposals(sig, **_PROPOSAL)
    typed = []
    for p in props:
        ev = np.concatenate([proposal_features(p), _geom_feats(p, dur_ms)])
        proba = clf.predict_proba(ev.reshape(1, -1))[0]
        p4 = np.zeros(4)
        for ci, cls in enumerate(clf.classes_):
            p4[int(cls)] = proba[ci]
        typed.append((p.time_ms, p4))
    return hecb_decode(
        [(0.0, dur_ms, 1.0)],
        typed,
        HECBDurationPriors(),
        snap=False,
        gate_durations=False,
        min_internal_call_ms=120.0,
        file_id="oracle",
    )


def build_vocs(files, sr, species, call_event_fn, *, clf=None):
    """Build DualStreamVocalisations via the given call-event function (raw or typed)."""
    fp_fn = lambda seg, s: afp.extract(seg, s, species=species).combined  # noqa: E731
    vocs, labels = [], []
    for fi, (path, lab) in enumerate(files):
        audio, _ = load_audio(path, target_sr=sr)
        if audio is None or audio.size < int(sr * 0.05):
            continue
        if clf is not None:
            call = typed_hecb_call_event(audio, sr, clf)
            call = call[0] if call else None
        else:
            call = XSA.nbd_internal_call_event(audio, sr)
        if call is None:
            continue
        voc = build_vocalisation(call, audio, sr, fingerprint_fn=fp_fn, species=species)
        if voc is not None:
            vocs.append(voc)
            labels.append(lab)
        if (fi + 1) % 500 == 0:
            print(f"    {fi + 1}/{len(files)} → {len(vocs)} vocs", flush=True)
    return vocs, labels


def main():
    ap = argparse.ArgumentParser(description="Typed HECB vs raw NBD on adult marmoset")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_files = XSA.load_marmoset(args.n_train + args.n_eval, args.seed)
    train_files, eval_files = all_files[: args.n_train], all_files[args.n_train :]
    print(f"{len(train_files)} train / {len(eval_files)} eval marmoset calls", flush=True)

    # ─── Train typed classifier ───
    print("Training typed classifier...", flush=True)
    clf = train_typed_clf(train_files, n_train=args.n_train)

    # ─── Raw NBD ablation (baseline from cross_species_ablation) ───
    print("\n=== RAW NBD internal boundaries ===", flush=True)
    rv, rl = build_vocs(eval_files, SR, "marmoset", XSA.nbd_internal_call_event)
    print(f"  {len(rv)} vocs | {dict(Counter(rl))}", flush=True)
    raw_res = whole_sub_pooled_sequence_ablation(rv, rl, k=args.k)
    print(format_4way_table(raw_res))

    # ─── Typed HECB ablation ───
    print("\n=== TYPED HECB internal boundaries ===", flush=True)
    tv, tl = build_vocs(eval_files, SR, "marmoset", None, clf=clf)
    print(f"  {len(tv)} vocs | {dict(Counter(tl))}", flush=True)
    typed_res = whole_sub_pooled_sequence_ablation(tv, tl, k=args.k)
    print(format_4way_table(typed_res))

    # ─── Direct comparison ───
    print("\n=== typed HECB vs raw NBD — pooled > whole? ===")
    for name, res in [("raw NBD", raw_res), ("typed HECB", typed_res)]:
        w, p, s = res["whole"]["V"], res["pooled"]["V"], res["sequence"]["V"]
        wi = res["n_with_internal"]
        print(
            f"  {name:12} (internal: {wi})  whole={w:.3f}  pooled={p:.3f}  sequence={s:.3f}  "
            f"pooled-whole={p - w:+.3f}"
        )
    print("TYPED HECB VS RAW DONE")


if __name__ == "__main__":
    main()
