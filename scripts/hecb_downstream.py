#!/usr/bin/env python3
"""HECB-Seg downstream: whole/sub-call/pooled call-type recovery (standard ablation).

Thin wrapper around ``representation.subcall`` (the canonical whole/sub/pooled
ablation). Loads the cached HECB CRNN + typed classifier, runs HECB inference per
MeerKAT clip, matches CRNN call events to GT call-types, builds whole/sub/pooled
representations, and runs the standardized ablation.

Result framing (validated 2026-06-21): pooled (whole+sub) > whole > sub-call-alone.
Sub-call features augment whole-call representations when pooled but are weaker alone —
internal segmentation augments, not replaces, whole-event representations.

Usage:
  python3 scripts/hecb_downstream.py --n-clips 1500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.representation.subcall import (  # noqa: E402
    build_subcall_representations,
    format_ablation_table,
    whole_sub_pooled_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import fit_hecb_priors, hecb_decode  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

# Reuse the cached CRNN + typed classifier + inference helpers from the training script.
import train_hecb_seg as H  # noqa: E402

LBL, WAV, SR = H.LBL, H.WAV, H.SR


def read_calls(h5path):
    """Parse one label .h5 -> [(start_s, end_s, call_type), ...]."""
    with h5py.File(h5path, "r") as h:
        st = np.asarray(h["start_time_lbl"][:]).ravel()
        et = np.asarray(h["end_time_lbl"][:]).ravel()
        lbl = np.asarray(h["lbl"][:]).ravel()
    out = []
    for s, e, lab in zip(st, et, lbl):
        s = float(s)
        if not np.isfinite(s):
            continue
        e = float(e) if np.isfinite(e) else s + 0.08
        if isinstance(lab, bytes):
            ls = lab.decode()
        elif np.isscalar(lab) and np.isfinite(lab):
            ls = str(int(lab))
        else:
            continue
        out.append((s, e, ls))
    return out


def sample_clips(n, seed):
    rng = random.Random(seed)
    files = sorted(glob.glob(str(LBL / "*.h5")))
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        calls = read_calls(f)
        if not calls:
            continue
        stem = Path(f).stem
        if (WAV / f"{stem}.wav").exists():
            out.append((stem, calls))
    return out


def match_events_to_calls(events_onset_ms, calls, tol_ms=100.0):
    """Greedy one-to-one match CRNN event onsets → GT calls. Returns {ev_idx: call_type}."""
    used = set()
    matched = {}
    gt_ms = [(s * 1000.0, lab) for s, _e, lab in calls]
    for ei, pon in enumerate(events_onset_ms):
        best, bd = None, tol_ms
        for gi, (g, lab) in enumerate(gt_ms):
            if gi in used:
                continue
            d = abs(pon - g)
            if d < bd:
                bd, best = d, gi
        if best is not None:
            used.add(best)
            matched[ei] = gt_ms[best][1]
    return matched


def main():
    ap = argparse.ArgumentParser(description="HECB-Seg downstream call-type recovery")
    ap.add_argument("--n-clips", type=int, default=1500)
    ap.add_argument("--k", type=int, default=12, help="k-means token count")
    ap.add_argument("--tol-ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # ─── Load cached CRNN + typed classifier ───
    cache = ROOT / "checkpoints" / "hecb"
    crnn = H.CRNNDetector().to(device)
    crnn.load_state_dict(torch.load(cache / "crnn.pt", map_location=device))
    crnn.eval()
    import pickle  # noqa: PLC0415

    with open(cache / "typed.pkl", "rb") as fh:
        _X, _y, clf = pickle.load(fh)
    print("loaded cached CRNN + typed classifier", flush=True)

    # Classifier was trained WITH fingerprint delta (32-D) → inference must match.
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="meerkat").combined  # noqa: E731
    profile = load_species_profiles()["meerkat"]

    clips = sample_clips(args.n_clips, args.seed)
    print(f"{len(clips)} clips with call-type labels", flush=True)

    all_spans = [[(s, e) for s, e, _l in calls] for _, calls in clips]
    priors = fit_hecb_priors(all_spans)

    # ─── Per clip: HECB inference → CallEvents → subcall reps (matched calls only) ───
    all_reps, all_labels = [], []
    for ci, (stem, calls) in enumerate(clips):
        try:
            audio, _ = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
        except Exception:
            continue
        proba = H.crnn_inference(crnn, audio, device)
        crnn_events = H.proba_to_events(proba)
        props = H.event_conditioned_proposals(audio, SR, proba, crnn_events, profile, fp_fn)
        typed = [(t, clf.predict_proba(ev.reshape(1, -1))[0]) for t, _p, ev in props]
        events = hecb_decode(crnn_events, typed, priors, file_id=stem)
        matched = match_events_to_calls([e.onset_s * 1000 for e in events], calls, args.tol_ms)
        matched_events = [e for ei, e in enumerate(events) if ei in matched]
        matched_labels = [matched[ei] for ei, _ in enumerate(events) if ei in matched]
        reps = build_subcall_representations(matched_events, audio, SR, fp_fn)
        # reps aligns 1:1 with matched_events (skips only too-short/silent calls)
        all_reps.extend(reps)
        all_labels.extend(matched_labels[: len(reps)])
        if (ci + 1) % 200 == 0:
            print(f"  {ci + 1}/{len(clips)} clips | {len(all_reps)} calls", flush=True)

    print(
        f"\nmatched labeled calls: {len(all_reps)} | call types: {dict(Counter(all_labels))}",
        flush=True,
    )

    result = whole_sub_pooled_ablation(all_reps, all_labels, k=args.k)
    print(f"\n=== Whole / sub-call / pooled ablation (k={args.k}, MeerKAT) ===")
    print(format_ablation_table(result))
    w, p = result["whole"]["V"], result["pooled"]["V"]
    print(
        f"\n  pooled V ({p:.3f}) {'>' if p > w else '<='} whole V ({w:.3f}) → "
        f"sub-call structure {'AUGMENTS' if p > w else 'does not augment'} whole-call recovery"
    )
    print("HECB DOWNSTREAM DONE")


if __name__ == "__main__":
    main()
