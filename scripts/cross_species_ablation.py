#!/usr/bin/env python3
"""Cross-species 4-way ablation: adult marmoset + egyptian fruit bat.

Per-file oracle-parent HECB internal segmentation → whole/sub/pooled/relative-sequence
ablation, across two NEW taxa (LibriSpeech is the speech-regime reference, already done).
Each audio file IS one vocalization/bout → the whole file is the oracle parent event;
HECB (raw NBD inside the call) annotates internal sub-call boundaries.

  marmoset (Vocalizations, 96 kHz)  : label = call-type (Twitter/Tsik/Phee/Trill/Infant/Seep)
  bat (egyptian_fruit_bats, 250 kHz): label = Context (+ Emitter)

Why raw NBD (no typed classifier) here: in the per-file oracle case EVERY NBD proposal is
inside a real call (no out-of-call noise to gate), so the typed classifier's noise class is
empty — the internal mechanism reduces to NBD peaks inside the call. This is the honest
mechanistic test ("does internal structure help"), segmenter-agnostic + profile-free.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.representation.vocalisation import (  # noqa: E402
    build_vocalisation,
    format_4way_table,
    whole_sub_pooled_sequence_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import CallEvent, InternalSegment  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

MARMOSET = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
BAT = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
_PROPOSAL = {"floor_energy": 0.05, "floor_flux": 0.03, "debounce_ms": 30.0}
MIN_INTERNAL_MS = 120.0
MAX_INTERNAL = 6
NAMED_TYPES = {"Twitter", "Tsik", "Phee", "Trill", "Infant", "Seep"}


def nbd_internal_call_event(audio, sr, *, max_internal=MAX_INTERNAL, hop_ms=10.0):
    """Whole file = oracle parent; NBD peaks inside → internal sub-call boundaries."""
    dur_s = len(audio) / sr
    dur_ms = dur_s * 1000.0
    call = CallEvent(call_id="oracle", onset_s=0.0, offset_s=dur_s)
    if dur_ms < MIN_INTERNAL_MS:
        return call
    sig = compute_signals(audio, sr, hop_ms=hop_ms)
    props = generate_proposals(sig, **_PROPOSAL)
    # internal cuts = NBD peaks in the body (exclude 50ms edges = onset/offset)
    cuts = sorted(p.time_ms for p in props if 50 < p.time_ms < dur_ms - 50)
    cuts = cuts[:max_internal]
    edges = [0.0] + [c / 1000.0 for c in cuts] + [dur_s]
    call.internal_segments = [
        InternalSegment(
            parent_call_id="oracle", start_s=edges[k], end_s=edges[k + 1], boundary_type="internal"
        )
        for k in range(len(edges) - 1)
    ]
    return call


def load_marmoset(n, seed):
    """(flac_path, calltype) for the 6 named call-types."""
    rng = random.Random(seed)
    rows = []
    with open(MARMOSET / "Annotations.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = r["label"].strip()
            if lab in NAMED_TYPES and r["file_name"].strip():
                rows.append((lab, r["parent_name"].strip(), r["file_name"].strip()))
    rng.shuffle(rows)
    out = []
    for lab, parent, fname in rows:
        flac = MARMOSET / parent / fname
        if flac.exists():
            out.append((str(flac), lab))
        if len(out) >= n:
            break
    return out


def load_bat(n, seed, label_col="Context"):
    """(wav_path, label) for the bat corpus (Context or Emitter)."""
    rng = random.Random(seed)
    rows = []
    with open(BAT / "annotations.csv") as f:
        for r in csv.DictReader(f):
            rows.append((r["File Name"].strip(), r[label_col].strip()))
    rng.shuffle(rows)
    out = []
    for fname, lab in rows:
        wav = BAT / "audio" / fname
        if wav.exists():
            out.append((str(wav), f"c{lab}"))  # prefix to keep distinct
        if len(out) >= n:
            break
    return out


def run_corpus(name, files, sr, species, k, *, cache=None):
    fp_fn = lambda seg, s: afp.extract(seg, s, species=species).combined  # noqa: E731
    vocs, labels = [], []
    for fi, (path, lab) in enumerate(files):
        audio, _ = load_audio(path, target_sr=sr)
        if audio is None or audio.size < int(sr * 0.05):
            continue
        call = nbd_internal_call_event(audio, sr)
        voc = build_vocalisation(call, audio, sr, fingerprint_fn=fp_fn, species=species)
        if voc is not None:
            vocs.append(voc)
            labels.append(lab)
        if (fi + 1) % 500 == 0:
            print(f"  {name}: {fi + 1}/{len(files)} → {len(vocs)} vocs", flush=True)
    print(f"{name}: {len(vocs)} vocs | {dict(Counter(labels))}", flush=True)
    if len(vocs) < 50:
        print(f"  {name}: too few vocalisations, skipping", flush=True)
        return None
    result = whole_sub_pooled_sequence_ablation(vocs, labels, k=k)
    print(f"\n=== {name} ({species}, {sr / 1000:.0f} kHz) — 4-way ablation ===")
    print(format_4way_table(result))
    return result


def main():
    ap = argparse.ArgumentParser(description="Cross-species 4-way ablation")
    ap.add_argument("--n-marmoset", type=int, default=2000)
    ap.add_argument("--n-bat", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=== adult marmoset (Vocalizations, 96 kHz, call-type) ===", flush=True)
    marm = load_marmoset(args.n_marmoset, args.seed)
    print(f"  loaded {len(marm)} marmoset calls", flush=True)
    rm = run_corpus("marmoset_vocalizations", marm, 96000, "marmoset", args.k)

    print("\n=== egyptian fruit bat (250 kHz, Context) ===", flush=True)
    bat = load_bat(args.n_bat, args.seed, "Context")
    print(f"  loaded {len(bat)} bat bouts", flush=True)
    rb = run_corpus("egyptian_fruit_bat", bat, 250000, "bat", args.k)

    print("\n=== egyptian fruit bat (250 kHz, Emitter) ===", flush=True)
    bat_e = load_bat(args.n_bat, args.seed, "Emitter")
    re_ = run_corpus("egyptian_fruit_bat (emitter)", bat_e, 250000, "bat", args.k)

    # cross-species summary
    print("\n=== pooled > whole? (cross-species) ===")
    for name, res in [("marmoset(calltype)", rm), ("bat(context)", rb), ("bat(emitter)", re_)]:
        if res is None:
            continue
        w, p, s = res["whole"]["V"], res["pooled"]["V"], res["sequence"]["V"]
        print(f"  {name:24} whole={w:.3f}  pooled={p:.3f}  sequence={s:.3f}")
    print("CROSS-SPECIES ABLATION DONE")


if __name__ == "__main__":
    main()
