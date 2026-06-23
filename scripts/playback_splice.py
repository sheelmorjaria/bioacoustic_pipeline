#!/usr/bin/env python3
"""DSP Audio Splicer demo — generate H0-CMC counterfactual WAV stimuli from bat bouts.

Track C (Architecture v1.0). Loads Egyptian-fruit-bat bouts, segments them into sub-call
units (raw NBD via dusted_bat.segment_bout), then generates the full 5-condition counterfactual
set per candidate bout (natural / order_shuffled / onset_offset_swap / caller_swapped /
amplitude_matched) using experiments.playback_design.audio_splicer. Writes WAVs + a manifest
to outputs/playback_stimuli/.

These are exemplar-based DSP transforms (slice + reorder + concatenate), NOT neural synthesis.
Deterministic, firewall-clean. For the wet lab: the H0-CMC falsifiable hypotheses are that
receiver response differs between natural and each counterfactual (T2–T4 evidence tiers).

Run:  python scripts/playback_splice.py --n-bouts 20 --out outputs/playback_stimuli

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import dusted_bat as DB  # noqa: E402

from bioacoustic_pipeline.experiments.playback_design.audio_splicer import (  # noqa: E402
    extract_unit_audio,
    generate_all_counterfactuals,
)
from bioacoustic_pipeline.utils.audio import load_audio, write_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="DSP Audio Splicer — H0-CMC counterfactual stimuli")
    ap.add_argument("--n-bouts", type=int, default=20, help="candidate bouts to process")
    ap.add_argument("--min-units", type=int, default=3, help="min sub-call units to splice")
    ap.add_argument("--out", type=str, default="outputs/playback_stimuli")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    ann = DB.parse_annotations(DB.DATA / "annotations.csv")
    rng = random.Random(args.seed)
    files = sorted(ann.keys())
    rng.shuffle(files)

    # Pass 1: load bouts, segment, collect units + build a donor pool (for caller_swapped).
    bouts = []  # [(fn, emitter, context, units_list)]
    donor_pool = []  # units from all bouts (for caller_swapped — different emitter)
    for fn in files:
        if len(bouts) >= args.n_bouts * 3:  # over-sample to get enough with ≥min_units
            break
        try:
            audio, sr = load_audio(str(DB.DATA / "audio" / fn), target_sr=None)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.2 * sr):
            continue
        segments = DB.segment_bout(audio, sr)
        unit_bounds = [(s / 1000.0, e / 1000.0) for s, e in segments]
        units = extract_unit_audio(0.0, len(audio) / sr, unit_bounds, audio, sr)
        if len(units) >= 2:
            emitter, _add, ctx = ann[fn]
            bouts.append((fn, emitter, ctx, units, sr))
            donor_pool.extend(units[:3])  # contribute up to 3 units to the pool

    # Filter to bouts with enough units for meaningful splicing.
    candidates = [b for b in bouts if len(b[3]) >= args.min_units]
    rng.shuffle(candidates)
    candidates = candidates[: args.n_bouts]
    print(f"{len(bouts)} bouts loaded ({len(candidates)} with ≥{args.min_units} units)", flush=True)
    if not candidates:
        print("No candidates — try lowering --min-units", flush=True)
        return

    # Pass 2: generate counterfactuals + write WAVs + build manifest.
    manifest = []
    for i, (fn, emitter, ctx, units, sr) in enumerate(candidates):
        call_id = f"call{i:03d}"
        results = generate_all_counterfactuals(units, sr, donor_pool=donor_pool, seed=args.seed + i)
        for r in results:
            wav_name = f"{call_id}_{r.condition}.wav"
            write_audio(str(wav_dir / wav_name), r.audio, r.sr)
            manifest.append(
                {
                    "call_id": call_id,
                    "condition": r.condition,
                    "wav": f"wavs/{wav_name}",
                    "n_units": r.n_units,
                    "duration_s": round(len(r.audio) / r.sr, 3),
                    "emitter": emitter,
                    "context": ctx,
                    "provenance": r.provenance,
                    "source_file": fn,
                }
            )
        print(
            f"  {call_id}: {len(results)} conditions written "
            f"({len(units)} units, emitter={emitter}, ctx={ctx})",
            flush=True,
        )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(
            {
                "n_calls": len(candidates),
                "n_stimuli": len(manifest),
                "conditions": sorted({m["condition"] for m in manifest}),
                "stimuli": manifest,
            },
            fh,
            indent=2,
        )

    print(f"\n{'=' * 80}\nDSP AUDIO SPLICER — H0-CMC counterfactual stimuli\n{'=' * 80}")
    print(
        f"  calls: {len(candidates)} | stimuli: {len(manifest)} | "
        f"conditions: {sorted({m['condition'] for m in manifest})}"
    )
    print(f"  output: {out_dir}/")
    print("\n  H0-CMC hypotheses (for the wet lab):")
    print("    natural vs order_shuffled     → does sequence syntax matter? (T2)")
    print("    natural vs onset_offset_swap   → do positional roles matter? (T2)")
    print("    natural vs caller_swapped      → does caller identity matter? (T2)")
    print("    natural vs amplitude_matched   → control: rules out loudness confound")
    print("\n  NOTE: exemplar-based DSP (slice+reorder+concatenate), NOT neural synthesis.")
    print("  Firewall-clean (raw audio, no model weights). semantic_validation: none.")
    print(f"\nsaved {manifest_path}")
    print("\nPLAYBACK SPLICE DONE")


if __name__ == "__main__":
    main()
