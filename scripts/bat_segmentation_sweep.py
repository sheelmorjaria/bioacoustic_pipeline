#!/usr/bin/env python3
"""Bat NBD segmentation sweep — choose a profile by PLAUSIBILITY, not by results.

The shipped ``bat`` profile carries a call-level holophrasis prior (energy floor
0.20, spectral floor 0.50, debounce 200 ms) that collapses multi-call bouts into
~0-1 segments. That makes every sequence diagnostic a false null (see
diagnostics.sequence_sufficiency). This sweep compares six candidate profiles on
segmentation plausibility — segments/bout, nonzero rate, segment-duration
distribution, boundary-type mix, and sufficiency-gate status — so the profile
choice is defensible and independent of any alias / motif outcome.

Selection rule (never "the one with the most exciting result"):
  * nonzero_bout_rate high (most bouts yield units)
  * median segment duration in the call range (~15-120 ms for R. aegyptiacus)
  * sufficiency gate EVALUABLE (enough units/bout + bigrams)

Usage:
    python scripts/bat_segmentation_sweep.py --n-files 400
    python scripts/bat_segmentation_sweep.py --n-files 400 --audio-dir /path/to/bat

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import segment_audio  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

BAT_AUDIO_DIR = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
OUT = ROOT / "outputs" / "bat_segmentation_sweep.json"

# Call-duration range for R. aegyptiacus social calls (used only for the
# plausibility flag, not as a decision threshold).
_CALL_MS = (15.0, 120.0)


def bat_profiles() -> dict[str, dict]:
    """Six candidate bat NBD profiles + a one-segment-per-file negative control."""
    base = load_species_profiles()["bat"]
    sr = base["sample_rate"]
    hop = base["hop_ms"]

    def p(energy, flux, debounce, desc, drop=float("inf")):
        return {
            "sample_rate": sr, "hop_ms": hop,
            "energy_threshold": energy, "spectral_threshold": flux, "debounce_ms": debounce,
            "spectral_drop_threshold": drop,
            "description": desc,
        }

    return {
        # Shipped profile — the call-level holophrasis prior (the confound).
        "bat_holophrastic": p(0.20, 0.50, 200.0, "current conservative call-level prior"),
        # Candidate: call-granular (robust region from the grid).
        "bat_call_granular": p(0.05, 0.05, 25.0, "individual calls inside bouts (candidate)"),
        # Fire-on-drop variants: same granular base + spectral energy disappearance detection.
        "bat_granular_drop010": p(0.05, 0.05, 25.0, "granular + fire-on-drop 0.10", drop=0.10),
        "bat_granular_drop015": p(0.05, 0.05, 25.0, "granular + fire-on-drop 0.15", drop=0.15),
        "bat_granular_drop025": p(0.05, 0.05, 25.0, "granular + fire-on-drop 0.25", drop=0.25),
        # Finer: sub-call / micro structure.
        "bat_microcall": p(0.02, 0.02, 15.0, "sub-call acoustic structure"),
        # Baselines isolating one signal family.
        "bat_energy_only": p(0.05, 2.0, 25.0, "pause/onset (energy) baseline"),
        "bat_spectral_only": p(2.0, 0.05, 25.0, "FM/spectral-change baseline"),
        # Negative control: exactly one unit per bout (forces NOT_EVALUABLE).
        "bat_metadata_bout": p(0.0, 0.0, 1e9, "one segment per file (neg control)"),
    }


def _gather(audio_dir: Path, n_files: int) -> list[tuple[np.ndarray, int, float]]:
    files = sorted(audio_dir.rglob("*.wav")) + sorted(audio_dir.rglob("*.flac"))
    out = []
    for f in files[:n_files]:
        try:
            a, sr = load_audio(str(f), target_sr=load_species_profiles()["bat"]["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        out.append((a, sr, len(a) / sr * 1000.0))
    return out


def sweep_profile(profile: dict, audios: list) -> dict:
    counts, seg_durs, btypes = [], [], {"hard": 0, "soft": 0, "transitional": 0}
    for a, sr, _dur in audios:
        segs = segment_audio(a, sr, profile)
        counts.append(len(segs))
        for s in segs:
            seg_durs.append(s.end_ms - s.start_ms)
            btypes[s.boundary_type] = btypes.get(s.boundary_type, 0) + 1
    c = np.array(counts, dtype=float)
    sd = np.array(seg_durs) if seg_durs else np.array([0.0])
    total_bt = max(sum(btypes.values()), 1)

    # Sufficiency gate over per-bout unit counts (1 unit == 1 segment here).
    unit_seqs = [[0] * int(n) for n in counts]
    suff = evaluate_sufficiency(unit_seqs)

    med = float(np.median(sd)) if seg_durs else 0.0
    call_like = bool(
        suff.nonzero_bout_rate >= 0.9
        and suff.median_tokens_per_bout >= 3.0
        and _CALL_MS[0] <= med <= _CALL_MS[1]
    )
    return {
        "segments_per_bout_mean": round(float(c.mean()), 2),
        "segments_per_bout_median": float(np.median(c)),
        "nonzero_bout_rate": round(float((c >= 1).mean()), 3),
        "segment_duration_ms": {
            "p10": round(float(np.percentile(sd, 10)), 1),
            "p50": round(med, 1),
            "p90": round(float(np.percentile(sd, 90)), 1),
        },
        "boundary_mix": {k: round(v / total_bt, 3) for k, v in btypes.items()},
        "sufficiency": suff.as_dict(),
        "call_like": call_like,
    }


def main():
    ap = argparse.ArgumentParser(description="Bat NBD segmentation plausibility sweep")
    ap.add_argument("--audio-dir", type=Path, default=BAT_AUDIO_DIR)
    ap.add_argument("--n-files", type=int, default=400)
    args = ap.parse_args()

    audios = _gather(args.audio_dir, args.n_files)
    if not audios:
        raise SystemExit(f"no bat audio under {args.audio_dir}")
    durs = np.array([d for *_a, d in audios])
    print(f"{len(audios)} bouts | duration median={np.median(durs):.0f}ms "
          f"mean={durs.mean():.0f}ms\n")

    profiles = bat_profiles()
    results = {}
    header = (f"{'profile':22s} {'segs/bt':>8} {'nz%':>5} {'med ms':>7} "
              f"{'p90 ms':>7} {'gate':>14} {'call-like':>10}")
    print(header)
    print("-" * len(header))
    for name, prof in profiles.items():
        r = sweep_profile(prof, audios)
        results[name] = r
        print(
            f"{name:22s} "
            f"{r['segments_per_bout_mean']:>8.1f} "
            f"{r['nonzero_bout_rate']*100:>5.0f} "
            f"{r['segment_duration_ms']['p50']:>7.0f} "
            f"{r['segment_duration_ms']['p90']:>7.0f} "
            f"{r['sufficiency']['status']:>14} "
            f"{'yes' if r['call_like'] else 'no':>10}"
        )

    # Recommendation: among call_like + evaluable, the median-granularity profile
    # (bat_call_granular). Selection is by plausibility (documented), NOT by any
    # downstream alias/motif number.
    candidates = [
        n for n, r in results.items()
        if r["call_like"] and r["sufficiency"]["status"] == "evaluable"
    ]
    recommended = "bat_call_granular" if "bat_call_granular" in candidates else (
        candidates[0] if candidates else None)
    print(f"\nRecommended (plausibility-selected): {recommended}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_bouts": len(audios),
        "bout_duration_ms": {
            "median": round(float(np.median(durs)), 1), "mean": round(float(durs.mean()), 1),
        },
        "profiles": {n: {**p, **results[n]} for n, p in profiles.items()},
        "recommended": recommended,
    }, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
