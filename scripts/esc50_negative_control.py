#!/usr/bin/env python3
"""ESC-50 negative-control report (BEANS-Zero subset).

Runs the pipeline's acoustic stages + the domain gate on ESC-50 and emits a dataset-level
negative-control report whose final verdict is NON_BIOACOUSTIC_COMMUNICATION: the pipeline
detects acoustic structure (NBD segments, fingerprints, tokens) but REFUSES ethological-
communication inference (the ethological axes are not_applicable). ESC-50 biological-source
labels (dog, rooster, crickets, …) are classified BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK,
never forced into NON_BIOLOGICAL_SOUND.

Data sources (``--source``):
  * ``synthetic`` (default, offline): deterministic environmental-like clips tagged with a
    spread of ESC-50 labels — exercises the gate end-to-end without network/checkpoints.
  * ``beans-zero``: ``load_dataset("EarthSpeciesProject/BEANS-Zero", split="test")``,
    filtered to ``dataset == "esc-50"`` (needs the ``datasets`` lib + network).
  * ``local``: ``--audio-dir`` of wavs + ``--labels`` CSV (``path,label`` rows).

Usage:
  python scripts/esc50_negative_control.py --source synthetic --n-files 40 \
      --report outputs/esc50_negative_control_report.json

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics.negative_control_report import (  # noqa: E402
    build_negative_control_report,
)
from bioacoustic_pipeline.hashing import write_canonical_json  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

SR = 16000  # generic acoustic profile sample rate (matches the human NBD profile)
# A representative spread across ESC-50: non-biological + biological-source labels.
ESC50_LABELS = [
    "rain",
    "sea_waves",
    "crackling_fire",
    "wind",
    "engine",
    "train",
    "siren",
    "helicopter",
    "dog",
    "rooster",
    "crickets",
    "chirping_birds",
    "frog",
    "cat",
    "hen",
    "crow",
]


# --------------------------------------------------------------------------- #
# Synthetic ESC-50-like corpus (offline default)
# --------------------------------------------------------------------------- #


def _synth_clip(label: str, rng: np.random.Generator) -> np.ndarray:
    """A deterministic ~2 s clip shaped loosely by its ESC-50 label class.

    Non-biological labels -> broadband/gust noise; biological-source labels ->
    tonal / AM pulse trains. The exact waveform is irrelevant — the DOMAIN GATE is what
    the negative control demonstrates; audio just needs onset structure for NBD.
    """
    dur = 2.0
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float64) / SR
    bio = label in {"dog", "rooster", "crickets", "chirping_birds", "frog", "cat", "hen", "crow"}
    if bio:
        f0 = float(rng.uniform(400, 1400))
        am = 6.0 + float(rng.uniform(0, 18))  # AM rate (trill/call pulsing)
        x = np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * am * t))
    else:
        # gusts of broadband noise with slow amplitude modulation
        amp_env = 0.4 + 0.4 * np.sin(2 * np.pi * float(rng.uniform(0.5, 2.0)) * t)
        x = amp_env * rng.standard_normal(n)
    # a couple of sharp onsets so NBD finds boundaries
    for _ in range(int(rng.integers(2, 5))):
        onset = int(rng.integers(int(0.1 * n), int(0.9 * n)))
        x[onset : onset + int(0.05 * SR)] *= 1.8
    peak = float(np.abs(x).max())
    return (0.6 * x / (peak + 1e-9)).astype(np.float32)


def synthetic_clips(n_files: int, seed: int):
    rng = np.random.default_rng(seed)
    return [
        (
            _synth_clip(ESC50_LABELS[i % len(ESC50_LABELS)], rng),
            SR,
            ESC50_LABELS[i % len(ESC50_LABELS)],
        )
        for i in range(n_files)
    ]


def beans_zero_clips(n_files: int):
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "--source beans-zero needs the `datasets` library (pip install datasets)"
        ) from exc
    ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test")
    esc = ds.filter(lambda r: r.get("dataset") == "esc-50")
    clips = []
    for i, row in enumerate(esc):
        if i >= n_files:
            break
        wav = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = int(row["audio"]["sampling_rate"])
        clips.append((wav, sr, str(row.get("label", ""))))
    return clips


def local_clips(audio_dir: Path, labels_csv: Path):
    clips = []
    with open(labels_csv) as fh:
        for row in csv.DictReader(fh):
            wav, sr = load_audio(str(audio_dir / row["path"]), target_sr=SR)
            clips.append((wav, sr, row["label"]))
    return clips


def main() -> int:
    ap = argparse.ArgumentParser(description="ESC-50 negative-control report")
    ap.add_argument("--source", choices=["synthetic", "beans-zero", "local"], default="synthetic")
    ap.add_argument("--n-files", type=int, default=40)
    ap.add_argument("--audio-dir", help="(local) directory of wav files")
    ap.add_argument("--labels", help="(local) CSV with path,label rows")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--report", default="outputs/esc50_negative_control_report.json")
    args = ap.parse_args()

    if args.source == "synthetic":
        clips = synthetic_clips(args.n_files, args.seed)
    elif args.source == "beans-zero":
        clips = beans_zero_clips(args.n_files)
    else:
        if not args.audio_dir or not args.labels:
            ap.error("--source local requires --audio-dir and --labels")
        clips = local_clips(Path(args.audio_dir), Path(args.labels))

    report = build_negative_control_report("esc-50", clips, k=args.k, seed=args.seed)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(out, report)
    print(f"wrote {out}")
    print(
        f"  sample_count={report['sample_count']}  "
        f"final_verdict={report['final_verdict']['class']}  "
        f"confidence={report['final_verdict']['confidence']}"
    )
    print(
        f"  nonbiological_source={report['summary']['n_nonbiological_source']}  "
        f"biological_source_noncommunication={report['summary']['n_biological_source_noncommunication']}  "
        f"communication_candidates={report['summary']['communication_candidate_files']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
