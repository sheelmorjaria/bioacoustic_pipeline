#!/usr/bin/env python3
"""Pipeline entry point: tokens-file OR raw audio -> schema-valid verdict.

Two modes:

* **tokens-file** (legacy): a hand-written JSON of pre-tokenized streams feeds the
  verdict engine directly.
    python scripts/run.py --species bat --tokens-file tokens.json [--with-null]

* **audio** (end-to-end): real audio files are segmented, firewall-gated
  tokenized, diagnosed, and verdicted in one pass, with an execution trace.
    python scripts/run.py --species human --audio-dir /data/libri [--with-null]
    python scripts/run.py --species human --audio a.wav --audio b.wav

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.orchestration import run_pipeline, run_report  # noqa: E402


def _gather_audio(args) -> list[str]:
    paths: list[str] = list(args.audio or [])
    if args.audio_dir:
        root = Path(args.audio_dir)
        for ext in ("*.wav", "*.flac", "*.mp3"):
            paths.extend(str(p) for p in sorted(root.rglob(ext)))
    return paths


def main():
    ap = argparse.ArgumentParser(description="Tokens or audio -> verdict report")
    ap.add_argument("--species", default="bat")
    ap.add_argument(
        "--tokens-file",
        help="JSON: {tokens:[...], boundary_types:[...], sequences:[[...]]}",
    )
    ap.add_argument("--audio", action="append", help="audio file path (repeatable)")
    ap.add_argument("--audio-dir", help="directory of audio (rglob wav/flac/mp3)")
    ap.add_argument(
        "--with-null", action="store_true", help="run permutation null on aliases"
    )
    args = ap.parse_args()

    if args.tokens_file:
        data = json.loads(Path(args.tokens_file).read_text())
        report = run_report(
            args.species,
            data["tokens"],
            data.get("boundary_types", []),
            sequences=data.get("sequences"),
            with_null=args.with_null,
        )
        print(json.dumps(report, indent=2))
        return

    audio_paths = _gather_audio(args)
    if not audio_paths:
        ap.error("provide --tokens-file, or --audio/--audio-dir")

    report, trace = run_pipeline(
        audio_paths, args.species, with_null=args.with_null
    )
    print(json.dumps({"report": report, "trace": trace}, indent=2))


if __name__ == "__main__":
    main()
