#!/usr/bin/env python3
"""Blind-safe per-pair judgement recorder for the 40-pair source review.

Records the REVIEWER's judgements into outputs/blinded_review/review.csv.
Shows ONLY the anonymous pair id and clip paths — never the condition,
distance, recurrence count or any search label (those live in the sealed
key, opened only after ALL judgements are locked).

Fields (one line per pair; uncertainty is a valid value everywhere):
  q1_valid_material   both intervals interpretable source audio? yes/no/uncertain
  q2_shared_core      a SPECIFIC corresponding pattern (rise + interruption +
                      pulses + ending...), not generic resemblance:
                      yes/no/uncertain
  q2_core_type        what corresponds, in your words
  q3_core_start_end   your own start/end judgement in A and B (s), with
                      uncertainty — never adjusted to agree with any algorithm
  q4_diff_surroundings different surrounding material around the core?
                      yes/no/uncertain
  notes               supporting features and limitations

MODALITY: record "visual correspondence observed" for spectrogram/waveform
inspection; do NOT record auditory agreement unless the audio was also
inspected. Waveform and spectrogram are two views of one source — not two
independent confirmations.

Usage (interactive):
    python scripts/record_judgement.py                # next unjudged pair
    python scripts/record_judgement.py --all          # loop all unjudged
Usage (non-interactive):
    python scripts/record_judgement.py --pair pair_000 --q2 yes \\
        --core-type "rising band, brief attenuation, two broadband pulses" \\
        --q3 "A 0.9-1.6s / B 0.4-1.1s (boundaries +-0.1s)" --q1 yes \\
        --q4 yes --notes "..."

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs" / "blinded_review"
FIELDS = [
    "q1_valid_material",
    "q1_notes",
    "q2_shared_core",
    "q2_core_type",
    "q3_core_start_end",
    "q4_different_surroundings",
    "uncertainty_notes",
]


def _load():
    rows = list(csv.DictReader(open(REVIEW / "review.csv")))
    return rows, list(rows[0].keys()) if rows else FIELDS


def _save(rows, fields):
    with open(REVIEW / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _unjudged(rows):
    return [r for r in rows if not r.get("q2_shared_core")]


def _apply(row, args) -> None:
    if args.q1:
        row["q1_valid_material"] = args.q1
    if args.q2:
        row["q2_shared_core"] = args.q2
    if args.core_type:
        row["q2_core_type"] = args.core_type
    if args.q3:
        row["q3_core_start_end"] = args.q3
    if args.q4:
        row["q4_different_surroundings"] = args.q4
    if args.notes:
        row["uncertainty_notes"] = args.notes


def _prompt(row) -> None:
    print(f"\n=== {row['pair_id']}  (clips: {row['clip_A']} / {row['clip_B']}) ===")
    print("Open the pair's source_view.html and BOTH clip WAVs if you are")
    print("also auditing audio. Judgement fields:")
    row["q1_valid_material"] = input("  q1 valid material (yes/no/uncertain): ").strip()
    row["q2_shared_core"] = input("  q2 SHARED ACOUSTIC CORE (yes/no/uncertain): ").strip()
    row["q2_core_type"] = input("  what corresponds (specific pattern): ").strip()
    row["q3_core_start_end"] = input("  core start/end in A and B (with uncertainty): ").strip()
    row["q4_different_surroundings"] = input(
        "  q4 different surroundings (yes/no/uncertain): "
    ).strip()
    row["uncertainty_notes"] = input("  notes / limitations: ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--q1", type=str, default=None)
    ap.add_argument("--q2", type=str, default=None)
    ap.add_argument("--core-type", type=str, default=None)
    ap.add_argument("--q3", type=str, default=None)
    ap.add_argument("--q4", type=str, default=None)
    ap.add_argument("--notes", type=str, default=None)
    args = ap.parse_args()

    rows, fields = _load()
    if args.pair:
        match = [r for r in rows if r["pair_id"] == args.pair]
        if not match:
            sys.exit(f"unknown pair {args.pair}")
        _apply(match[0], args)
        _save(rows, fields)
        print(f"recorded {args.pair}")
        return
    todo = _unjudged(rows)
    if not todo:
        print("all pairs already judged — lock the review, then open the key")
        return
    if not args.all:
        _prompt(todo[0])
        _save(rows, fields)
        print(f"recorded {todo[0]['pair_id']} ({len(_unjudged(rows))} remaining)")
        return
    for row in todo:
        _prompt(row)
        _save(rows, fields)
    print("all judgements recorded — LOCK the review before opening the key")


if __name__ == "__main__":
    main()
