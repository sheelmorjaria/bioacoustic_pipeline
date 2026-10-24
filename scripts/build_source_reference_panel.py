#!/usr/bin/env python3
"""Build the source-reference review panel (independent human validation).

Two separately reported parts, per the consolidation plan:

  REPRESENTATIVE sample — selected WITHOUT looking at any model score:
      deterministic hash sample over the eligible bat corpus (seed fixed,
      rule written down before drawing).

  CHALLENGE sample — deterministic difficulty criteria computed from
      recordings (never from downstream model outputs): mask disagreement
      (activity says vocal, fingerprint gate says invalid), multi-unit
      recordings, high gate-rejection fraction, and near-edge activity.

The reviewer (a human — this package only PREPARES the review) inspects
COMPLETE source intervals without detector boundaries, including regions
with no proposals (missed activity must be visible). Deliverables per file:
original sample coordinates, annotation provenance, both masks, and review
columns for the three validation questions:

  Q1  does the activity mask select the intended material?
  Q2  do valid feature trajectories cover that material adequately?
  Q3  (paired subsample) does a proposed match reflect source resemblance?

The four rejected recurrence pairs ship as NEGATIVE regression fixtures —
known failure modes, never positives.

Usage:
    python scripts/build_source_reference_panel.py [--n-representative 40]
                                                   [--n-challenge 30] [--jobs 16]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED_RULE = (
    "representative sample = files sorted by sha1(file_name) with seed-salted "
    "hash, first N taken — no model output, score, or downstream statistic "
    "was consulted at any point in the draw"
)


def _stable_rank(name: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{name}".encode()).hexdigest()


def _mask_profile(path: Path) -> dict | None:
    """Both masks on the SAME 40 ms / 10 ms grid (release-contract rule)."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.segmentation.silence_units import (
        HysteresisConfig,
        extract_activity_units,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    audio, sr = load_audio(str(path), target_sr=250000)
    win, hop = int(sr * 0.04), int(sr * 0.01)
    rows, i = [], 0
    while i + win <= len(audio) and len(rows) < 300:
        q = afp.extract(audio[i : i + win], sr, species="bat").quality
        rows.append(bool(q.get("low_energy") or q.get("too_short")))
        i += hop
    if len(rows) < 20:
        return None
    gated = np.asarray(rows)  # True = fingerprint-invalid
    # activity mask: peak-relative vocal criterion (the sixpair rule)
    f = int(sr * 0.01)
    x = audio[: (len(audio) // f) * f].reshape(-1, f).astype(np.float64)
    rms10 = np.sqrt((x**2).mean(axis=1))
    peak = float(np.abs(audio).max()) or 1.0
    vocal10 = rms10 >= 1e-3 * peak
    # resample the 10 ms activity decision onto the 40 ms frame grid (mean
    # of the 4 sub-frames > 0.5 = vocal) — recorded, never merged
    n = min(len(vocal10) // 4, len(gated))
    act = vocal10[: 4 * n].reshape(n, 4).mean(axis=1) > 0.5
    gated_n = gated[:n]
    res = extract_activity_units(
        audio, sr=sr, config=HysteresisConfig(frame_ms=10.0), ref_amplitude=peak
    )
    return {
        "n_frames": int(n),
        "gate_invalid_fraction": round(float(gated_n.mean()), 4),
        "activity_fraction": round(float(act.mean()), 4),
        # vocal-by-activity AND gate-invalid — the failure signature
        "disagreement_fraction": round(float((act & gated_n).mean()), 4),
        "n_activity_units": len(res.spans),
        "duration_s": round(len(audio) / sr, 3),
    }


def build_panel(n_rep: int, n_challenge: int, jobs: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    by_file = {a.file_name: a for a in annos}

    # representative: hash-ranked over the FULL corpus, no scores consulted
    ranked = sorted(by_file, key=lambda n: _stable_rank(n, "panel-representative-v1"))
    representative = ranked[:n_rep]

    # challenge candidate pool: deterministic difficulty signals only
    # (mask profiles need audio, so compute on a bounded hash-prefix pool)
    pool = sorted(by_file, key=lambda n: _stable_rank(n, "panel-challenge-v1"))[:400]
    audio_dir = ROOT / "data/egyptian_fruit_bats/audio"
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        profiles = list(ex.map(_mask_profile, [(audio_dir / f) for f in pool], chunksize=4))
    cand = []
    for f, p in zip(pool, profiles):
        if p is None:
            continue
        if p["n_frames"] < 20:
            continue
        cand.append((f, p))
    # challenge criteria (deterministic ordering, no model scores):
    # 1. mask disagreement signature (activity vocal, gate invalid)
    # 2. multi-unit recordings
    # 3. high gate rejection with sustained activity
    cand.sort(
        key=lambda fp: (
            -fp[1]["disagreement_fraction"],
            -min(fp[1]["n_activity_units"], 6),
            -fp[1]["gate_invalid_fraction"],
        )
    )
    challenge = [f for f, _ in cand[:n_challenge]]

    panel = {
        "version": "v1 (2026-10-18)",
        "selection_rules": {
            "representative": SEED_RULE,
            "challenge": "deterministic difficulty ordering on mask-profile "
            "signals only (disagreement fraction, unit count, gate rejection); "
            "no downstream model outputs consulted",
        },
        "review_protocol": {
            "first_pass": "inspect the COMPLETE recording interval without "
            "detector boundaries; include regions with no proposals (missed "
            "activity must be visible)",
            "questions": {
                "Q1_activity_mask": "does the activity mask select the intended "
                "material? (coverage, non-vocal inclusion, uncertainty)",
                "Q2_feature_support": "do valid feature trajectories cover that "
                "material adequately? (weak tails, boundaries)",
                "Q3_pair_resemblance": "blinded pair review with explicit "
                "correspondence and ambiguity labels (paired subsample only)",
            },
            "status_vocabulary": [
                "valid measurement",
                "exploratory association",
                "rejected match",
                "source-supported candidate",
                "receiver-tested result",
            ],
            "negative_fixtures": "the four rejected recurrence pairs ship as "
            "known-failure regression fixtures (recurrence_candidate_"
            "adjudication.json) — never as positives",
        },
        "representative": [],
        "challenge": [],
        "files": [],
    }
    meta = {}
    prof_by_file = dict(cand)
    for part, names in (("representative", representative), ("challenge", challenge)):
        for nm in names:
            a = by_file[nm]
            p = prof_by_file.get(nm)
            entry = {
                "file": nm,
                "part": part,
                "caller": a.emitter,
                "addressee": a.addressee,
                "context": a.context,
                "audio": f"data/egyptian_fruit_bats/audio/{nm}",
                "mask_profile": p,
                "review_columns": [
                    "vocal_start_s",
                    "vocal_end_s",
                    "q1_activity_agrees",
                    "q1_notes",
                    "q2_support_adequate",
                    "q2_notes",
                    "overall_status",
                ],
            }
            panel[part].append(nm)
            meta[nm] = entry
    panel["files"] = list(meta.values())

    out_dir = ROOT / "outputs" / "source_reference_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(panel, indent=1))

    # review CSV skeleton (human fills; machine never prefills verdicts)
    import csv

    with open(out_dir / "review.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "file",
                "part",
                "vocal_start_s",
                "vocal_end_s",
                "q1_activity_agrees",
                "q1_notes",
                "q2_support_adequate",
                "q2_notes",
                "overall_status",
            ]
        )
        for nm in panel["representative"] + panel["challenge"]:
            w.writerow([nm, meta[nm]["part"], "", "", "", "", "", "", ""])
    print(
        f"panel: {len(representative)} representative + {len(challenge)} "
        f"challenge files -> {out_dir}",
        flush=True,
    )
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-representative", type=int, default=40)
    ap.add_argument("--n-challenge", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    build_panel(args.n_representative, args.n_challenge, args.jobs)


if __name__ == "__main__":
    main()
