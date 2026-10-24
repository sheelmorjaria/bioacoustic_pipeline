#!/usr/bin/env python3
"""Prepare the blinded candidate-vs-control source-review package.

REVIEW DESIGN (per review): a fixed, comparable sample of natural
CANDIDATE hits and matched RANDOM-FRAGMENT-CONTROL hits, presented
blind — the reviewer must not see which condition a pair came from,
nor its distance or recurrence count, until the source judgements are
locked. Clips include SURROUNDING source audio, not just the isolated
interval, and the sample is drawn deterministically over ALL hits (never
the most attractive ones).

CONDITIONS:
  CANDIDATE: the caller-215 individual study's dominant fragment
  (exemplar 37972-family head 37942.wav, L=8) — its frozen exemplar
  interval paired with a deterministic sample of its validation hits.
  CONTROL: random-fragment queries regenerated under the study's
  declared control rule (same seed, same lengths, same frozen radius),
  each paired with one of its accepted validation hits.

QUESTIONS (review columns; uncertainty preserved, never forced):
  q1 is the source interval valid acoustic material? (weak boundaries /
     overlapping sources recorded as uncertainty)
  q2 is there a recognisable SHARED ACOUSTIC CORE between the two clips?
     (detailed correspondence vs same-band/generic-onset)
  q3 where does that correspondence begin and end? (reviewer's own
     judgement; NOT adjusted to agree with DTW)
  q4 does the fragment occur within DIFFERENT surrounding material in
     the two clips? (supports reuse beyond complete-call copies; says
     nothing about biological call types)

BLINDING: pairs are shuffled with a fixed seed; the condition key lives
ONLY in outputs/blinded_review/blinded_review_key.json (sealed — open
after judgements are locked). The review CSV carries no distances, no
recurrence counts, no condition labels.

OUTCOME TABLE (to be determined by the review, not licensed in advance):
  both sets coherent / candidate-only coherent / neither / ambiguous —
  see fragment_sensitivity_benchmark_report.md.

Human review supports SOURCE-LEVEL ACOUSTIC CORRESPONDENCE only: not
receiver perception, not function, not word status. The candidate stays
non-promoted under the frozen rule regardless of the review outcome.

Usage:
    python scripts/prepare_blinded_review.py [--n-per-condition 20]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 0
N_PER = 20
CONTEXT_S = 0.5  # surrounding source audio on each side of an interval
SR = 250000
WIN_MS, HOP_MS = 40.0, 10.0
LENGTHS = (8, 16, 24)
FROZEN_RADIUS = 1.0796


def _frame_to_samples(frame_start: int, frame_end: int) -> tuple[int, int]:
    """Analysis grid -> source samples (frame i covers [i*hop, i*hop+win))."""
    hop = int(SR * HOP_MS / 1000)
    win = int(SR * WIN_MS / 1000)
    return frame_start * hop, (frame_end - 1) * hop + win


def _clip(file_name: str, s_ms: float, e_ms: float, out_path: Path):
    import numpy as np
    import soundfile as sf

    from bioacoustic_pipeline.utils.audio import load_audio

    audio, sr = load_audio(str(ROOT / "data/egyptian_fruit_bats/audio" / file_name), target_sr=SR)
    a = max(0, int(s_ms / 1000 * sr))
    b = min(len(audio), int(e_ms / 1000 * sr))
    sf.write(str(out_path), audio[a:b].astype(np.float32), sr)
    return b - a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-per-condition", type=int, default=N_PER)
    args = ap.parse_args()

    import numpy as np

    import individual_repertoire as ir
    import subcall_term_discovery as std

    parents, _ = std.load_parents()
    mine = [p for p in parents if p["caller"] == 215]
    for p in mine:
        p["stage"] = ir._stage(p["block"])
    disc = sorted(
        [p for p in mine if p["stage"] == "discovery"], key=lambda p: (p["file"], p["start"])
    )
    val = sorted(
        [p for p in mine if p["stage"] == "validation"], key=lambda p: (p["file"], p["start"])
    )
    pooled = [p for p in parents if std.part_of_block(p["block"]) == "discovery"]
    allf = np.vstack([p["track"] for p in pooled])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    for p in mine:
        p["z"] = (p["track"] - mu) / sd

    study = json.loads((ROOT / "outputs" / "individual_repertoire.json").read_text())
    fam = max(study["arm_b_fragments"]["families"], key=lambda f: f.get("validation_rate") or 0)
    ex_file, (ex_s, ex_e) = fam["exemplar"], fam["exemplar_interval"]

    rng = np.random.RandomState(SEED)
    pairs = []  # (condition, query_clip_spec, hit_clip_spec)

    # ---- CANDIDATE: deterministic sample over ALL validation hits
    hits = fam["validation_hits"]
    ordered = sorted(hits, key=lambda h: hashlib.sha1(h["file"].encode()).hexdigest())
    for h in ordered[: args.n_per_condition]:
        pairs.append(
            ("candidate", (ex_file, ex_s, ex_e), (h["file"], h["interval"][0], h["interval"][1]))
        )

    # ---- CONTROL: regenerate random-fragment queries under the study's
    # declared rule (same seed order), pair each with one accepted hit
    made = 0
    for i in range(400):
        if made >= args.n_per_condition:
            break
        p = disc[rng.randint(0, len(disc))]
        L = LENGTHS[i % len(LENGTHS)]
        if len(p["z"]) < L + 2:
            continue
        s0 = int(rng.randint(0, len(p["z"]) - L))
        q = p["z"][s0 : s0 + L]
        acc = [
            pv
            for pv in val
            if np.isfinite(std.subseq_dtw(q, pv["z"])[0])
            and std.subseq_dtw(q, pv["z"])[0] <= FROZEN_RADIUS
        ]
        if not acc:
            continue
        # deterministic pick among accepted hits (hash order, first N)
        acc = sorted(acc, key=lambda pv: hashlib.sha1(pv["file"].encode()).hexdigest())
        _, s_hit, e_hit = std.subseq_dtw(q, acc[0]["z"])
        pairs.append(("control", (p["file"], s0, s0 + L), (acc[0]["file"], int(s_hit), int(e_hit))))
        made += 1

    # ---- blind order + clips
    out_dir = ROOT / "outputs" / "blinded_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(pairs)))
    np.random.RandomState(SEED + 1).shuffle(order)
    key = {}
    rows = []
    for k, idx in enumerate(order):
        cond, qa, qb = pairs[idx]
        pid = f"pair_{k:03d}"
        d = out_dir / pid
        d.mkdir(exist_ok=True)
        for tag, (f_, a, b) in (("A", qa), ("B", qb)):
            s_smp, e_smp = _frame_to_samples(a, b)
            s_ms = max(0.0, s_smp / SR * 1000 - CONTEXT_S * 1000)
            e_ms = e_smp / SR * 1000 + CONTEXT_S * 1000
            n = _clip(f_, s_ms, e_ms, d / f"{tag}.wav")
            key.setdefault(pid, {})["duration_ms_" + tag] = round(n / SR * 1000, 1)
        key[pid]["condition"] = cond  # sealed
        rows.append(
            {
                "pair_id": pid,
                "clip_A": f"{pid}/A.wav",
                "clip_B": f"{pid}/B.wav",
                "q1_valid_material": "",
                "q1_notes": "",
                "q2_shared_core": "",
                "q2_core_type": "",
                "q3_core_start_end": "",
                "q4_different_surroundings": "",
                "uncertainty_notes": "",
            }
        )
    (out_dir / "blinded_review_key.json").write_text(
        json.dumps(
            {
                "sealed": "open ONLY after all source judgements are locked",
                "pairs": key,
                "design": "candidate = dominant caller-215 fragment exemplar vs its "
                "validation hits; control = random-fragment queries under the "
                "study's declared rule vs their accepted hits; deterministic "
                "sampling over ALL hits; clips include 0.5 s surrounding source",
            },
            indent=1,
        )
    )
    with open(out_dir / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "README.md", "w") as fh:
        fh.write(__doc__.split("Usage:")[0])
    print(
        f"blinded review package: {len(rows)} pairs "
        f"({sum(1 for c, *_ in pairs if c == 'candidate')} candidate / "
        f"{sum(1 for c, *_ in pairs if c == 'control')} control) -> {out_dir}",
        flush=True,
    )
    print(
        "key sealed in blinded_review_key.json — open only after judgements are locked", flush=True
    )


if __name__ == "__main__":
    main()
