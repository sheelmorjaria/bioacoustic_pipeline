#!/usr/bin/env python3
"""RQA on RookID — graded recurrent dynamics of corvid field calls.

Rook was deferred from the §39 cross-corpus RQA table (its ~2 h 4-channel field wavs can't be
loaded whole). This script runs RQA per-call instead: parse the per-recording TSV
(Source/Event/Start/End), seek just each call's audio window (mono mixdown), build the 44-D
frame-fingerprint trajectory, and compute RQA. This (a) completes the §39 cross-corpus RQA
profile with rook, and (b) stratifies DET/LAM by Source — do individuals differ in graded
dynamics? (the §41/§42 individual-specificity theme, read at the continuous-dynamics level).

Metrics (recap): DET = graded recurrent units (diagonal motifs); L = motif length; LAM =
laminar/segment dwell. REC held ~constant → DET/LAM length-comparable. §34 showed bat LAM is
context-dependent within bouts; here we ask if rook DET/LAM is *individual*-dependent across
calls. Firewall-clean: all-species acoustic fingerprint. Ceiling-bound: not meaning.

Reuses rookid_individual_id.{parse_tsvs, seek_audio} + structural_alternatives.rqa_for_trajectory.

Run:  python scripts/rook_rqa.py --max-calls 400

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

WIN_MS = 80.0
HOP_MS = 40.0
MAX_FRAMES = 120


def frame_trajectory(audio, sr, fp_fn):
    win = int(WIN_MS / 1000 * sr)
    hop = int(HOP_MS / 1000 * sr)
    frames, i = [], 0
    while i + win <= len(audio) and len(frames) < MAX_FRAMES:
        v = np.asarray(fp_fn(audio[i : i + win], sr), dtype=np.float32)
        if np.any(v):
            frames.append(v)
        i += hop
    return np.asarray(frames, dtype=np.float32) if frames else np.zeros((0, 1), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--top-k-sources", type=int, default=6)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import rookid_individual_id as rk  # noqa: E402
    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
    from bioacoustic_pipeline.diagnostics.structural_alternatives import rqa_for_trajectory  # noqa: E402
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: E402

    fw = default_firewall()
    fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)  # all-species → legal
    print("firewall: acoustic_fingerprint (all-species) legal for rook")

    calls = rk.parse_tsvs()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(calls)
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species="corvid").combined, dtype=np.float32
    )

    rows = []
    for wav_stem, source, event, s, e in calls:
        if len(rows) >= args.max_calls:
            break
        dur = e - s
        if dur < 0.15 or dur > 3.0:  # skip clips / over-long (multi-call) windows
            continue
        wav = rk.DATA / f"{wav_stem}.wav"
        if not wav.exists():
            continue
        try:
            audio = rk.seek_audio(wav, s, e)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.1 * rk.SR):
            continue
        traj = frame_trajectory(audio, rk.SR, fp_fn)
        if traj.shape[0] < args.min_frames:
            continue
        r = rqa_for_trajectory(traj)
        rows.append(
            {
                "source": source,
                "event": event,
                "dur": round(dur, 2),
                "DET": r["determinism"],
                "LAM": r["laminarity"],
                "L": r["mean_diag_length"],
                "TT": r["trapping_time"],
                "REC": r["recurrence_rate"],
                "n_frames": int(traj.shape[0]),
            }
        )
    if not rows:
        print("ERROR: no calls produced enough frames for RQA (calls too short?)")
        return
    print(
        f"rook: RQA on {len(rows)} calls ({len(set(r['source'] for r in rows))} sources); "
        f"frames/call med={int(np.median([r['n_frames'] for r in rows]))} "
        f"min={min(r['n_frames'] for r in rows)} max={max(r['n_frames'] for r in rows)}"
    )

    def med(k):
        return float(np.median([r[k] for r in rows]))

    # ---- stratify by Source: do individuals differ in graded dynamics? ----
    from scipy.stats import kruskal  # noqa: PLC0415

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    top = sorted(by_src, key=lambda s: -len(by_src[s]))[: args.top_k_sources]
    det_h = det_p = lam_h = lam_p = float("nan")
    if len(top) >= 2:
        det_groups = [[r["DET"] for r in by_src[s]] for s in top if len(by_src[s]) >= 5]
        lam_groups = [[r["LAM"] for r in by_src[s]] for s in top if len(by_src[s]) >= 5]
        if len(det_groups) >= 2:
            det_h, det_p = kruskal(*det_groups)
            lam_h, lam_p = kruskal(*lam_groups)

    print("\n" + "=" * 94)
    print("ROOK RQA — graded recurrent dynamics of corvid field calls")
    print("=" * 94)
    print(
        f"  corpus-level: DET={med('DET'):.3f}  LAM={med('LAM'):.3f}  motif_L={med('L'):.2f}  "
        f"trap_TT={med('TT'):.2f}  REC={med('REC'):.3f}  (n={len(rows)} calls)"
    )
    print(f"\n  Per-Source (top {len(top)} by call count):")
    print(f"    {'source':>10} {'n':>4} {'DET':>7} {'LAM':>7} {'motif_L':>8} {'dur_s':>6}")
    for s in top:
        rs = by_src[s]
        print(
            f"    {s:>10} {len(rs):>4} {np.median([r['DET'] for r in rs]):>7.3f} "
            f"{np.median([r['LAM'] for r in rs]):>7.3f} "
            f"{np.median([r['L'] for r in rs]):>8.2f} {np.median([r['dur'] for r in rs]):>6.2f}"
        )
    print(f"\n  Kruskal-Wallis across sources: DET p={det_p:.4g} | LAM p={lam_p:.4g}")
    ind = det_p < 0.05 or lam_p < 0.05
    print(
        f"  => graded dynamics are {'INDIVIDUAL-DEPENDENT' if ind else 'not clearly source-dependent'} "
        f"across rooks."
    )

    print("\n  Read:")
    print("    Rook joins the §39 cross-corpus RQA table (corvid field calls). DET/LAM quantify")
    print("    graded recurrent structure; the per-Source KW tests whether individuals differ in")
    print("    dynamics (the §41/§42 individual-specificity theme at the continuous level).")
    print("  CAVEAT: rook calls are short (~1 s → ~25 frames); DET is resolution-limited like the")
    print("  §39 bat/zebra-finch short-recording caveat — read LAM and the per-source comparison,")
    print("  not the absolute DET. Ceiling-bound: not meaning.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "n_calls": len(rows),
        "n_sources": len(set(r["source"] for r in rows)),
        "corpus": {
            "DET": med("DET"),
            "LAM": med("LAM"),
            "L": med("L"),
            "TT": med("TT"),
            "REC": med("REC"),
        },
        "per_source": {
            s: {
                "n": len(by_src[s]),
                "DET": float(np.median([r["DET"] for r in by_src[s]])),
                "LAM": float(np.median([r["LAM"] for r in by_src[s]])),
                "L": float(np.median([r["L"] for r in by_src[s]])),
            }
            for s in top
        },
        "kruskal_source": {"DET_p": det_p, "LAM_p": lam_p, "individual_dependent": bool(ind)},
        "frames_per_call": {
            "med": int(np.median([r["n_frames"] for r in rows])),
            "min": min(r["n_frames"] for r in rows),
            "max": max(r["n_frames"] for r in rows),
        },
        "framing": "Graded recurrent dynamics of rook calls + per-individual KW. NOT meaning.",
    }
    with open(out / "rook_rqa.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved outputs/rook_rqa.json")
    print("ROOK RQA DONE")


if __name__ == "__main__":
    main()
