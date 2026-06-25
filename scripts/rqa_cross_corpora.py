#!/usr/bin/env python3
"""Cross-corpus RQA — graded recurrent units / motifs / compositional substrate.

RQA (Recurrence Quantification Analysis) is the GRADED-dynamics lens: it sees recurrence in
the continuous 44-D frame-fingerprint trajectory that discrete tokenization (DUSTED / PST /
transition-network) misses. §34 showed bat bout LAM is context-dependent; this script runs the
same RQA across corpora and compares the graded-recurrent profile of each repertoire.

What the metrics mean for "graded recurrent units / motifs / compositional structure":
  * DET (determinism) — fraction of recurrent points on DIAGONAL lines. High DET ⇒ the
    trajectory revisits near-identical sequences of states ⇒ **graded recurrent units /
    recurrent motifs** (repeated continuous-acoustic patterns).
  * mean_diag_length (L) — mean diagonal line length ⇒ **motif duration** (how long a
    recurrent graded pattern sustains).
  * LAM (laminarity) — fraction on VERTICAL lines ⇒ laminar / trapping structure ⇒
    segment-like graded dwell (§34's bat context signal).
  * REC held ~constant across recordings (length-invariant) so DET/LAM are comparable.

HONEST SCOPE: RQA characterises the graded recurrent SUBSTRATE (recurrent units, motif-like
repetition, laminar segmentation). It does NOT measure compositionality (order-dependent AB
meaning) directly — that is the discrete-sequence methods' job (§37 bat / §38 zebra finch).
RQA tells us which repertoires *have* the graded recurrent material motifs/composition could
build on. Firewall-clean: all-species acoustic fingerprint (deterministic DSP).

Corpora: bat (graded bouts), Bengalese (song motifs), marmoset + zebra finch (discrete
calls), rook (field calls), LibriSpeech (speech anchor).

Run:  python scripts/rqa_cross_corpora.py --n 80

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# corpus -> (root, species, glob-pattern kind)
CORPORA = {
    "bat": (
        Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio"),
        "egyptian_fruit_bat",
        "wavs",
    ),
    "marmoset": (Path("/home/sheel/birdsong_analysis/data/Vocalizations"), "marmoset", "flacs"),
    "zebra_finch": (
        Path("/mnt/c/Users/sheel/Desktop/data/zebrafinch/adult"),
        "zebra_finch",
        "wavs",
    ),
    "bengalese": (
        Path("/mnt/c/Users/sheel/Desktop/data/bengalifinch"),
        "bengalese_finch",
        "bengalese",
    ),
    "rook": (Path("/mnt/c/Users/sheel/Desktop/data/RookID"), "corvid", "rook"),
    "librispeech": (
        Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360"),
        "human",
        "libri",
    ),
}
WIN_MS = 80.0
HOP_MS = 40.0
MAX_FRAMES = 120
CAP_S = 6.0  # cap loaded audio (MAX_FRAMES already bounds the trajectory to ~4.8 s)


def _bounded_files(root: Path, kind: str, n: int):
    """Yield up to n recording paths without materialising a huge glob (marmoset 871k)."""
    count = 0
    if kind == "wavs":
        for p in sorted(root.glob("*.wav")):
            yield p
            count += 1
            if count >= n * 4:
                return
    elif kind == "flacs":  # marmoset: Vocalizations/<session>/<file>.flac
        for sess in sorted(root.iterdir()):
            if not sess.is_dir():
                continue
            for p in sorted(sess.glob("*.flac")):
                yield p
                count += 1
                if count >= n * 4:
                    return
    elif kind == "bengalese":  # Bird*/Bird*/Wave/*.wav
        for p in sorted(root.rglob("Wave/*.wav")):
            yield p
            count += 1
            if count >= n * 4:
                return
    elif kind == "rook":  # *.wav (4-channel, ~2h)
        for p in sorted(root.glob("*.wav")):
            yield p
            count += 1
            if count >= n * 4:
                return
    elif kind == "libri":  # **/*.flac
        for p in sorted(root.rglob("*.flac")):
            yield p
            count += 1
            if count >= n * 4:
                return


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
    ap.add_argument("--n", type=int, default=80, help="recordings per corpus")
    ap.add_argument("--corpora", default="all", help="comma list or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.structural_alternatives import rqa_for_trajectory  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    fw = default_firewall()
    fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)  # all-species → legal everywhere
    print("firewall: acoustic_fingerprint (all-species, deterministic) legal for all corpora")

    names = list(CORPORA) if args.corpora == "all" else [c.strip() for c in args.corpora.split(",")]
    rng = random.Random(args.seed)

    per_corpus = {}  # corpus -> list of per-recording metric dicts
    for name in names:
        root, species, kind = CORPORA[name]
        fp_fn = lambda seg, sr, sp=species: np.asarray(  # noqa: E731
            afp.extract(seg, sr, species=sp).combined, dtype=np.float32
        )
        files = list(_bounded_files(root, kind, args.n))
        rng.shuffle(files)
        rows = []
        for p in files:
            if len(rows) >= args.n:
                break
            try:
                audio, sr = load_audio(str(p), target_sr=None)
            except Exception:  # noqa: BLE001
                continue
            if audio is None or len(audio) < int(0.1 * sr):
                continue
            if audio.ndim > 1:  # rook 4-channel → mono
                audio = audio.mean(axis=1) if audio.ndim > 1 else audio
            audio = audio[: int(CAP_S * sr)]  # cap for speed (trajectory already bounded)
            traj = frame_trajectory(audio, sr, fp_fn)
            if traj.shape[0] < 8:
                continue
            r = rqa_for_trajectory(traj)
            rows.append(
                {
                    "DET": r["determinism"],
                    "LAM": r["laminarity"],
                    "L": r["mean_diag_length"],
                    "TT": r["trapping_time"],
                    "REC": r["recurrence_rate"],
                    "n_frames": int(traj.shape[0]),
                }
            )
        per_corpus[name] = rows
        print(f"  {name:14}: {len(rows)} recordings", flush=True)

    # ---- aggregate + cross-corpus test ----
    from scipy.stats import kruskal  # noqa: PLC0415

    def med(rows, k):
        return float(np.median([r[k] for r in rows])) if rows else float("nan")

    print("\n" + "=" * 96)
    print("CROSS-CORPUS RQA — graded recurrent units / motifs (44-D frame-fingerprint trajectory)")
    print("=" * 96)
    hdr = f"  {'corpus':14} {'n':>4} {'DET':>7} {'LAM':>7} {'motif_L':>8} {'trap_TT':>8} {'REC':>7} {'frames':>7}"
    print(hdr)
    print("  " + "-" * 92)
    summary = {}
    for name, rows in per_corpus.items():
        if not rows:
            continue
        summary[name] = {
            "n": len(rows),
            "DET_med": med(rows, "DET"),
            "LAM_med": med(rows, "LAM"),
            "L_med": med(rows, "L"),
            "TT_med": med(rows, "TT"),
            "REC_med": med(rows, "REC"),
            "frames_med": med(rows, "n_frames"),
        }
        s = summary[name]
        print(
            f"  {name:14} {s['n']:>4} {s['DET_med']:>7.3f} {s['LAM_med']:>7.3f} "
            f"{s['L_med']:>8.2f} {s['TT_med']:>8.2f} {s['REC_med']:>7.3f} {s['frames_med']:>7.0f}"
        )

    def kw(metric):
        groups = [
            np.array([r[metric] for r in rows]) for rows in per_corpus.values() if len(rows) >= 5
        ]
        if len(groups) < 2:
            return float("nan"), float("nan")
        h, p = kruskal(*groups)
        return float(h), float(p)

    det_h, det_p = kw("DET")
    lam_h, lam_p = kw("LAM")
    print(
        f"\n  Kruskal-Wallis across corpora: DET H={det_h:.2f} p={det_p:.4g} | LAM H={lam_h:.2f} p={lam_p:.4g}"
    )

    # rank corpora by graded-recurrent motif strength (DET * motif_L)
    ranked = sorted(summary.items(), key=lambda kv: -(kv[1]["DET_med"] * kv[1]["L_med"]))
    print("\n  Ranked by graded-recurrent motif strength (DET × motif_L):")
    for name, s in ranked:
        print(
            f"    {name:14} DET={s['DET_med']:.3f}  motif_L={s['L_med']:.2f}  "
            f"score={s['DET_med'] * s['L_med']:.3f}"
        )

    print("\n  Read:")
    print(
        "    DET = graded recurrent units (repeated continuous-acoustic sequences); motif_L = their duration."
    )
    print("    LAM = laminar/segment dwell (graded dynamics; bat §34's context signal).")
    print("    High DET×motif_L repertoires carry strong graded recurrent MOTIFS — the substrate")
    print(
        "    composition could build on. RQA does NOT measure AB compositionality (order) directly"
    )
    print("    — that is §37/§38 (PST / transition-network). Ceiling-bound: not meaning.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "n_per_corpus": args.n,
        "win_ms": WIN_MS,
        "hop_ms": HOP_MS,
        "max_frames": MAX_FRAMES,
        "summary": summary,
        "kruskal_wallis": {"DET_H": det_h, "DET_p": det_p, "LAM_H": lam_h, "LAM_p": lam_p},
        "per_recording": {k: v for k, v in per_corpus.items()},
        "framing": "Graded recurrent substrate (DET/motif_L/LAM). NOT compositionality (order). Ceiling-bound.",
    }
    with open(out / "rqa_cross_corpora.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved outputs/rqa_cross_corpora.json")
    print("CROSS-CORPUS RQA DONE")


if __name__ == "__main__":
    main()
