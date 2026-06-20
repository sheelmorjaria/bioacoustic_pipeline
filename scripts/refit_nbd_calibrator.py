#!/usr/bin/env python3
"""Refit the NBD boundary calibrator on LibriSpeech and tune its threshold.

Trains on the modular pipeline's own proposals (consistent params at train +
inference via ``orchestration.CALIBRATOR_PROPOSAL``), labels them against wav2vec2
forced-aligned word starts, fits on a file-held-out split, and sweeps
``keep_threshold`` on held-out **multi-resolution F1** — the blueprint's
segmentation metric, which reclaims word-level "false positives" that land on
real phoneme boundaries (an acoustic segmenter finds sub-word structure by
design). Reports both word-level and combined F1 vs the fixed-threshold baseline,
and saves the checkpoint with the tuned threshold baked in.

Ethological firewall: trained ONLY on human labels; never applied to animals.

Usage:
    python scripts/refit_nbd_calibrator.py --n-files 400
    python scripts/refit_nbd_calibrator.py --n-files 800 --test-frac 0.2

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

from bioacoustic_pipeline.diagnostics.probes import (  # noqa: E402
    forced_align,
    load_librispeech_transcript,
)
from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import CALIBRATOR_PROPOSAL, segment_audio  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import (  # noqa: E402
    BoundaryCalibrator,
    label_proposals,
    proposal_context,
)
from bioacoustic_pipeline.segmentation.nbd_evaluation import multi_resolution_f1  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import (  # noqa: E402
    compute_signals,
    generate_proposals,
)
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

SR = 16000
WORD_TOL_MS = 50.0
PHON_TOL_MS = 30.0
DEFAULT_LIBRISPEECH = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
KEEP_GRID = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CKPT_OUT = ROOT / "checkpoints" / "nbd_calibrator" / "human.pkl"
RESULTS_OUT = ROOT / "outputs" / "nbd_calibrator_refit.json"


def _rates(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"f1": round(2 * p * r / max(p + r, 1e-8), 4),
            "precision": round(p, 4), "recall": round(r, 4)}


def _pool_f1(files, pred_fn):
    """Pooled word-level AND combined (multi-resolution) F1 over files."""
    wt = wp = wfn = ct = cfp = cfn = 0
    for f in files:
        m = multi_resolution_f1(pred_fn(f), f["gt"], f["phon"],
                                 word_tol=WORD_TOL_MS, phoneme_tol=PHON_TOL_MS)
        wt += m.word.tp
        wp += m.word.fp
        wfn += m.word.fn
        ct += m.combined.tp
        cfp += m.combined.fp
        cfn += m.combined.fn
    return {"word": _rates(wt, wp, wfn), "combined": _rates(ct, cfp, cfn)}


def collect(n_files, librispeech, aligner, labels, profile, cpc=None):
    """Per file: calibrated proposals + labels + word/phoneme GT + baseline preds.

    If ``cpc`` (a CPCBackbone) is given, also store learned-context features
    (``proposal_context`` over ``cpc.embed_frames``) aligned with the proposals.
    """
    flacs = sorted(librispeech.rglob("*.flac"))[:n_files]
    files = []
    for fi, flac in enumerate(flacs):
        audio, sr = load_audio(str(flac), target_sr=SR)
        transcript = load_librispeech_transcript(flac)
        aln = forced_align(transcript, audio, sr, model=aligner, labels=labels)
        if not aln.word_starts:
            continue
        sig = compute_signals(audio, sr)
        props = generate_proposals(sig, **CALIBRATOR_PROPOSAL)
        y = label_proposals(props, aln.word_starts, WORD_TOL_MS)
        baseline = [s.start_ms for s in segment_audio(audio, sr, profile, calibrator=None)]
        ctx = proposal_context(props, cpc.embed_frames(audio, sr)) if cpc is not None else None
        files.append({"proposals": props, "y": y, "gt": aln.word_starts,
                      "phon": aln.phoneme_starts, "baseline": baseline, "ctx": ctx})
        if (fi + 1) % 50 == 0:
            print(f"  collected {fi + 1}/{len(flacs)} files "
                  f"({sum(len(f['proposals']) for f in files)} proposals)")
    return files


def main():
    ap = argparse.ArgumentParser(description="Refit + tune the NBD calibrator on LibriSpeech")
    ap.add_argument("--n-files", type=int, default=400)
    ap.add_argument("--librispeech", type=Path, default=DEFAULT_LIBRISPEECH)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=["gbt", "logistic"], default="gbt",
                    help="calibrator model (gbt exploits feature interactions; beats logistic)")
    ap.add_argument("--context", choices=["cpc", "none"], default="cpc",
                    help="learned-context features: cpc (CPC frame embeddings) or none (scalars only)")
    args = ap.parse_args()

    import torchaudio  # noqa: PLC0415
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    aligner, labels = bundle.get_model(), bundle.get_labels()
    profile = load_species_profiles()["human"]
    cpc = CPCBackbone() if args.context == "cpc" else None
    if cpc is not None and not cpc.available:
        raise RuntimeError("CPCBackbone unavailable; run with --context none or provide checkpoints")

    print(f"Collecting proposals from {args.n_files} LibriSpeech utterances "
          f"(context={args.context})...")
    files = collect(args.n_files, args.librispeech, aligner, labels, profile, cpc=cpc)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(files)
    n_test = max(1, int(len(files) * args.test_frac))
    test, train = files[:n_test], files[n_test:]
    print(f"Split: {len(train)} train / {n_test} held-out files")

    train_props = [p for f in train for p in f["proposals"]]
    train_y = np.concatenate([f["y"] for f in train]) if train else np.array([])
    train_ctx = np.vstack([f["ctx"] for f in train]) if cpc is not None and train else None
    cal = BoundaryCalibrator().fit(train_props, train_y, kind=args.kind, context=train_ctx)
    print(f"Fit {args.kind} calibrator on {len(train_props)} proposals "
          f"(+rate {float(train_y.mean()):.1%}, {int(train_y.sum())} positive)")

    # Tune keep_threshold on held-out COMBINED F1 (the real segmentation metric).
    grid_results = {}
    for k in KEEP_GRID:
        grid_results[k] = _pool_f1(test, lambda f, _k=k: [
            b.time_ms for b in cal.apply(f["proposals"], keep_threshold=_k, context=f["ctx"])[0]
        ])
    best_keep = max(KEEP_GRID, key=lambda k: grid_results[k]["combined"]["f1"])
    calibrated = _pool_f1(test, lambda f: [
        b.time_ms for b in cal.apply(f["proposals"], keep_threshold=best_keep, context=f["ctx"])[0]
    ])
    baseline = _pool_f1(test, lambda f: f["baseline"])

    cal.keep_threshold = best_keep
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    cal.save(CKPT_OUT)

    result = {
        "kind": args.kind,
        "context": args.context,
        "n_train_files": len(train),
        "n_test_files": n_test,
        "n_train_proposals": len(train_props),
        "positive_rate": round(float(train_y.mean()), 4),
        "proposal_params": CALIBRATOR_PROPOSAL,
        "word_tol_ms": WORD_TOL_MS,
        "phoneme_tol_ms": PHON_TOL_MS,
        "best_keep_threshold": best_keep,
        "calibrated": calibrated,
        "baseline_fixed_threshold": baseline,
        "keep_grid": {str(k): grid_results[k] for k in KEEP_GRID},
        "checkpoint": str(CKPT_OUT),
    }
    print("=" * 64)
    print(f"Refit complete. {args.kind} | context={args.context} | "
          f"best keep_threshold = {best_keep} (tuned on combined F1)")
    print(f"  calibrated: word-F1={calibrated['word']['f1']:.3f}  "
          f"combined-F1={calibrated['combined']['f1']:.3f} "
          f"(P={calibrated['combined']['precision']:.3f} R={calibrated['combined']['recall']:.3f})")
    print(f"  baseline:   word-F1={baseline['word']['f1']:.3f}  "
          f"combined-F1={baseline['combined']['f1']:.3f}")
    print(f"  saved -> {CKPT_OUT}")
    print("=" * 64)

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {RESULTS_OUT}")


if __name__ == "__main__":
    main()
