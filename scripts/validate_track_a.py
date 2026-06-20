#!/usr/bin/env python3
"""Track A: engineering validation on human speech (LibriSpeech).

Proves the machine works on the one species with ground truth, in three sections:

  §1 Segmentation accuracy — calibrated NBD boundaries vs wav2vec2 forced-aligned
     word starts (multi-resolution F1; per-type PRF).
  §2 Semantic ceiling (live pipeline) — unsupervised VQ-VAE token→word
     consistency. Expected near chance: tokens are acoustic-similarity classes,
     not word identity.
  §3 Linear probe — logistic readout of word identity from CPC latents. Expected
     well above chance: the word signal EXISTS, just entangled.

wav2vec2 forced alignment is a GROUND-TRUTH YARDSTICK, not a pipeline model: it
never feeds a verdict, is human-speech-trained, and is used ONLY for human Track
A — never applied to animals.

Usage:
    python scripts/validate_track_a.py --self-test            # synthetic, no network
    python scripts/validate_track_a.py --n-files 200          # real LibriSpeech
    python scripts/validate_track_a.py --n-files 500 --json out.json
    python scripts/validate_track_a.py --features-npz path.npz  # offline probe data

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.segmentation.nbd_evaluation import (  # noqa: E402
    PRF,
    multi_resolution_f1,
    per_type_prf,
)
from bioacoustic_pipeline.diagnostics.probes import (  # noqa: E402
    forced_align,
    linear_probe,
    load_librispeech_transcript,
    load_word_features,
    match_span_to_label,
    token_label_consistency,
)
from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import segment_audio  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import BoundaryCalibrator  # noqa: E402
from bioacoustic_pipeline.tokenization.multiview import AcousticTokenizer  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

SR = 16000
DEFAULT_LIBRISPEECH = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _pool(prfs: list[PRF]) -> dict:
    """Corpus-level PRF from per-file PRFs (sum TP/FP/FN, recompute rates)."""
    tp = sum(p.tp for p in prfs)
    fp = sum(p.fp for p in prfs)
    fn = sum(p.fn for p in prfs)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }


def _tone(freq, dur):
    tt = np.arange(int(SR * dur)) / SR
    return np.sin(2 * np.pi * freq * tt).astype(np.float32)


def _segment_report(segs, word_starts, phon_starts):
    """Per-file multi-resolution F1 (word + combined, reclaiming phoneme GT) + per-type PRF."""
    mres = multi_resolution_f1([s.start_ms for s in segs], word_starts, phon_starts)
    typed = [(s.start_ms, s.boundary_type) for s in segs]
    return mres, per_type_prf(typed, word_starts)


def _collect_segments(segs, tokenizer, cpc, word_spans):
    """Token + matched word + CPC latent per labeled segment."""
    tokens, words, latents = [], [], []
    for seg in segs:
        tok = tokenizer.tokenize(seg.audio, seg.sr)
        if tok == -1:
            continue
        word = match_span_to_label(seg.start_ms, seg.end_ms, word_spans)
        if word:
            tokens.append(tok)
            words.append(word)
            latents.append(cpc.embed(seg.audio, seg.sr))
    latents = np.asarray(latents) if latents else np.empty((0, 256))
    return tokens, words, latents


def _self_test_corpus():
    """Synthetic tone bursts with a known schedule — exercises the wiring offline.

    Returns (audio, [(start_ms, end_ms, label), ...]) where label is the frequency
    bucket (a CPC-detectable signal) so the linear probe has something to find.
    """
    audio_parts, schedule = [], []
    period, t = 0.30, 0.0
    for i in range(160):
        bucket = str(i % 10)  # 10-class "vocab"
        freq = 300.0 + 60.0 * (i % 10)
        dur = period / 2
        start_ms = t * 1000.0
        audio_parts.append(_tone(freq, dur))
        audio_parts.append(np.zeros(int(SR * dur), dtype=np.float32))
        schedule.append((start_ms, start_ms + dur * 1000.0, bucket))
        t += period
    return np.concatenate(audio_parts), schedule


# --------------------------------------------------------------------------- #
# core run
# --------------------------------------------------------------------------- #

def _load_models(species: str):
    profile = load_species_profiles()[species]
    cal_path = ROOT / "checkpoints" / "nbd_calibrator" / f"{species}.pkl"
    calibrator = BoundaryCalibrator.load(cal_path) if cal_path.exists() else None
    return profile, calibrator, AcousticTokenizer(), CPCBackbone()


def run(librispeech: Path, n_files: int, features_npz, self_test: bool, species: str,
        calibrated: bool = False) -> dict:
    profile, calibrator_loaded, tokenizer, cpc = _load_models(species)
    # Track A's proven path is the fixed-threshold NBD (what the prototype's
    # human_word_level_validation used). The learned calibrator is opt-in: at its
    # default keep_threshold=0.5 the committed checkpoint is too conservative on
    # real speech, so it stays off until retuned/refit.
    calibrator = calibrator_loaded if calibrated else None

    word_prfs, combined_prfs = [], []
    per_type_buckets: dict[str, list[PRF]] = defaultdict(list)
    all_tokens, all_words = [], []
    all_latents = np.empty((0, 256))

    if self_test:
        audio, schedule = _self_test_corpus()
        # Force the threshold NBD: the speech-trained calibrator's behavior on pure
        # tones isn't meaningful (it phase-locks to tone offsets, not the known
        # burst onsets), while the fixed-threshold path reliably marks onsets — so
        # the self-test demonstrates the F1/consistency/probe wiring against a
        # known schedule. Real runs also default to the fixed-threshold NBD (the
        # proven Track A path); pass --calibrated to opt into the calibrator.
        segs = segment_audio(audio, SR, profile, calibrator=None)
        mres, per_type = _segment_report(segs, [s for s, _e, _l in schedule], [])
        word_prfs.append(mres.word)
        combined_prfs.append(mres.combined)
        for tname, prf in per_type.items():
            per_type_buckets[tname].append(prf)
        tokens, words, latents = _collect_segments(segs, tokenizer, cpc, schedule)
        all_tokens, all_words, all_latents = tokens, words, latents
        n_files_done = 1
    else:
        import torchaudio  # noqa: PLC0415
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        aligner, labels = bundle.get_model(), bundle.get_labels()
        flacs = sorted(librispeech.rglob("*.flac"))[:n_files]
        print(f"Track A: {len(flacs)} LibriSpeech utterances "
              f"(calibrator_fitted={calibrator is not None})")
        for fi, flac in enumerate(flacs):
            audio, sr = load_audio(str(flac), target_sr=SR)
            transcript = load_librispeech_transcript(flac)
            aln = forced_align(transcript, audio, sr, model=aligner, labels=labels)
            segs = segment_audio(audio, sr, profile, calibrator=calibrator, cpc=cpc)
            mres, per_type = _segment_report(segs, aln.word_starts, aln.phoneme_starts)
            word_prfs.append(mres.word)
            combined_prfs.append(mres.combined)
            for tname, prf in per_type.items():
                per_type_buckets[tname].append(prf)
            tokens, words, latents = _collect_segments(segs, tokenizer, cpc, aln.word_spans)
            all_tokens.extend(tokens)
            all_words.extend(words)
            if len(latents):
                all_latents = np.vstack([all_latents, latents])
            if (fi + 1) % 25 == 0:
                print(f"  {fi + 1}/{len(flacs)} files, {len(all_tokens)} labeled segments")
        n_files_done = len(flacs)

    seg_report = {
        "n_files": n_files_done,
        "word_F1": _pool(word_prfs),
        "combined_F1": _pool(combined_prfs),
        "per_type": {t: _pool(lst) for t, lst in per_type_buckets.items()},
    }

    consistency = token_label_consistency(all_tokens, all_words).as_dict()
    consistency["n_unique_tokens"] = len(set(all_tokens))
    consistency["n_active_tokens_ge2"] = (
        sum(1 for c in Counter(all_tokens).values() if c >= 2) if all_tokens else 0
    )

    if features_npz:
        latents_use, words_use = load_word_features(features_npz)
        probe = linear_probe(latents_use, words_use).as_dict()
        probe["source"] = f"precomputed npz ({features_npz})"
    else:
        probe = linear_probe(all_latents, all_words).as_dict()
        probe["source"] = "runtime CPC latents"

    return {
        "track": "A (engineering / human)",
        "species": species,
        "calibrated": calibrated,
        "calibrator_fitted": calibrator is not None,
        "segmentation": seg_report,
        "semantic_ceiling": consistency,
        "linear_probe": probe,
    }


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def _print(report: dict) -> None:
    s, c, p = report["segmentation"], report["semantic_ceiling"], report["linear_probe"]
    print("=" * 64)
    print(f"TRACK {report['track']} | species={report['species']} | n_files={s['n_files']}")
    print("=" * 64)
    print("§1 Segmentation accuracy (vs wav2vec2 word starts)")
    print(f"   word-F1:     {s['word_F1']['f1']:.3f}  "
          f"(P={s['word_F1']['precision']:.3f} R={s['word_F1']['recall']:.3f})")
    print(f"   combined-F1: {s['combined_F1']['f1']:.3f}")
    for t, prf in s["per_type"].items():
        print(f"     {t:14s} F1={prf['f1']:.3f} (P={prf['precision']:.3f} R={prf['recall']:.3f})")
    print("§2 Semantic ceiling (unsupervised token→word consistency)")
    if c.get("skipped"):
        print(f"   skipped: {c['skipped']}")
    else:
        print(f"   consistency={c['consistency']:.1%} (chance={c['chance_baseline']:.1%}, "
              f"lift={c['lift_over_chance']}×, unique tokens={c['n_unique_tokens']})")
    print("§3 Linear probe (word identity from CPC latents)")
    if p.get("skipped"):
        print(f"   skipped: {p['skipped']}")
    else:
        print(f"   accuracy={p['accuracy']:.1%} (chance={p['chance_baseline']:.1%}, "
              f"lift={p['lift_over_chance']}×, signal_exists={p['signal_exists']})")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Track A engineering validation on LibriSpeech")
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--librispeech", type=Path, default=DEFAULT_LIBRISPEECH)
    ap.add_argument("--features-npz", type=Path, default=None,
                    help="precomputed (cpc_latents, word_labels) npz for §3")
    ap.add_argument("--species", default="human")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic corpus, no LibriSpeech/network")
    ap.add_argument("--calibrated", action="store_true",
                    help="use the learned NBD calibrator instead of the fixed-threshold NBD")
    args = ap.parse_args()

    report = run(args.librispeech, args.n_files, args.features_npz, args.self_test,
                 args.species, args.calibrated)
    _print(report)

    out = args.json or (ROOT / "outputs" / "track_a_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
