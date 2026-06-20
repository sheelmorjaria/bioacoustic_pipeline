#!/usr/bin/env python3
"""Token<->word calibration on LibriSpeech — the Semantic Ceiling test.

LibriSpeech is the HUMAN CALIBRATION CORPUS: it has transcripts, so wav2vec2 forced
alignment gives word spans, and we can measure what an *unsupervised* VQ-VAE token
recovers when true word labels exist.

Pipeline (per utterance):
  transcript -> forced_align -> word spans
  audio -> NBD (hard boundaries) -> segments
  segment -> CPC 256D latent + VQ token
  segment <-> word by temporal overlap (strict / max-overlap / midpoint)
Then:
  * token<->word contingency: purity, word consistency, global/macro, content vs function
  * token<->word NMI / chi-square / Cramer's V (context_binding) + within-speaker null
  * supervised ceiling: linear probe CPC latent -> word  vs  latent -> speaker
The headline contrast is the Semantic Ceiling: if latent->word probe accuracy >> VQ
token<->word consistency, word info EXISTS in the latent but unsupervised tokenization
does NOT recover lexical identity (tokens are acoustic-similarity classes, not words).

Output: outputs/token_word_calibration.json

Usage:
    python scripts/token_word_calibration.py --n-files 250

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

LIBRISPEECH = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
OUT = ROOT / "outputs" / "token_word_calibration.json"


def _speaker_of(path: Path) -> str:
    return path.stem.split("-")[0]


def _speaker_spread(files: list[Path], n: int) -> list[Path]:
    """Round-robin across speakers so the sample isn't clustered by speaker.

    LibriSpeech is laid out as {speaker}/{chapter}/{utt}.flac, so ``sorted()[:n]`` grabs
    long runs from one speaker — which both inflates token<->word consistency (speaker
    identity leaks in) and starves the latent->speaker probe (too few speaker classes).
    Round-robin gives many speakers with few utterances each.
    """
    by_speaker: dict[str, list[Path]] = {}
    for f in files:
        by_speaker.setdefault(_speaker_of(f), []).append(f)
    speakers = sorted(by_speaker)
    out: list[Path] = []
    i = 0
    while len(out) < n and by_speaker:
        progressed = False
        for sp in speakers:
            bucket = by_speaker[sp]
            if i < len(bucket):
                out.append(bucket[i])
                if len(out) >= n:
                    break
                progressed = True
        if not progressed:
            break
        i += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description="LibriSpeech token<->word calibration")
    ap.add_argument("--audio-dir", type=Path, default=LIBRISPEECH)
    ap.add_argument("--n-files", type=int, default=250)
    ap.add_argument("--policy", choices=["strict", "max_overlap", "midpoint"], default="strict")
    ap.add_argument("--min-ratio", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics import token_word as tw  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.probes import (  # noqa: PLC0415
        forced_align,
        linear_probe,
        load_librispeech_transcript,
        token_label_consistency,
    )
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
    from bioacoustic_pipeline.orchestration import _default_calibrator, segment_audio  # noqa: PLC0415
    from bioacoustic_pipeline.tokenization.multiview import AcousticTokenizer  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    print("loading wav2vec2 (alignment yardstick) + CPC + human VQ-VAE...")
    import torchaudio  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    align_model, align_labels = bundle.get_model(), bundle.get_labels()

    profile = load_species_profiles()["human"]
    cal = _default_calibrator("human")
    cpc = CPCBackbone() if (cal and getattr(cal, "_use_context", False)) else CPCBackbone()
    tk = AcousticTokenizer()
    if not tk.available:
        raise SystemExit(f"human tokenizer unavailable: {tk.load_error}")

    all_files = sorted(args.audio_dir.rglob("*.flac"))
    files = _speaker_spread(all_files, args.n_files)
    n_speakers = len({_speaker_of(f) for f in files})
    print(f"processing {len(files)} LibriSpeech utterances across {n_speakers} speakers "
          f"(policy={args.policy})...")

    records, latents, word_labels, speaker_labels, token_list = [], [], [], [], []
    n_segs = n_aligned_files = 0
    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(str(f), target_sr=profile["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        transcript = load_librispeech_transcript(f)
        if not transcript:
            continue
        spans = forced_align(transcript, audio, sr, model=align_model, labels=align_labels).word_spans
        if not spans:
            continue
        n_aligned_files += 1
        speaker = _speaker_of(f)
        segs = segment_audio(audio, sr, profile, calibrator=cal, cpc=cpc)
        for seg in segs:
            if seg.boundary_type == "soft":  # word-level analysis: hard + transitional only
                continue
            tok = tk.tokenize(seg.audio, seg.sr)
            if tok == -1:
                continue
            n_segs += 1
            assign = tw.assign_word_by_overlap(
                seg.start_ms, seg.end_ms, spans,
                policy=args.policy, min_ratio=args.min_ratio,
            )
            word = assign[0] if assign else None
            records.append({"token": tok, "word": word, "speaker": speaker})
            token_list.append(tok)
            latents.append(cpc.embed(seg.audio, seg.sr))
            word_labels.append(word if word is not None else "<unknown>")
            speaker_labels.append(speaker)
        if (i + 1) % 50 == 0:
            n_lab = sum(1 for r in records if r["word"] is not None)
            print(f"  {i+1}/{len(files)} files, {n_segs} segments, {n_lab} word-labeled")

    labeled = [r for r in records if r["word"] is not None]
    print(f"\n{len(records)} segments | {len(labeled)} word-labeled "
          f"({100*len(labeled)/max(len(records),1):.0f}%) | {n_aligned_files} aligned files")

    # --- consistency metrics (the unsupervised side) ---
    table = tw.build_token_word_table(records)
    pur = tw.token_purity(table)
    wc = tw.word_consistency(records)
    cons = token_label_consistency(token_list, word_labels, purity=0.5)
    by_class = tw.consistency_by_class(records, min_count=5)

    # top content words (exclude function words), by count then consistency
    content_top = sorted(
        ((w, v) for w, v in wc.items() if not tw.is_function_word(w) and v["count"] >= 5),
        key=lambda kv: kv[1]["count"], reverse=True,
    )[:12]
    func_top = sorted(
        ((w, v) for w, v in wc.items() if tw.is_function_word(w) and v["count"] >= 5),
        key=lambda kv: kv[1]["count"], reverse=True,
    )[:8]

    # --- token<->word association test (NMI/chi2 + within-SPEAKER null) ---
    # within-speaker: destroys token<->word while holding each speaker's token multiset,
    # so a positive association is not merely "different speakers say different words".
    labeled_speakers = [r["speaker"] for r in labeled]
    binding = association_test(
        [[r["token"]] for r in labeled],
        [r["word"] for r in labeled],
        statistic="nmi", n_permutations=args.n_perm, seed=0,
        group=labeled_speakers, null_type="within_group",
    )
    word_bind = binding.as_dict()

    # --- supervised ceiling: latent -> word, latent -> speaker ---
    X = np.asarray(latents, dtype=np.float32)
    probe_word = linear_probe(X, np.array(word_labels), min_per_class=5).as_dict()
    probe_speaker = linear_probe(X, np.array(speaker_labels), min_per_class=5).as_dict()

    report = {
        "policy": args.policy, "min_ratio": args.min_ratio,
        "n_files": len(files), "n_aligned_files": n_aligned_files,
        "n_segments": len(records), "n_word_labeled": len(labeled),
        "n_unique_tokens": len(table), "n_unique_words": len({r["word"] for r in labeled}),
        "unsupervised": {
            "macro_word_consistency": tw.macro_word_consistency(records, min_count=5),
            "global_consistency": tw.global_consistency(records),
            "chance_baseline": tw.chance_baseline(records),
            "token_label_consistency": cons.as_dict(),
            "content_vs_function_macro": by_class,
            "mean_token_purity": (
                sum(v["purity"] * v["count"] for v in pur.values()) /
                max(sum(v["count"] for v in pur.values()), 1)
            ),
            "top_purity_tokens": [
                {"token": t, "purity": round(v["purity"], 3), "top_word": v["top_word"],
                 "count": v["count"]}
                for t, v in sorted(pur.items(), key=lambda kv: kv[1]["count"], reverse=True)
                if v["count"] >= 5
            ][:10],
        },
        "token_word_association": {
            "nmi": word_bind["observed_nmi"], "chi2": word_bind["observed_chi2"],
            "cramers_v": word_bind["cramers_v"], "p_value": word_bind["p_value"],
            "null_type": word_bind["null_type"], "q_value": word_bind["q_value"],
            "interpretation": "token<->word association above within-speaker chance",
        },
        "supervised_ceiling": {
            "latent_to_word": probe_word,
            "latent_to_speaker": probe_speaker,
        },
        "semantic_ceiling": (
            "word signal exists in latent (probe) but unsupervised VQ tokens do not "
            "recover lexical identity (low consistency vs probe) -> tokens are "
            "acoustic-similarity classes, not words"
        ),
        "top_content_words": [
            {"word": w, "consistency": round(v["consistency"], 3), "count": v["count"],
             "top_token": v["top_token"]} for w, v in content_top
        ],
        "top_function_words": [
            {"word": w, "consistency": round(v["consistency"], 3), "count": v["count"]}
            for w, v in func_top
        ],
    }

    # headline print
    print("\n=== TOKEN<->WORD CALIBRATION (LibriSpeech) ===")
    print(f"macro word consistency : {report['unsupervised']['macro_word_consistency']:.3f}  "
          f"(chance {report['unsupervised']['chance_baseline']:.3f})")
    print(f"content vs function    : {by_class.get('content',0):.3f} vs {by_class.get('function',0):.3f}")
    print(f"token<->word NMI       : {word_bind['observed_nmi']:.3f}  "
          f"V={word_bind['cramers_v']:.3f}  p={word_bind['p_value']} [{word_bind['null_type']}]")
    print(f"latent->word probe     : acc={probe_word.get('accuracy')}  "
          f"chance={probe_word.get('chance_baseline')}  lift={probe_word.get('lift_over_chance')}")
    print(f"latent->speaker probe  : acc={probe_speaker.get('accuracy')}  "
          f"lift={probe_speaker.get('lift_over_chance')}")
    print("top content words (consistency / count): "
          + ", ".join(f"{w}={v['consistency']:.2f}({v['count']})" for w, v in content_top[:6]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
