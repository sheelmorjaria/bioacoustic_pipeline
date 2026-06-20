#!/usr/bin/env python3
"""Playback-design CLI — turn discovered structures into Phase-1 playback stimuli + manifest.

Phase 1 (exemplar concatenation). Generates controlled stimulus variants for a chosen
experiment, prefights them for acoustic naturalness, writes ``.wav`` stimuli + a canonical
playback manifest whose predictions are falsifiable hypotheses (NOT claims).

Sources:
  * ``synthetic`` (default, offline): deterministic token-tagged corpus with planted motifs —
    exercises the whole package end-to-end without real audio.
  * ``bat`` / ``human``: real corpora via the firewall-gated species tokenizer
    (``SpeciesAcousticTokenizer`` / human VQ-VAE). Heavier; synthetic is the verified default.

Usage:
  python scripts/playback_design.py --species synthetic --experiment motif_disruption \
      --n-files 30 --out-dir playback_out

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics import motif_miner as mm  # noqa: E402
from bioacoustic_pipeline.diagnostics.alias_detector import detect_functional_aliases  # noqa: E402
from bioacoustic_pipeline.experiments.playback_design import (  # noqa: E402
    affect_morphing,
    alias_substitution,
    candidate_selector,
    motif_variants,
    naturalness_preflight,
    playback_manifest,
    spatial_router,
    stimulus_builder,
)
from bioacoustic_pipeline.hashing import write_canonical_json  # noqa: E402
from bioacoustic_pipeline.utils.audio import write_audio  # noqa: E402

SR = 16000


# --------------------------------------------------------------------------- #
# Synthetic offline corpus (token-tagged, planted motifs)
# --------------------------------------------------------------------------- #


def _tone_burst(freq: float, sr: int = SR, dur_ms: int = 120) -> np.ndarray:
    t = np.arange(int(sr * dur_ms / 1000)) / sr
    env = np.minimum(1.0, np.linspace(0, 1, len(t))) * np.minimum(
        1.0, np.linspace(0, 1, len(t))[::-1]
    )
    return (0.5 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def synthetic_corpus(n_tokens: int, n_seqs: int, seed: int):
    """Return ``(annotated_segments, sequences, token_fp)`` with a planted motif [0,1,2]."""
    rng = np.random.default_rng(seed)
    freqs = [300.0 + 180.0 * i for i in range(n_tokens)]
    token_audio = {t: _tone_burst(freqs[t]) for t in range(n_tokens)}
    token_fp = {
        t: np.asarray(
            afp.extract(token_audio[t], SR, species="synthetic").combined, dtype=np.float32
        )
        for t in range(n_tokens)
    }
    annotated = [
        {
            "token": t,
            "audio": token_audio[t],
            "sr": SR,
            "fingerprint": token_fp[t],
            "segment_id": f"synth_tok{t}",
        }
        for t in range(n_tokens)
    ]
    seqs = []
    for _ in range(n_seqs):
        n = int(rng.integers(5, 9))
        seq = [int(rng.integers(0, n_tokens)) for _ in range(n)]
        if rng.random() < 0.7:  # plant the motif frequently so it is discoverable
            pos = int(rng.integers(0, len(seq) + 1))
            seq = seq[:pos] + [0, 1, 2] + seq[pos:]
        seqs.append(seq)
    return annotated, seqs, token_fp


# --------------------------------------------------------------------------- #
# Stimulus construction for one motif
# --------------------------------------------------------------------------- #


def _stimulus(
    out_dir: Path, motif_id: str, cond: str, pattern, typical, token_fp, sr: int, seed: int
):
    audio, prov = stimulus_builder.build_stimulus_audio(pattern, typical, sr, seed=seed)
    wav = out_dir / f"{motif_id}_{cond}.wav"
    write_audio(str(wav), audio, sr)
    fp_seq = [token_fp[int(t)] for t in pattern]
    return {
        "audio": audio,
        "wav": str(wav),
        "provenance": prov,
        "fp_seq": fp_seq,
        "pattern": list(pattern),
    }


def build_motif_disruption(out_dir, motif, typical, vocab, token_fp, sr, seed):
    pattern = [int(t) for t in motif.pattern]
    mid = motif.motif_id or "motif0"
    conditions = {
        "natural": pattern,
        "reversed": motif_variants.reverse(pattern),
        "shuffled": motif_variants.shuffle(pattern, seed=seed),
        "token_deleted": motif_variants.token_delete(pattern, seed=seed),
        "random_control": motif_variants.random_control(vocab, len(pattern), seed=seed),
    }
    built = {
        c: _stimulus(out_dir, mid, c, p, typical, token_fp, sr, seed + i)
        for i, (c, p) in enumerate(conditions.items())
    }
    natural = built["natural"]
    stimuli = []
    for c, b in built.items():
        pf = naturalness_preflight.preflight_stimulus(
            b["audio"], natural["audio"], sr, b["fp_seq"], natural["fp_seq"]
        )
        stimuli.append(
            {
                "stimulus_id": f"{mid}_{c}",
                "condition": c,
                "token_sequence": b["pattern"],
                "affect_trajectory": affect_morphing.morph_descriptors(
                    []
                ),  # no affect model -> not_available
                "spatial_condition": spatial_router.CORRECT_DIRECTION,
                "audio_file": b["wav"],
                "exemplar_provenance": b["provenance"],
                "naturalness_preflight": pf,
            }
        )
    return stimuli, natural


def add_alias_conditions(
    out_dir, stimuli, natural, motif, typical, vocab, token_fp, alias_rep, sr, seed
):
    mid = motif.motif_id or "motif0"
    pattern = [int(t) for t in motif.pattern]
    for label, fn in (
        ("alias_substitution", alias_substitution.alias_substitute_pattern),
        ("nonalias_control", alias_substitution.nonalias_control_pattern),
    ):
        new, info = (
            fn(pattern, alias_rep, vocab, seed=seed)
            if "nonalias" in label
            else fn(pattern, alias_rep, seed=seed)
        )
        if new is None:
            stimuli.append(
                {
                    "stimulus_id": f"{mid}_{label}",
                    "condition": label,
                    "token_sequence": None,
                    "note": "no eligible alias / control token available",
                }
            )
            continue
        b = _stimulus(out_dir, mid, label, new, typical, token_fp, sr, seed)
        pf = naturalness_preflight.preflight_stimulus(
            b["audio"], natural["audio"], sr, b["fp_seq"], natural["fp_seq"]
        )
        stimuli.append(
            {
                "stimulus_id": f"{mid}_{label}",
                "condition": label,
                "token_sequence": b["pattern"],
                "substitution": info,
                "affect_trajectory": affect_morphing.morph_descriptors([]),
                "spatial_condition": spatial_router.CORRECT_DIRECTION,
                "audio_file": b["wav"],
                "naturalness_preflight": pf,
            }
        )
    return stimuli


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-1 playback-design stimulus + manifest builder")
    ap.add_argument("--species", default="synthetic", choices=["synthetic", "bat", "human"])
    ap.add_argument(
        "--experiment", default="motif_disruption", choices=list(playback_manifest.EXPERIMENTS)
    )
    ap.add_argument(
        "--n-files", type=int, default=30, help="number of synthetic sequences / real bouts"
    )
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default="playback_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    stim_dir = out_dir / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)

    if args.species == "synthetic":
        annotated, sequences, token_fp = synthetic_corpus(args.n_tokens, args.n_files, args.seed)
    else:
        raise SystemExit(
            f"--species {args.species}: real-corpus path not wired in this script; "
            f"use --species synthetic for the offline demo"
        )

    pool = candidate_selector.build_exemplar_pool(annotated)
    typical = candidate_selector.typical_exemplars(pool, n_per_token=1)
    motifs = mm.build_motifs(mm.mine_ngrams(sequences, 2, 4, 2), sequences, motif_type="token")
    selected, vocab = candidate_selector.select_candidates(
        motifs, typical, n_motifs=3, min_support=2
    )
    if not selected:
        raise SystemExit("no eligible motifs (try more files / a lower min-support)")
    motif = selected[0]

    stimuli, natural = build_motif_disruption(
        stim_dir, motif, typical, vocab, token_fp, SR, args.seed
    )

    if args.experiment == "alias_substitution":
        alias_rep = detect_functional_aliases(sequences, similarity_threshold=0.5, min_instances=3)
        if alias_rep.groups:
            stimuli = add_alias_conditions(
                stim_dir,
                stimuli,
                natural,
                motif,
                typical,
                vocab,
                token_fp,
                alias_rep,
                SR,
                args.seed,
            )
        else:
            stimuli.append(
                {"condition": "alias_substitution", "note": "no functional-alias groups found"}
            )
    elif args.experiment == "affect_dissociation":
        # attach the affect-trajectory descriptor (not_available without a species affect model)
        for s in stimuli:
            s["affect_trajectory"] = affect_morphing.morph_descriptors([])
    elif args.experiment == "spatial_mismatch":
        # replay the natural motif under each spatial cell (same audio, different spatial tag)
        for sc in (
            spatial_router.CORRECT_DIRECTION,
            spatial_router.UNICAST_FROM_WRONG,
            spatial_router.BROADCAST,
        ):
            stimuli.append(
                {
                    "stimulus_id": f"{motif.motif_id}_natural_{sc}",
                    "condition": "natural",
                    "token_sequence": list(motif.pattern),
                    "spatial_condition": sc,
                    "audio_file": natural["wav"],
                }
            )
    elif args.experiment == "context_mismatch":
        for ctx in ("feeding", "resting", "conflict"):
            stimuli.append(
                {
                    "stimulus_id": f"{motif.motif_id}_natural_ctx_{ctx}",
                    "condition": "natural",
                    "token_sequence": list(motif.pattern),
                    "playback_context": ctx,
                    "spatial_condition": spatial_router.CORRECT_DIRECTION,
                    "audio_file": natural["wav"],
                }
            )

    manifest = playback_manifest.build_manifest(
        args.experiment,
        args.species,
        stimuli,
        source_motifs=[m.as_dict() for m in selected],
    )
    write_canonical_json(out_dir / "manifest.json", manifest)
    write_canonical_json(
        out_dir / "preflight.json",
        {
            "experiment": args.experiment,
            "per_stimulus": {
                s.get("stimulus_id", s.get("condition")): s.get("naturalness_preflight")
                for s in stimuli
                if s.get("naturalness_preflight")
            },
        },
    )
    n_pass = sum(1 for s in stimuli if s.get("naturalness_preflight", {}).get("pass"))
    print(
        f"wrote {out_dir}/  stimuli={len(stimuli)}  motif={list(motif.pattern)}  "
        f"preflight_pass={n_pass}  caveats={len(manifest['caveats'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
