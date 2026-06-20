#!/usr/bin/env python3
"""Cross-species comparison on a SHARED acoustic-fingerprint tokenization.

Same segmentation as the per-species runs, but the tokenization backbone is swapped from
per-species VQ-VAE-on-CPC to ONE shared k-means codebook over the dimensionless acoustic
fingerprint (all-species, firewall-legal, CPC-free). Token id X is the same acoustic-shape
class in human and bat, removing the per-species-tokenizer confound.

Drops the 5 absolute-Hz fingerprint features (zcr/centroid/spread/rolloff/dom_freq) so the
shared codebook clusters on acoustic SHAPE, not species-specific frequency bands.

Then: sufficiency gate + speaker/emitter-controlled functional aliasing + motif on each
species (at matched token counts), and bat context_binding (within-Emitter). The headline
is the de-confounded human-vs-bat aliasing contrast on identical tokenization.

Output: outputs/cross_species_fingerprint.json

Usage:
    python scripts/cross_species_fingerprint.py --n-human 220 --n-bat 1500 --k 128

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIBRISPEECH = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
BAT_AUDIO_DIR = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
ANNOT_CSV = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")
OUT = ROOT / "outputs" / "cross_species_fingerprint.json"
CALL_GRANULAR = {"sample_rate": 250000, "hop_ms": 10.0,
                 "energy_threshold": 0.05, "spectral_threshold": 0.05, "debounce_ms": 25.0}


def _speaker_of(p: Path) -> str:
    return p.stem.split("-")[0]


def _load_bat_annotations() -> dict[str, dict]:
    out = {}
    with ANNOT_CSV.open() as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("File Name") or "").strip()
            if fn:
                out[fn] = {"context": int(row["Context"]), "emitter": row["Emitter"].strip()}
    return out


def _gather_human(n_files, segment_audio, load_audio, profile, cal, cpc, afp):
    files = sorted(LIBRISPEECH.rglob("*.flac"))
    seqs, spks = [], []
    for f in files[:n_files]:
        try:
            a, sr = load_audio(str(f), target_sr=profile["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        seq_fp, spk = [], _speaker_of(f)
        for seg in segment_audio(a, sr, profile, calibrator=cal, cpc=cpc):
            fp = afp.extract(seg.audio, seg.sr, species="human").combined
            if not np.any(fp):
                continue
            seq_fp.append(fp)
        if seq_fp:
            seqs.append(seq_fp)
            spks.append(spk)
    return seqs, spks, files[:n_files]


def _gather_bat(n_files, segment_audio, load_audio, profile, annot, afp):
    files = [f for f in sorted(BAT_AUDIO_DIR.rglob("*.wav")) if f.name in annot][:n_files]
    seqs, emitters, contexts = [], [], []
    for f in files:
        try:
            a, sr = load_audio(str(f), target_sr=profile["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        seq_fp = []
        for seg in segment_audio(a, sr, profile):
            fp = afp.extract(seg.audio, seg.sr, species="bat").combined
            if not np.any(fp):
                continue
            seq_fp.append(fp)
        if seq_fp:
            seqs.append(seq_fp)
            emitters.append(annot[f.name]["emitter"])
            contexts.append(annot[f.name]["context"])
    return seqs, emitters, contexts, files


def _alias_report(seqs, group, n_perm):
    from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
        controlled_alias_tests, detect_functional_aliases,
    )
    rep = detect_functional_aliases(seqs, similarity_threshold=0.80, min_instances=10)
    elig = len(rep.singleton_tokens) + rep.n_merged
    density = rep.n_merged / max(elig, 1)
    res = controlled_alias_tests(seqs, emitter=group, n_permutations=n_perm, seed=0)
    per_p = {k: round(v.p_value, 4) for k, v in res.items()}
    eff_p = max(per_p.values()) if per_p else None
    return {"density": round(density, 4), "n_groups": rep.n_groups,
            "per_null_p": per_p, "effective_p": eff_p}


def _downsample_to_tokens(seqs, groups, target):
    """File-subset greedily to ~target tokens; returns (seqs, groups) (no RNG -> stable)."""
    acc_s, acc_g, tot = [], [], 0
    for s, g in zip(seqs, groups):
        acc_s.append(s)
        acc_g.append(g)
        tot += len(s)
        if tot >= target:
            break
    return acc_s, acc_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-human", type=int, default=220)
    ap.add_argument("--n-bat", type=int, default=1500)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-perm", type=int, default=150)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (  # noqa: PLC0415
        FingerprintTokenizer,
    )
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
    from bioacoustic_pipeline.orchestration import _default_calibrator, segment_audio  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    profiles = load_species_profiles()
    hprof = profiles["human"]
    cal = _default_calibrator("human")
    cpc = CPCBackbone() if (cal and getattr(cal, "_use_context", False)) else None
    bprof = {**profiles["bat"], **CALL_GRANULAR}
    annot = _load_bat_annotations()

    print(f"gathering human ({args.n_human} LibriSpeech)...")
    h_seqs_raw, h_spk, _ = _gather_human(args.n_human, segment_audio, load_audio, hprof, cal, cpc, afp)
    print(f"gathering bat ({args.n_bat} bouts)...")
    b_seqs_raw, b_emit, b_ctx, _ = _gather_bat(args.n_bat, segment_audio, load_audio, bprof, annot, afp)

    h_fps = [fp for s in h_seqs_raw for fp in s]
    b_fps = [fp for s in b_seqs_raw for fp in s]
    print(f"human: {len(h_seqs_raw)} utterances, {len(h_fps)} fingerprint-segments")
    print(f"bat:   {len(b_seqs_raw)} bouts, {len(b_fps)} fingerprint-segments")

    # shared codebook on pooled human+bat
    print(f"fitting shared fingerprint tokenizer (k={args.k}) on pooled {len(h_fps)+len(b_fps)} fps...")
    tok = FingerprintTokenizer.fit(h_fps + b_fps, k=args.k)
    print(f"  human utilization: {tok.utilization(h_fps)}")
    print(f"  bat   utilization: {tok.utilization(b_fps)}")

    def _tokenize_seqs(raw):
        return [[tok.tokenize(fp) for fp in s] for s in raw]
    h_seqs, b_seqs = _tokenize_seqs(h_seqs_raw), _tokenize_seqs(b_seqs_raw)
    h_tok = sum(len(s) for s in h_seqs)
    b_tok = sum(len(s) for s in b_seqs)
    h_suff = evaluate_sufficiency(h_seqs)
    b_suff = evaluate_sufficiency(b_seqs)
    print(f"\nhuman sufficiency: {h_suff.status} median={h_suff.median_tokens_per_bout:.1f} "
          f"bigrams={h_suff.total_bigrams} | bat: {b_suff.status} median={b_suff.median_tokens_per_bout:.1f} "
          f"bigrams={b_suff.total_bigrams}")

    # matched-size alias contrast (downsample the larger to the smaller)
    matched = min(h_tok, b_tok)
    h_ms, h_mg = _downsample_to_tokens(h_seqs, h_spk, matched)
    b_ms, b_mg = _downsample_to_tokens(b_seqs, b_emit, matched)
    print(f"\n--- functional aliasing on SHARED fingerprint tokens (matched ~{matched} tokens) ---")
    h_alias = _alias_report(h_ms, h_mg, args.n_perm) if evaluate_sufficiency(h_ms).evaluable else None
    b_alias = _alias_report(b_ms, b_mg, args.n_perm) if evaluate_sufficiency(b_ms).evaluable else None
    for name, rep in (("human", h_alias), ("bat", b_alias)):
        if rep:
            print(f"  {name:6s} density={rep['density']} groups={rep['n_groups']} "
                  f"eff_p={rep['effective_p']} nulls={rep['per_null_p']}")
        else:
            print(f"  {name:6s} NOT_EVALUABLE at matched size")

    # bat context binding on the shared tokens (within-Emitter). Build per-segment
    # (token, context, emitter) aligned lists from the tokenized bouts.
    b_tok_flat, b_ctx_flat, b_emit_flat = [], [], []
    for s, e, c in zip(b_seqs, b_emit, b_ctx):
        for t in s:
            b_tok_flat.append(t)
            b_ctx_flat.append(c)
            b_emit_flat.append(e)
    b_ctx_report = None
    if len(b_tok_flat) > 50:
        r = association_test([[t] for t in b_tok_flat], b_ctx_flat, statistic="nmi",
                             n_permutations=args.n_perm, seed=0,
                             group=b_emit_flat, null_type="within_group").as_dict()
        b_ctx_report = {k: r[k] for k in ("observed_nmi", "observed_chi2", "cramers_v",
                                          "p_value", "null_type")}
        print(f"\nbat context_binding (shared tokens): nmi={r['observed_nmi']:.3f} "
              f"V={r['cramers_v']:.3f} p={r['p_value']} [{r['null_type']}]")

    report = {
        "representation": "shared acoustic fingerprint (k-means, dimensionless features, CPC-free, firewall all-species)",
        "k": args.k, "dropped_hz_features": "zcr,centroid,spread,rolloff,dom_freq",
        "human": {"n_utterances": len(h_seqs_raw), "n_tokens": h_tok,
                  "utilization": tok.utilization(h_fps),
                  "sufficiency": h_suff.as_dict(), "alias_matched": h_alias},
        "bat": {"n_bouts": len(b_seqs_raw), "n_tokens": b_tok,
                "utilization": tok.utilization(b_fps),
                "sufficiency": b_suff.as_dict(), "alias_matched": b_alias,
                "context_binding": b_ctx_report},
        "matched_token_count": matched,
        "interpretation": (
            "Human and bat tokenized by the SAME shared fingerprint codebook "
            "(no per-species VQ-VAE confound). Compare alias_matched.density / effective_p."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
