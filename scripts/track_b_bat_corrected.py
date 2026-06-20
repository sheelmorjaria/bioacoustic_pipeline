#!/usr/bin/env python3
"""Track B (corrected) — annotation-grounded multi-axis bat verdict.

Fixes the two confounds in the earlier bat run:
  1. Holophrasic NBD profile collapsed bouts to ~1 token -> the sequence-sufficiency
     gate now abstains; here we re-segment with the plausibility-selected
     ``bat_call_granular`` profile so bouts yield multi-token sequences.
  2. Tokenization happened off the firewall-gated path -> here we tokenize with the
     registered ``bat_vqvae`` species tokenizer (CPC provenance ``bat_cpc``), checked
     by the ethological firewall for bat.

Then it runs the annotation-grounded battery the bout labels (Emitter / Addressee /
Context) enable:
  * functional aliasing + speaker/context-controlled nulls (alias must survive all)
  * context binding: token<->Context (within-Emitter null), token<->Emitter, token<->Addressee
  * motif structure, LZ, call unitization, acoustic coherence (fingerprint)
and assembles a schema-valid multi-axis verdict with ``not_evaluable`` where the
sequence is too sparse to test.

Output: outputs/track_b_bat_corrected.json

Usage:
    python scripts/track_b_bat_corrected.py --n-files 1000

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BAT_AUDIO_DIR = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
ANNOT_CSV = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")
OUT = ROOT / "outputs" / "track_b_bat_corrected.json"

# Plausibility-selected call-granular profile (from bat_segmentation_sweep).
CALL_GRANULAR = {
    "sample_rate": 250000, "hop_ms": 10.0,
    "energy_threshold": 0.05, "spectral_threshold": 0.05, "debounce_ms": 25.0,
}


def load_annotations() -> dict[str, dict]:
    """file basename -> {context, emitter, addressee} from the bout CSV."""
    out: dict[str, dict] = {}
    with ANNOT_CSV.open() as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("File Name") or "").strip()
            if not fn:
                continue
            out[fn] = {
                "context": int(row["Context"]),
                "emitter": row["Emitter"].strip(),
                "addressee": row["Addressee"].strip(),
            }
    return out


def _binding_axis(result, label_name: str):
    """Map a context_binding BindingResult (already speaker-controlled for Context) -> AxisResult."""
    from bioacoustic_pipeline.verdict_engine.evidence_profile import AxisResult  # noqa: PLC0415

    if result is None:
        return AxisResult("context_binding", "absent", 0.9, "no annotation test")
    sig = result.p_value is not None and result.p_value < 0.05
    note = (
        f"token<->{label_name} nmi={result.observed_nmi:.3f} chi2={result.observed_chi2:.0f} "
        f"V={result.cramers_v:.3f} p={result.p_value:.3f} [{result.null_type}, q={result.q_value}]"
    )
    if not sig:
        return AxisResult("context_binding", "absent", 0.85, note + " — not significant")
    level = "strong" if result.observed_nmi >= 0.05 else "weak"
    conf = min(1.0, 1.0 - result.p_value)
    return AxisResult(
        "context_binding", level, conf,
        note + " — functional-CONTEXT evidence (NOT validated meaning; needs Track C playback)",
    )


def main():
    ap = argparse.ArgumentParser(description="Corrected annotation-grounded bat verdict")
    ap.add_argument("--audio-dir", type=Path, default=BAT_AUDIO_DIR)
    ap.add_argument("--n-files", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
        controlled_alias_tests,
        detect_functional_aliases,
    )
    from bioacoustic_pipeline.diagnostics.context_binding import test_token_annotations  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics import motif_miner as mm  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
    from bioacoustic_pipeline.orchestration import segment_audio  # noqa: PLC0415
    from bioacoustic_pipeline.tokenization.species_acoustic_tokenizer import (  # noqa: PLC0415
        SpeciesAcousticTokenizer,
    )
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415
    from bioacoustic_pipeline.verdict_engine import is_valid  # noqa: PLC0415
    from bioacoustic_pipeline.verdict_engine.engine import report_from_profile  # noqa: PLC0415
    from bioacoustic_pipeline.verdict_engine.evidence_profile import (  # noqa: PLC0415
        compute_evidence_profile,
        score_functional_aliasing,
    )

    fw = default_firewall()
    for name in ("bat_cpc", "bat_vqvae"):
        fw.assert_allowed(name, Use.TOKENIZE)
        fw.assert_cross_species(name, "bat")
    print("firewall: bat_cpc + bat_vqvae legal for bat tokenize")

    tk = SpeciesAcousticTokenizer("bat", "bat_cpc", "bat_vqvae")
    if not tk.available:
        raise SystemExit(f"species tokenizer unavailable: {tk.load_error}")

    annot = load_annotations()
    profile = {**load_species_profiles()["bat"], **CALL_GRANULAR}
    files = sorted(args.audio_dir.rglob("*.wav"))
    # keep only files with annotations, in stable order
    files = [f for f in files if f.name in annot][: args.n_files]
    print(f"processing {len(files)} annotated bouts (call-granular segmentation)...")

    sequences, contexts, emitters, addressees = [], [], [], []
    tokens_flat, btypes = [], []
    n_segments = n_loaded = 0
    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(str(f), target_sr=profile["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        n_loaded += 1
        segs = segment_audio(audio, sr, profile)
        n_segments += len(segs)
        seq = []
        for seg in segs:
            tok = tk.tokenize(seg.audio, seg.sr)
            if tok == -1:
                continue
            seq.append(tok)
            tokens_flat.append(tok)
            btypes.append(seg.boundary_type)
        sequences.append(seq)
        a = annot[f.name]
        contexts.append(a["context"])
        emitters.append(a["emitter"])
        addressees.append(a["addressee"])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} bouts, {len(tokens_flat)} tokens")

    suff = evaluate_sufficiency(sequences)
    print(f"\nsufficiency: {suff.status} [{suff.reason}] | "
          f"median tokens/bout={suff.median_tokens_per_bout:.1f} "
          f"nonzero_rate={suff.nonzero_bout_rate:.2f} bigrams={suff.total_bigrams}")

    # Boundary / LZ signals (gate handles sequence axes if not evaluable).
    n_hard = sum(1 for b in btypes if b == "hard")
    n_soft = sum(1 for b in btypes if b == "soft")
    lz = lempel_ziv_complexity(tokens_flat[:5000]).normalized

    alias_trace, controlled_trace = None, None
    if suff.evaluable:
        rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
        elig = len(rep.singleton_tokens) + rep.n_merged
        ad = rep.n_merged / max(elig, 1)
        controlled_trace = controlled_alias_tests(
            sequences, emitter=emitters, context=contexts, n_permutations=args.n_perm, seed=0
        )
        controlled_trace = {k: round(v.p_value, 4) for k, v in controlled_trace.items()}
        # Aliasing is accepted only if it survives EVERY null -> take the weakest p.
        eff_p = max(controlled_trace.values()) if controlled_trace else None
        alias_trace = {"alias_density": round(ad, 4), "n_groups": rep.n_groups,
                       "effective_p_survives_all": eff_p,
                       "per_null_p": controlled_trace}
    else:
        ad, eff_p = 0.0, None

    # Motifs + null (only meaningful if evaluable).
    motif_trace = None
    if suff.evaluable and sum(len(s) for s in sequences) > 50:
        raw = mm.mine_ngrams(sequences, min_n=2, max_n=5, min_support=3)
        motifs = mm.build_motifs(raw, sequences, motif_type="token")
        try:
            mp = round(float(mm.motif_support_null(sequences, n_permutations=args.n_perm, seed=0).p_value), 4)
        except Exception:  # noqa: BLE001
            mp = None
        motif_trace = {"n_motifs": len(motifs), "motif_null_p": mp,
                       "motifs_per_1k_tokens": round(len(motifs) / max(len(tokens_flat) / 1000, 1e-9), 2)}

    # Annotation context binding (Context uses within-Emitter null; BH-FDR across family).
    binding = test_token_annotations(
        sequences,
        {"Context": contexts, "Emitter": emitters, "Addressee": addressees},
        n_permutations=args.n_perm, seed=0, emitter=emitters,
    )
    binding_dict = {k: v.as_dict() for k, v in binding.items()}

    # Assemble the multi-axis profile, then override functional_aliasing + context_binding.
    p = compute_evidence_profile(
        "bat", lz_normalized=lz, alias_density=ad, alias_pvalue=eff_p,
        n_hard=n_hard, n_soft=n_soft, n_total_boundaries=n_hard + n_soft,
        sufficiency=suff,
        n_motifs=(motif_trace or {}).get("n_motifs"),
        motif_null_p=(motif_trace or {}).get("motif_null_p"),
        context_evaluable=True,
        turn_evaluable=False,
        turn_reason="annotations lack timestamps; bout order is not guaranteed temporal",
    )
    # Override functional_aliasing so its evidence cites the surviving-null p, and
    # context_binding with the annotation-grounded, speaker-controlled result.
    if suff.evaluable and eff_p is not None:
        p.axes["functional_aliasing"] = score_functional_aliasing(ad, eff_p)
    else:
        p.axes["functional_aliasing"] = p.axes.get(
            "functional_aliasing",
            score_functional_aliasing(ad, eff_p),
        )
    p.axes["context_binding"] = _binding_axis(binding.get("Context"), "Context")
    report = report_from_profile(p)

    trace = {
        "species": "bat",
        "segmentation_profile": "bat_call_granular",
        "tokenizer_provenance": "bat_vqvae (CPC: bat_cpc) — firewall-legal",
        "n_bouts": len(files), "n_bouts_loaded": n_loaded,
        "n_segments": n_segments, "n_tokens": len(tokens_flat),
        "sequence_sufficiency": suff.as_dict(),
        "alias": alias_trace,
        "motif": motif_trace,
        "context_binding": binding_dict,
        "lz_normalized": round(lz, 4),
        "schema_valid": is_valid(report),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"report": report, "trace": trace}, indent=2))

    print("\n=== CORRECTED BAT VERDICT ===")
    for name, ax in report["axes"].items():
        print(f"  {name:24s} {ax['level']:16s} conf={ax['confidence']:.0%}")
    print(f"\nschema_valid={is_valid(report)}  ->  {OUT}")
    if binding_dict:
        print("\ncontext binding (token<->label, nmi / p [null] / q):")
        for k, v in binding_dict.items():
            print(f"  {k:10s} nmi={v['observed_nmi']:.3f} p={v['p_value']} [{v['null_type']}] q={v['q_value']}")


if __name__ == "__main__":
    main()
