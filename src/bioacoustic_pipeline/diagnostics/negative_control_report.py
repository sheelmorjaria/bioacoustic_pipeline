"""Negative-control report — acoustic structure computed, communication inference refused.

For an out-of-domain corpus (e.g. BEANS-Zero ESC-50 environmental sound), the pipeline must
*detect acoustic structure but refuse to infer biological communication*. This builder runs
only the firewall-safe, species-agnostic acoustic stages (fixed-threshold NBD + acoustic
fingerprint + fingerprint-k-means tokens) per clip, attaches the domain gate (so the
ethological axes are ``not_applicable``), and aggregates into a dataset-level report.

Learned human/bat-only backbones (CPC, BioMAE, β-VAE) are deliberately NOT run — recorded
as ``not_computed_*_only`` — keeping the negative control offline, torch-free, and
firewall-clean (only the ``acoustic_fingerprint`` / spectrogram all-species views apply).

Reuses: ``orchestration.segment_audio``, ``acoustic_fingerprint.extract``,
``FingerprintTokenizerAdapter`` (tokenization/fingerprint_tokenizer_adapter),
``verdict_engine.build_report`` (+ ``diagnostics.domain_gate``).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
from bioacoustic_pipeline.diagnostics.domain_gate import (
    DISABLED_AXES,
    NON_BIOACOUSTIC_COMMUNICATION,
    gate_clip,
    gate_dataset,
    summarize_dataset,
)
from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity
from bioacoustic_pipeline.orchestration import segment_audio
from bioacoustic_pipeline.tokenization.fingerprint_tokenizer_adapter import (
    FingerprintTokenizerAdapter,
)
from bioacoustic_pipeline.verdict_engine import build_report

NOT_COMPUTED_HUMAN_ONLY = "not_computed_human_only"
NOT_COMPUTED_SPECIES_ONLY = "not_computed_species_only_backbone"


def _fit_fingerprint_tokenizer(clips, profile, species, k, seed):
    """Segment every clip + extract fingerprints -> fit one shared seeded codebook."""
    sr = profile.get("sample_rate", 16000)
    matrix: list[np.ndarray] = []
    for audio, _sr, _label in clips:
        segs = segment_audio(audio, sr, profile, calibrator=None)  # fixed-threshold, torch-free
        for seg in segs:
            fp = afp.extract(
                seg.audio, seg.sr, start_ms=seg.start_ms, end_ms=seg.end_ms, species=species
            )
            if np.any(fp.combined):
                matrix.append(np.asarray(fp.combined, dtype=np.float64))
    if not matrix:
        return None, 0
    from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import FingerprintTokenizer  # noqa: PLC0415

    fitted = FingerprintTokenizer.fit(np.stack(matrix), k=k, seed=seed)
    return fitted, len(matrix)


def _per_clip(audio, sr, label, profile, tokenizer, species, dataset):
    """Acoustic stages + schema-valid gate-rejected verdict for one clip."""
    segs = segment_audio(audio, sr, profile, calibrator=None)
    tokens: list[int] = []
    boundary_types: list[str] = []
    n_fingerprinted = 0
    for seg in segs:
        fp = afp.extract(
            seg.audio, seg.sr, start_ms=seg.start_ms, end_ms=seg.end_ms, species=species
        )
        if np.any(fp.combined):
            n_fingerprinted += 1
        if tokenizer is not None:
            tok = tokenizer.tokenize(seg.audio, seg.sr)
            if tok != -1:
                tokens.append(tok)
                boundary_types.append(seg.boundary_type)
    lz = lempel_ziv_complexity(tokens[:5000]).normalized if tokens else 0.0
    n_hard = sum(1 for b in boundary_types if b == "hard")
    n_soft = sum(1 for b in boundary_types if b == "soft")
    gate = gate_clip(dataset, label)
    report = build_report(
        species,
        lz_normalized=lz,
        n_hard=n_hard,
        n_soft=n_soft,
        n_total_boundaries=n_hard + n_soft,
        domain_gate=gate,
    )
    return {
        "label": label,
        "source_type": gate.source_type,
        "clip_verdict": gate.verdict,
        "n_segments": len(segs),
        "n_tokens": len(tokens),
        "n_fingerprinted": n_fingerprinted,
        "overall_confidence": report["overall_confidence"],
        "axes": report["axes"],
    }


def build_negative_control_report(
    dataset,
    clips,
    *,
    species="none",
    profile=None,
    k: int = 32,
    seed: int = 0,
) -> dict:
    """Build the dataset-level negative-control report.

    Args:
        dataset: dataset id (e.g. "esc-50").
        clips: iterable of (audio, sr, label) tuples (label is the ESC-50 class string).
        species: nominal species label for the report (default "none").
        profile: species NBD profile dict (generic acoustic profile; default human).
        k, seed: fingerprint k-means codebook size + seed.
    """
    if profile is None:
        from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415

        profile = load_species_profiles()["human"]
    clips = list(clips)

    fitted, n_fit = _fit_fingerprint_tokenizer(clips, profile, species, k, seed)
    tokenizer = FingerprintTokenizerAdapter(fitted, species=species) if fitted is not None else None

    clip_reports = [
        _per_clip(audio, sr, label, profile, tokenizer, species, dataset)
        for audio, sr, label in clips
    ]
    gates = [gate_clip(dataset, cr["label"]) for cr in clip_reports]
    dataset_gate = gate_dataset(dataset)

    summary = summarize_dataset(gates)
    n_segments = sum(cr["n_segments"] for cr in clip_reports)
    n_tokens = sum(cr["n_tokens"] for cr in clip_reports)
    n_fingerprinted = sum(cr["n_fingerprinted"] for cr in clip_reports)

    return {
        "dataset": f"BEANS-Zero/{dataset}" if dataset == "esc-50" else dataset,
        "dataset_role": dataset_gate.dataset_role,
        "sample_count": len(clip_reports),
        "label_space": "50 environmental sound classes" if dataset == "esc-50" else "environmental",
        "domain_gate": {
            "status": dataset_gate.communication_status,
            "verdict": dataset_gate.dataset_verdict,
            "reason": list(dataset_gate.reason),
            "confidence": round(dataset_gate.confidence, 3),
        },
        "pipeline_outputs": {
            "nbd_segments": f"computed ({n_segments} total)",
            "acoustic_fingerprints": f"computed ({n_fingerprinted} segment fingerprints)",
            "spectrogram": "available (deterministic log-mel; not materialized here)",
            "vq_tokens": (
                f"computed ({n_tokens} tokens via acoustic-fingerprint k-means)"
                if tokenizer is not None
                else "not_computed_no_fingerprintable_segments"
            ),
            "beta_vae_affect": NOT_COMPUTED_HUMAN_ONLY,
            "cpc_embeddings": NOT_COMPUTED_HUMAN_ONLY,
            "biomae_embeddings": NOT_COMPUTED_SPECIES_ONLY,
        },
        "disabled_biological_diagnostics": {ax: "NOT_APPLICABLE" for ax in sorted(DISABLED_AXES)},
        "summary": summary,
        "clips": clip_reports,
        "final_verdict": {
            "class": NON_BIOACOUSTIC_COMMUNICATION,
            "confidence": round(dataset_gate.confidence, 3),
            "interpretation": (
                "The audio contains environmental sound events rather than an annotated animal "
                "communication sequence. Acoustic fingerprints, NBD segment summaries, and discrete "
                "acoustic tokens describe sound structure, but the domain gate refuses any "
                "ethological-communication claim (no call-type timestamps, emitter/addressee, or "
                "behavioral context). Biological-source clips (e.g. dog, crickets) are labeled "
                "BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK, not communication candidates."
            ),
        },
    }
