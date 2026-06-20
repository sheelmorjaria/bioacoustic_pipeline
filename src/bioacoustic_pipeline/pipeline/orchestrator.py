"""Canonical typed orchestrator — the single entry point Phase 2 diagnostics plug into.

``run(...)`` resolves config -> domain gate -> segment -> build a ``DualStreamSegment``
event stream -> run the ``DiagnosticRegistry`` -> assemble the verdict -> return a
``PipelineResult`` (and optionally write the canonical artifact bundle). The legacy
``orchestration.run_pipeline`` becomes a thin wrapper over this.

Phase 1: diagnostics still consume primitives via adapters that read off the stream
(``ctx`` carries sequences/tokens/boundary_types). Phase 2 migrates diagnostics to
consume the events directly — no orchestrator change needed.

Reuses: ``orchestration.segment_audio``, ``representation.dual_stream.build_dual_stream``,
``verdict_engine.build_report``/``is_valid``, ``determinism.build_repro_manifest``,
``acoustic_fingerprint.extract``, ``domain_gate.gate_from_manifest``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
from bioacoustic_pipeline.diagnostics.domain_gate import gate_from_manifest
from bioacoustic_pipeline.determinism import build_repro_manifest
from bioacoustic_pipeline.embeddings.segment_representation import assert_compatible
from bioacoustic_pipeline.orchestration import segment_audio
from bioacoustic_pipeline.pipeline.config import ConfigResolver
from bioacoustic_pipeline.pipeline.contracts import PipelineResult
from bioacoustic_pipeline.pipeline.registry import default_registry
from bioacoustic_pipeline.representation.dual_stream import build_dual_stream
from bioacoustic_pipeline.utils.audio import load_audio
from bioacoustic_pipeline.verdict_engine import build_report, is_valid

REPO_ROOT = Path(__file__).resolve().parents[3]


def run(
    audio_paths,
    *,
    species: str,
    dataset: str | None = None,
    annotations: str | None = None,
    config: dict | None = None,
    segment_source=None,
    tokenizer=None,
    fingerprint: bool = True,
    calibrator=False,
    out_dir: str | Path | None = None,
    seed: int = 0,
    registry=None,
) -> PipelineResult:
    """Canonical typed pipeline run. Returns a :class:`PipelineResult`."""
    cfg = ConfigResolver.resolve(
        species=species,
        dataset=dataset,
        config=config,
        segment_source=segment_source,
        seed=seed,
    )
    gate = gate_from_manifest(
        dataset or species, has_context=bool(annotations), annotations=annotations
    )
    profile = cfg.profile
    sr = profile.get("sample_rate")

    tk_available = bool(tokenizer is not None and getattr(tokenizer, "available", False))
    if cfg.segment_source is not None and tk_available:
        assert_compatible(tokenizer, cfg.segment_source, head_kind="vq_vae")

    fp_fn = (lambda a, s: afp.extract(a, s, species=species).combined) if fingerprint else None

    events: list = []
    segments_records: list[dict] = []
    sequences: list[list[int]] = []
    tokens: list[int] = []
    boundary_types: list[str] = []
    n_hard = n_soft = n_files = n_loaded = n_segments = 0

    for path in sorted(audio_paths):
        n_files += 1
        try:
            audio, asr = load_audio(str(path), target_sr=sr)
        except Exception:  # noqa: BLE001 — QA skips unreadable files
            continue
        n_loaded += 1
        segs = segment_audio(
            audio, asr, profile, calibrator=None if calibrator is False else calibrator
        )
        n_segments += len(segs)
        ds = build_dual_stream(
            segs, tokenizer=tokenizer if tk_available else None, fingerprint_fn=fp_fn
        )
        seq: list[int] = []
        for e in ds:
            events.append(e)
            segments_records.append(
                {
                    "file_idx": n_loaded - 1,
                    "segment_id": e.segment_id,
                    "start_ms": round(float(e.start_ms), 4),
                    "end_ms": round(float(e.end_ms), 4),
                    "boundary_type": e.boundary_type,
                    "confidence": round(float(e.boundary_confidence), 4),
                }
            )
            if e.vq_token != -1:
                tokens.append(e.vq_token)
                boundary_types.append(e.boundary_type)
                seq.append(e.vq_token)
            if e.boundary_type == "hard":
                n_hard += 1
            elif e.boundary_type == "soft":
                n_soft += 1
        sequences.append(seq)

    ctx = {
        "sequences": sequences,
        "tokens": tokens,
        "boundary_types": boundary_types,
        "n_hard": n_hard,
        "n_soft": n_soft,
        "seed": seed,
        "domain_gate": gate,
        "sufficiency": None,
    }
    reg = registry or default_registry()
    diagnostics = reg.run(events, ctx)
    suff = ctx.get("sufficiency")

    lz_dr = diagnostics.get("lempel_ziv")
    lz_norm = lz_dr.value["normalized"] if (lz_dr and lz_dr.computed) else 0.0
    mo_dr = diagnostics.get("motif")
    n_motifs = mo_dr.value.get("n_motifs") if (mo_dr and mo_dr.computed) else None
    motif_p = mo_dr.value.get("motif_null_p") if (mo_dr and mo_dr.computed) else None

    report = build_report(
        species,
        lz_normalized=lz_norm,
        n_hard=n_hard,
        n_soft=n_soft,
        n_total_boundaries=n_hard + n_soft,
        n_motifs=n_motifs,
        motif_null_p=motif_p,
        sufficiency=suff,
        domain_gate=gate if gate.rejected else None,
    )

    manifest = build_repro_manifest(
        species=species,
        seed=seed,
        audio_paths=sorted(audio_paths),
        profile=profile,
        tokenizer_info=None,
        calibrator_fitted=False,
        repo=REPO_ROOT,
    )
    trace = {
        "species": species,
        "n_files": n_files,
        "n_files_loaded": n_loaded,
        "n_segments": n_segments,
        "n_tokens": len(tokens),
        "schema_valid": is_valid(report),
        "segment_representation": cfg.segment_source.as_dict() if cfg.segment_source else None,
        "domain_gate": gate.as_dict(),
        "sequence_sufficiency": suff.as_dict() if suff else None,
    }

    result = PipelineResult(
        report=report,
        trace=trace,
        manifest=manifest,
        domain_gate=gate.as_dict(),
        segments=segments_records,
        dual_stream=[e.as_event() for e in events],
        sequences=sequences,
        diagnostics=diagnostics,
        config=cfg.as_dict(),
    )
    if out_dir is not None:
        from bioacoustic_pipeline.pipeline.artifacts import ArtifactWriter  # noqa: PLC0415

        result.artifacts = ArtifactWriter.write(result, out_dir)
    return result
