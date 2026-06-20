"""Pipeline orchestration: raw audio -> schema-valid verdict report.

The integration seam. Two layers:

* ``synthesize_signals`` / ``run_report`` reduce *already-tokenized* streams to
  the verdict engine's primitive signals (kept for the tokens-file CLI + tests).
* ``segment_audio`` / ``run_pipeline`` turn *real audio* into a report: load +
  NBD-segment (calibrated if a checkpoint exists, else fixed-threshold fallback)
  -> firewall-gated VQ-VAE tokenization per segment -> diagnostics -> verdict.
  Every stage reports what ran vs. what was skipped, mirroring
  ``embeddings.bank``'s pattern; nothing is silently dropped.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity
from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency
from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles
from bioacoustic_pipeline.segmentation.nbd_calibrator import (
    BoundaryCalibrator,
    proposal_context,
)
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals
from bioacoustic_pipeline.tokenization.multiview import AcousticTokenizer
from bioacoustic_pipeline.utils.audio import load_audio
from bioacoustic_pipeline.verdict_engine import build_report, is_valid

# orchestration.py -> bioacoustic_pipeline -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical proposal-generation params for the CALIBRATED path. These MUST match
# what the calibrator was trained on (refit_nbd_calibrator.py uses the same), so
# inference sees the same proposal distribution as training. Low floors + tight
# debounce over-generate candidates; the calibrator decides which are real.
CALIBRATOR_PROPOSAL = {"floor_energy": 0.04, "floor_flux": 0.02, "debounce_ms": 25.0}


def synthesize_signals(tokens, boundary_types, sequences=None, with_null=False) -> dict:
    """Reduce raw stage outputs to the verdict engine's primitive signals."""
    lz = lempel_ziv_complexity(list(tokens)[:5000]).normalized
    n_hard = sum(1 for b in boundary_types if b == "hard")
    n_soft = sum(1 for b in boundary_types if b == "soft")

    alias_density, alias_pvalue = 0.0, None
    if sequences and len(sequences) > 5:
        from bioacoustic_pipeline.diagnostics.alias_detector import (
            detect_functional_aliases,
        )

        rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
        eligible = len(rep.singleton_tokens) + rep.n_merged
        alias_density = rep.n_merged / max(eligible, 1)
        if with_null:
            from bioacoustic_pipeline.diagnostics.null_models import (
                permutation_test,
                within_file_shuffle,
            )

            def _density(seqs):
                r = detect_functional_aliases(
                    seqs, similarity_threshold=0.80, min_instances=10
                )
                e = len(r.singleton_tokens) + r.n_merged
                return r.n_merged / max(e, 1)

            alias_pvalue = permutation_test(
                sequences, _density, within_file_shuffle, n_permutations=100
            ).p_value

    return {
        "lz_normalized": lz,
        "n_hard": n_hard,
        "n_soft": n_soft,
        "n_total_boundaries": n_hard + n_soft,
        "alias_density": alias_density,
        "alias_pvalue": alias_pvalue,
    }


def run_report(species, tokens, boundary_types, sequences=None, with_null=False) -> dict:
    """Assemble + self-validate the canonical verdict report."""
    return build_report(species, **synthesize_signals(
        tokens, boundary_types, sequences, with_null
    ))


# --------------------------------------------------------------------------- #
# Raw-audio end-to-end path
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    """One audio slice between consecutive NBD boundaries + its boundary type."""

    start_ms: float
    end_ms: float
    boundary_type: str  # hard | soft | transitional
    confidence: float
    audio: np.ndarray
    sr: int
    # Opt-in physics-grounded descriptor (acoustic_fingerprint.SegmentFingerprint);
    # None unless run_pipeline(fingerprint=True).
    fingerprint=None


def _detector_to_boundary_type(detector: str) -> str:
    """Map NBD detector tag -> boundary type.

    Mirrors ``segmentation.nbd_calibrator._boundary_type`` (energy->hard,
    spectral->soft, both->transitional); inlined here to avoid importing a
    private name.
    """
    if detector == "energy":
        return "hard"
    if detector in ("spectral", "spectral_drop"):
        return "soft"
    return "transitional"


def _default_calibrator(species: str) -> BoundaryCalibrator | None:
    """Auto-load checkpoints/nbd_calibrator/{species}.pkl if present + fitted."""
    p = REPO_ROOT / "checkpoints" / "nbd_calibrator" / f"{species}.pkl"
    if p.exists():
        try:
            cal = BoundaryCalibrator.load(p)
            if cal.fitted:
                return cal
        except Exception:  # noqa: BLE001 — a corrupt checkpoint falls back to threshold
            return None
    return None


def segment_audio(
    audio: np.ndarray, sr: int, profile: dict, calibrator: BoundaryCalibrator | None = None,
    cpc=None,
) -> list[Segment]:
    """Raw audio -> calibrated Segments via NBD proposal + (calibrator | threshold).

    With a fitted ``calibrator`` (the NBD calibrator trained on human forced
    alignment) proposals are gated by P(boundary); otherwise we fall back to the
    fixed-threshold NBD using the species profile's energy/spectral thresholds
    (the behavior the calibrator was designed to replace, kept honest by the
    trace's ``calibrator_fitted`` flag). Segments are the intervals between
    consecutive detected boundaries; pre/post-roll outside the first/last
    boundary is dropped (typically silence).

    The first boundary of each interval defines the segment's ``boundary_type``
    and ``confidence``; one boundary_type per segment is exactly what
    ``synthesize_signals`` consumes.
    """
    hop_ms = profile.get("hop_ms", 10.0)
    debounce_ms = profile.get("debounce_ms", 30.0)
    signals = compute_signals(audio, sr, hop_ms=hop_ms)

    if calibrator is not None and getattr(calibrator, "fitted", False):
        # Train/inference consistency: generate proposals with the SAME params the
        # calibrator was refit on, then gate via its tuned keep_threshold.
        proposals = generate_proposals(signals, **CALIBRATOR_PROPOSAL)
        ctx = None
        if getattr(calibrator, "_use_context", False):
            if cpc is None:
                raise ValueError(
                    "calibrator was fit with CPC learned-context features; "
                    "pass cpc=<CPCBackbone> to segment_audio()"
                )
            ctx = proposal_context(proposals, cpc.embed_frames(audio, sr))
        kept, _abstained = calibrator.apply(proposals, context=ctx)
        kept.sort(key=lambda b: b.time_ms)
        times_types = [(b.time_ms, b.boundary_type, b.confidence) for b in kept]
    else:
        proposals = generate_proposals(
            signals,
            floor_energy=profile.get("energy_threshold", 0.05),
            floor_flux=profile.get("spectral_threshold", 0.03),
            floor_drop=profile.get("spectral_drop_threshold", float("inf")),
            debounce_ms=debounce_ms,
        )
        times_types = [
            (p.time_ms, _detector_to_boundary_type(p.detector), 1.0) for p in proposals
        ]

    min_samples = max(int(sr * 0.01), 1)  # drop sub-10ms slivers
    segments: list[Segment] = []
    for (t0, bt0, c0), (t1, _bt1, _c1) in zip(times_types, times_types[1:]):
        s = int(t0 / 1000.0 * sr)
        e = int(t1 / 1000.0 * sr)
        if e - s < min_samples:
            continue
        segments.append(Segment(t0, t1, bt0, c0, audio[s:e], sr))
    return segments


def run_pipeline(
    audio_paths,
    species: str,
    *,
    profiles: dict | None = None,
    calibrator: BoundaryCalibrator | None | bool = None,
    tokenizer=None,
    bank=None,
    cpc=None,
    firewall=None,
    cache=None,
    with_null: bool = False,
    fingerprint: bool = False,
) -> tuple[dict, dict]:
    """Run raw audio files -> (schema-valid verdict report, execution trace).

    Stages: load (species sample rate) -> NBD segment -> firewall-gated VQ-VAE
    tokenize per segment -> diagnostics -> verdict. Tokenization is refused
    cross-species by the firewall (e.g. the human VQ-VAE on bat audio); when that
    happens — or no species tokenizer is available — token-based axes stay
    degenerate and the trace says why, but a schema-valid report still emits.

    Args:
        audio_paths: iterable of audio file paths.
        species: target species (key in species_profiles.yaml + the firewall).
        calibrator: a fitted ``BoundaryCalibrator``, ``None`` to auto-load
            ``checkpoints/nbd_calibrator/{species}.pkl``, or ``False`` to force
            the fixed-threshold fallback.
        tokenizer: defaults to ``AcousticTokenizer()``; inject a fake in tests.
        bank: optional ``EmbeddingBank``; if given, segments are embedded to warm
            the cache and ``bank.report()`` is recorded (embeddings are not yet
            consumed by the verdict axes — plumbed for the multi-view/probe work).
        cache: forwarded to the bank only if you construct it yourself.
    """
    fw = firewall or default_firewall()
    all_profiles = profiles or load_species_profiles()
    if species not in all_profiles:
        raise ValueError(f"unknown species {species!r}; known: {sorted(all_profiles)}")
    profile = all_profiles[species]

    # Calibrator: explicit > auto-load {species}.pkl > None (threshold fallback).
    if calibrator is False:
        calibrator = None
    elif calibrator is None:
        calibrator = _default_calibrator(species)
    cal_fitted = bool(calibrator is not None and getattr(calibrator, "fitted", False))
    # If the calibrator uses learned CPC context, supply a CPC backbone for inference.
    if cal_fitted and getattr(calibrator, "_use_context", False) and cpc is None:
        from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
        cpc = CPCBackbone()
        if not cpc.available:
            raise RuntimeError(
                "calibrator needs CPC context features but CPCBackbone is unavailable"
            )

    # Tokenizer: default AcousticTokenizer, firewall-gated for the target species.
    if tokenizer is None:
        tokenizer = AcousticTokenizer()
    tk_available = bool(getattr(tokenizer, "available", False))
    tk_skip: str | None = None
    if tk_available:
        try:
            fw.assert_allowed(tokenizer.provenance_name, Use.TOKENIZE)
            fw.assert_cross_species(tokenizer.provenance_name, species)
        except Exception as exc:  # noqa: BLE001 — firewall refusal -> skip tokens
            tk_available = False
            tk_skip = f"firewall: {exc}"

    tokens: list[int] = []
    boundary_types: list[str] = []
    sequences: list[list[int]] = []
    token_fps: list[tuple[int, np.ndarray]] = []  # (token, fingerprint.combined)
    n_files = n_loaded = n_segments = n_fingerprinted = 0

    for path in audio_paths:
        n_files += 1
        try:
            audio, sr = load_audio(str(path), target_sr=profile.get("sample_rate"))
        except Exception:  # noqa: BLE001 — QA skips unreadable files (cf. build_manifest)
            continue
        n_loaded += 1

        segs = segment_audio(audio, sr, profile, calibrator=calibrator, cpc=cpc)
        n_segments += len(segs)

        if fingerprint:  # physics-grounded descriptor, post-segmentation / pre-tokenization
            from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
            for seg in segs:
                seg.fingerprint = afp.extract(
                    seg.audio, seg.sr, start_ms=seg.start_ms, end_ms=seg.end_ms,
                    species=species, boundary_type=seg.boundary_type,
                    boundary_confidence=seg.confidence,
                )
                n_fingerprinted += 1

        if bank is not None:
            for seg in segs:
                try:
                    bank.views(seg.audio, seg.sr)  # populate embedding cache
                except Exception:  # noqa: BLE001 — embedding failure must not abort the verdict
                    pass

        if not tk_available:
            continue

        seq: list[int] = []
        for seg in segs:
            tok = tokenizer.tokenize(seg.audio, seg.sr)
            if tok == -1:  # segment too short for CPC framing
                continue
            tokens.append(tok)
            boundary_types.append(seg.boundary_type)
            seq.append(tok)
            if fingerprint:
                token_fps.append((tok, np.asarray(seg.fingerprint.combined)))
        sequences.append(seq)

    # Acoustic-fingerprint evidence channel (opt-in): token coherence + alias divergence.
    afp_trace = None
    if fingerprint:
        from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
        token_to_fp: dict[int, list[np.ndarray]] = {}
        for tok, fpv in token_fps:
            token_to_fp.setdefault(tok, []).append(fpv)
        afp_trace = {"n_fingerprinted_tokens": len(token_fps)}
        if token_to_fp:
            tc = afp.token_fingerprint_coherence(token_to_fp)
            cohs = [v["cohesion"] for v in tc.values()]
            afp_trace["token_coherence_mean"] = round(float(np.mean(cohs)), 4) if cohs else 0.0
            afp_trace["token_coherence_null_p"] = round(
                float(afp.token_coherence_null(token_to_fp, n_permutations=100).p_value), 4)
        if sum(len(s) for s in sequences) > 50 and token_to_fp:
            try:
                from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
                    detect_functional_aliases,
                )
                rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
                cents = {t: np.mean(v, axis=0) for t, v in token_to_fp.items()}
                groups = {g.group_id: g.member_tokens for g in rep.groups}
                if groups:
                    ad = afp.alias_acoustic_divergence(groups, cents)
                    divs = [v["divergence"] for v in ad.values()]
                    afp_trace["alias_divergence_mean"] = round(float(np.mean(divs)), 4) if divs else 0.0
                    afp_trace["alias_divergence_null_p"] = round(
                        float(afp.alias_divergence_null(groups, cents, n_permutations=100).p_value), 4)
            except Exception:  # noqa: BLE001 — alias path is best-effort
                pass

    # Motif Miner: token n-gram motifs + alias-collapsed motifs + null (when sequences exist)
    motif_trace = None
    if sequences and sum(len(s) for s in sequences) > 10:
        from bioacoustic_pipeline.diagnostics import motif_miner as mm  # noqa: PLC0415
        raw = mm.mine_ngrams(sequences, min_n=2, max_n=5, min_support=3)
        token_motifs = mm.build_motifs(raw, sequences, motif_type="token")
        motif_trace = {
            "n_token_motifs": len(token_motifs),
            "top_token_motifs": [m.as_dict() for m in token_motifs[:3]],
        }
        # alias-collapsed motifs (if alias detection produces groups)
        if sum(len(s) for s in sequences) > 50:
            try:
                from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
                    detect_functional_aliases,
                )
                rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
                if rep.groups:
                    alias_map = {}
                    for g in rep.groups:
                        for t in g.member_tokens:
                            alias_map[t] = f"A{g.group_id}"
                    am, gain = mm.mine_alias_motifs(sequences, alias_map, min_support=3)
                    motif_trace["n_alias_motifs"] = len(am)
                    motif_trace["alias_gain"] = round(gain, 4)
                    motif_trace["top_alias_motifs"] = [m.as_dict() for m in am[:3]]
            except Exception:  # noqa: BLE001
                pass
        # null significance
        try:
            null = mm.motif_support_null(sequences, n_permutations=100, seed=0)
            motif_trace["motif_null_p"] = round(float(null.p_value), 4)
        except Exception:  # noqa: BLE001
            pass

    signals = synthesize_signals(tokens, boundary_types, sequences, with_null)
    if afp_trace:
        signals["token_coherence"] = afp_trace.get("token_coherence_mean")
        signals["alias_acoustic_divergence"] = afp_trace.get("alias_divergence_mean")
    if motif_trace:
        signals["n_motifs"] = motif_trace.get("n_token_motifs")
        signals["motif_null_p"] = motif_trace.get("motif_null_p")

    # Sequence-sufficiency gate: sequence diagnostics ABSTAIN (not_evaluable) when the
    # token sequences are too sparse to test — e.g. a conservative NBD profile that
    # collapses bouts to ~1 token. Prevents a false biological null.
    sufficiency = evaluate_sufficiency(sequences)
    report = build_report(species, sufficiency=sufficiency, **signals)

    trace = {
        "species": species,
        "n_files": n_files,
        "n_files_loaded": n_loaded,
        "n_segments": n_segments,
        "n_tokens": len(tokens),
        "tokenizer_available": tk_available,
        "tokenizer_skipped_reason": tk_skip,
        "calibrator_fitted": cal_fitted,
        "bank_report": bank.report() if bank is not None else None,
        "with_null": with_null,
        "acoustic_fingerprint": afp_trace,
        "motif_report": motif_trace,
        "sequence_sufficiency": sufficiency.as_dict(),
        "schema_valid": is_valid(report),
    }
    return report, trace
