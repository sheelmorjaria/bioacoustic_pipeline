"""Verdict engine — assembles the schema-conformant Evidence + Falsification report.

Stitches the evidence profile (6 axes) and the falsification tests into the
single canonical output dict defined by ``schema.py``. Every report is
self-validating: ``build_report`` raises if the assembled output violates the
schema (so upstream modules can never emit a malformed verdict).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from bioacoustic_pipeline.diagnostics.sequence_sufficiency import SufficiencyResult
from bioacoustic_pipeline.verdict_engine.evidence_profile import (
    EvidenceProfile,
    compute_evidence_profile,
)
from bioacoustic_pipeline.verdict_engine.falsification import falsification_tests
from bioacoustic_pipeline.verdict_engine.schema import SCHEMA_VERSION, validate_report


def build_report(
    species: str,
    *,
    lz_normalized: float,
    alias_density: float = 0.0,
    alias_pvalue: float | None = None,
    n_hard: int = 0,
    n_soft: int = 0,
    n_total_boundaries: int = 0,
    n_context_bound: int = 0,
    n_turn_bigrams: int = 0,
    entrained: bool | None = None,
    token_coherence: float | None = None,
    alias_acoustic_divergence: float | None = None,
    n_motifs: int | None = None,
    motif_null_p: float | None = None,
    semantic_status: str = "none",
    sufficiency: SufficiencyResult | None = None,
    context_evaluable: bool = True,
    context_reason: str | None = None,
    turn_evaluable: bool = True,
    turn_reason: str | None = None,
    domain_gate=None,
) -> dict:
    """Build + self-validate the canonical verdict report.

    Raises ValueError if the assembled report violates the schema (e.g. a
    pure-computational report that claims validated semantics).

    ``domain_gate`` (a ``diagnostics.domain_gate.DomainGateResult`` or None): when the
    corpus is rejected for bioacoustic-communication inference, the ethological axes go
    ``not_applicable`` and a ``domain`` block is attached.
    """
    profile = compute_evidence_profile(
        species,
        lz_normalized=lz_normalized,
        alias_density=alias_density,
        alias_pvalue=alias_pvalue,
        n_hard=n_hard,
        n_soft=n_soft,
        n_total_boundaries=n_total_boundaries,
        n_context_bound=n_context_bound,
        n_turn_bigrams=n_turn_bigrams,
        entrained=entrained,
        token_coherence=token_coherence,
        alias_acoustic_divergence=alias_acoustic_divergence,
        n_motifs=n_motifs,
        motif_null_p=motif_null_p,
        semantic_status=semantic_status,
        sufficiency=sufficiency,
        context_evaluable=context_evaluable,
        context_reason=context_reason,
        turn_evaluable=turn_evaluable,
        turn_reason=turn_reason,
        domain_gate=domain_gate,
    )
    return report_from_profile(profile)


def report_from_profile(profile: EvidenceProfile) -> dict:
    """Wrap an EvidenceProfile (+ falsification tests) into the schema report."""
    pd = profile.as_dict()
    report = {
        "schema_version": SCHEMA_VERSION,
        "species": pd["species"],
        "overall_confidence": pd["overall_confidence"],
        "validated_semantics": pd["validated_semantics"],
        "axes": pd["axes"],
        "falsification": falsification_tests(profile),
    }
    if pd.get("domain") is not None:
        report["domain"] = pd["domain"]
    errors = validate_report(report)
    if errors:
        raise ValueError("verdict report violates schema: " + "; ".join(errors))
    return report
