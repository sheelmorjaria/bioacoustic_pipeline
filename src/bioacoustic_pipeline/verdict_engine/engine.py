"""Verdict engine — assembles the schema-conformant Evidence + Falsification report.

Stitches the evidence profile (6 axes) and the falsification tests into the
single canonical output dict defined by ``schema.py``. Every report is
self-validating: ``build_report`` raises if the assembled output violates the
schema (so upstream modules can never emit a malformed verdict).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

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
    semantic_status: str = "none",
) -> dict:
    """Build + self-validate the canonical verdict report.

    Raises ValueError if the assembled report violates the schema (e.g. a
    pure-computational report that claims validated semantics).
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
        semantic_status=semantic_status,
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
    errors = validate_report(report)
    if errors:
        raise ValueError("verdict report violates schema: " + "; ".join(errors))
    return report
