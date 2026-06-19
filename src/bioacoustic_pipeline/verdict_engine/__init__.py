"""Verdict engine: evidence profile + falsification report (schema-conformant)."""

from bioacoustic_pipeline.verdict_engine.evidence_profile import (
    AxisResult,
    EvidenceProfile,
    compute_evidence_profile,
    score_call_unitization,
    score_context_binding,
    score_functional_aliasing,
    score_semantic_validation,
    score_sequence_recombination,
    score_turn_structure,
)
from bioacoustic_pipeline.verdict_engine.engine import build_report, report_from_profile
from bioacoustic_pipeline.verdict_engine.falsification import (
    falsification_tests,
    format_falsification_report,
)
from bioacoustic_pipeline.verdict_engine.schema import (
    AXIS_LEVELS,
    AXIS_NAMES,
    SCHEMA_VERSION,
    is_valid,
    validate_report,
)

__all__ = [
    "AxisResult",
    "EvidenceProfile",
    "compute_evidence_profile",
    "score_call_unitization",
    "score_context_binding",
    "score_functional_aliasing",
    "score_semantic_validation",
    "score_sequence_recombination",
    "score_turn_structure",
    "build_report",
    "report_from_profile",
    "falsification_tests",
    "format_falsification_report",
    "AXIS_LEVELS",
    "AXIS_NAMES",
    "SCHEMA_VERSION",
    "validate_report",
    "is_valid",
]
